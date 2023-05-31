from copy import copy
from pathlib import Path
import sys
from typing import Optional
from tempfile import gettempdir
import numpy as np
from tqdm import tqdm
from biotite.structure.io.pdbx import PDBxFile, get_structure, get_sequence
import biotite.structure as biostruc
import biotite.database.rcsb as rcsb
from biotite.sequence import ProteinSequence
from src.data.fasta import Fasta
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils import align_sequences_nw
import argparse
import multiprocessing as mp
import os

def calculate_scores_for_protein(protein: str, pdb_path: str, map_missing_res: list[str]) -> tuple:
    """
    Calculates the SASA and B-factor scores for a given protein. 
    The SASA and B-factor scores are calculated for each residue in the protein.
    Parameters
    ----------
    protein : str the protein name in the format <PDB ID>-<chain ID> Note that the character '-' is not allowed not the minus sign.
    pdb_path : str the path to the PDB files
    map_missing_res : list[str] a list of the same length as the protein sequence, where each element is either '-' or 'X'.
    Returns
    -------
    protein : str the protein name in the format <PDB ID>-<chain ID> Note that the character '-' is not allowed not the minus sign.
    res_sasa_masked : np.array the SASA scores for each residue in the protein
    res_bfactor_masked : np.array the B-factor scores for each residue in the protein
    """
    cif_header: str = protein.split('-')[0]

    try:
        pdbx = PDBxFile.read(os.path.join(pdb_path, f'{cif_header}.cif'))
    except FileNotFoundError:
        print(f'Could not find PDBx file for {protein}\nFetching from RCSB...')
        file_path = rcsb.fetch(cif_header, "cif")
        pdbx = PDBxFile.read(file_path)
    struct = get_structure(pdbx, model=1, extra_fields=["b_factor"])
    seq = get_sequence(pdbx)[0]
    seq_wo_X = str(seq).replace('X', '')
    seq_length = len(seq)
    chain_id = protein.split('-')[1]
    chain_starts = biostruc.get_chain_starts(struct).tolist()
    chain_ids = biostruc.get_chains(struct).tolist()
    if biostruc.get_chain_count(struct) == 1 or chain_starts[chain_ids.index(chain_id)] == chain_starts[-1]:
        struct = struct[chain_starts[chain_ids.index(chain_id)]:]
    else:
        struct = struct[chain_starts[chain_ids.index(chain_id)]:chain_starts[chain_ids.index(chain_id) + 1]]
    
    struct = struct[biostruc.filter_canonical_amino_acids(struct)]
    atom_sasa_scores = biostruc.sasa(struct, vdw_radii="Single", point_number=500)

    res_sasa = biostruc.apply_residue_wise(struct, atom_sasa_scores, np.nansum)
    res_bfactor = biostruc.apply_residue_wise(struct, struct.get_annotation("b_factor"), np.nansum)

    # clip so the mask can be recognized by the model
    # divide by 100 to get smaller values -> better for the model gradient 
    res_sasa = res_sasa.clip(0.00001) / 100
    res_bfactor = res_bfactor.clip(0.00001) / 100

    # mask the residues that are not in the PDB files, due to disorder
    disorder_residues = list("".join(map_missing_res))
    non_disorder_indices = [i for i, x in enumerate(disorder_residues) if x == "-"]
    
    # this is not clean and could lead to errors
    if len(disorder_residues) != res_sasa.shape[0] or res_sasa.shape[0] != seq_length:
        res_sasa_masked = np.zeros(len(disorder_residues))
        try:
            res_sasa_masked[non_disorder_indices] = res_sasa
        except ValueError:
            """
            Essentially, this is a hack to deal with the fact that the PDB file contains more residues than in the primary sequence, 
            even when only considering the residues that are not disordered.
            So if this is the case, this piece of code aligns the primary sequence against the residue sequence in the PDB file.
            This deals only with the case that the PDB sequence is longer than the primary sequence !!!
            """
            seq_chain_a_single = []
            for aa in biostruc.get_residues(struct)[1]:
                try:
                    seq_chain_a_single.append(ProteinSequence.convert_letter_3to1(aa))
                except KeyError:
                    if aa == 'PYL':
                        seq_chain_a_single.append('O')
                    elif aa == 'SEC':
                        seq_chain_a_single.append('U')
                    else:
                        seq_chain_a_single.append('X')
            alignment = align_sequences_nw(seq[non_disorder_indices], "".join(seq_chain_a_single))
            primary_seq_overlap = np.array(list(alignment[0])) != '-'
            res_sasa_masked[non_disorder_indices] = res_sasa[primary_seq_overlap] 
            res_bfactor = res_bfactor[primary_seq_overlap]
        

        res_bfactor_masked = np.zeros(len(disorder_residues))
        res_bfactor_masked[non_disorder_indices] = res_bfactor

        return protein, res_sasa_masked, res_bfactor_masked
    return protein, res_sasa, res_bfactor


def calculate_scores(fasta_file: Fasta, pdb_path: str, nprocesses: int, mapping_fasta) -> tuple[dict, dict]:
    """
    Calculates the SASA and B-factor scores for every protein in the fasta file. 
    The SASA and B-factor scores are calculated for each residue in the protein.
    Parameters
    ----------
    fasta_file : Fasta the fasta file containing the protein sequences
    pdb_path : str the path to the PDB files
    nprocesses : int the number of processes to use for multiprocessing
    Returns
    -------
    sasa_scores : dict the SASA scores for each residue in the protein
    bfactor_scores : dict the B-factor scores for each residue in the protein
    """
    proteins = fasta_file.get_headers()
    with mp.Pool(nprocesses) as pool:
        results = [pool.apply_async(calculate_scores_for_protein, 
                                    args=(protein, pdb_path, mapping_fasta[":".join((protein.upper() + "-disorder").split("-"))])) for protein in proteins]
        results = [r.get() for r in tqdm(results)]
    sasa_scores = {protein: sasa_scores for protein, sasa_scores, _ in results}
    bfactor_scores = {protein: bfactor_scores for protein, _, bfactor_scores in results}
    return sasa_scores, bfactor_scores



def main(args: Optional[list] = None):
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--fasta_files', nargs='+', help='Path(s) to fasta files')
    parser.add_argument('-p', '--pdb_path', required=True, help='Path to PDB structures')
    parser.add_argument("-m", "--mapping_file", required=True, help="Path to mapping file, which is required to fill in missing residues")
    parser.add_argument('-o', '--output_path', required=True, help='Output path')
    parser.add_argument('-n', '--n_processes', default=16, help='Number of processes to use')

    # Parse arguments
    if args is None:
        args = parser.parse_args()
    # Access arguments

    fasta_files = args.fasta_files
    pdb_path = args.pdb_path
    mapping_file = args.mapping_file
    output_path = args.output_path
    mapping_fasta = Fasta(mapping_file)
    for fasta_path in fasta_files:
        fasta = Fasta(fasta_path)
        sasa_scores, bfactor_scores = calculate_scores(fasta, pdb_path, args.n_processes, mapping_fasta)
        copy(fasta).append(bfactor_scores).write_fasta(f'{output_path}/bfactor/{Path(fasta_path).stem}_bfactor.fasta')
        fasta.append(sasa_scores).write_fasta(f'{output_path}/sasa/{Path(fasta_path).stem}_sasa.fasta')




if __name__ == '__main__':
    main()
