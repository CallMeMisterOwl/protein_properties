from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Optional
from tempfile import gettempdir
import numpy as np
from tqdm import tqdm
from biotite.structure.io.pdbx import PDBxFile, get_structure, get_sequence
import biotite.structure as biostruc
import biotite.database.rcsb as rcsb
from biotite.sequence import ProteinSequence, AlphabetError
from .fasta import Fasta
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils import align_sequences_nw
import argparse
import multiprocessing as mp
import os

# TODO find a way to automatically substitute non-generic amino acid with generic ones 
aa_dict = None

def calculate_scores_for_protein(protein: str, 
                                 pdb_path: str,
                                 map_missing_res: list[str], 
                                 protein_seq: list, 
                                 sub_dict: Optional[dict] = None) -> tuple:
    """
    Calculates the SASA and B-factor scores for a given protein. 
    The SASA and B-factor scores are calculated for each residue in the protein.
    
    Parameters
    ----------
    protein : str the protein name in the format <PDB ID>-<chain ID> Note that the character '-' is not the minus sign.
    pdb_path : str the path to the PDB files
    map_missing_res : list[str] a list of the same length as the protein sequence, where each element is either '-' or 'X'.
    protein_seq : list[str] the protein sequence in the one-letter amino acid code.
    sub_dict : dict a dictionary that maps non-generic amino acids to generic ones. In case this function is executed outside the script, a substitution dictionary must be provided.
    
    Returns
    -------
    protein : str the protein name in the format <PDB ID>-<chain ID> Note that the character '-' is not the minus sign.
    res_sasa_masked : np.array the SASA scores for each residue in the protein
    res_bfactor_masked : np.array the B-factor scores for each residue in the protein
    """
    cif_header: str = protein.split('-')[0]
    global aa_dict
    if sub_dict is None and not aa_dict:
        raise ValueError("No substitution dictionary provided!\nIf you import this function from another script, please provide a substitution dictionary")
    elif sub_dict is not None:
        aa_dict = sub_dict
        

    try:
        pdbx = PDBxFile.read(os.path.join(pdb_path, f'{cif_header}.cif'))
    except FileNotFoundError:
        print(f'Could not find PDBx file for {protein}\nFetching from RCSB...')
        file_path = rcsb.fetch(cif_header, "cif")
        pdbx = PDBxFile.read(file_path)
    struct = get_structure(pdbx, model=1, extra_fields=["b_factor"])
    # Thank you biotite for this wonderful class, NOT!
    seq = ProteinSequence(list(("".join(protein_seq)).replace('U', 'C').replace("O", "K")))

    seq_length = len(seq)
    chain_id = protein.split('-')[1]
    chain_starts = biostruc.get_chain_starts(struct).tolist()
    chain_ids = biostruc.get_chains(struct).tolist()
    if biostruc.get_chain_count(struct) == 1 or chain_starts[chain_ids.index(chain_id)] == chain_starts[-1]:
        struct = struct[chain_starts[chain_ids.index(chain_id)]:]
    else:
        struct = struct[chain_starts[chain_ids.index(chain_id)]:chain_starts[chain_ids.index(chain_id) + 1]]
    
    struct = struct[biostruc.filter_amino_acids(struct)]
    atom_sasa_scores = biostruc.sasa(struct, vdw_radii="Single", point_number=500)

    res_sasa = biostruc.apply_residue_wise(struct, atom_sasa_scores, np.nansum)
    res_bfactor = biostruc.apply_residue_wise(struct, struct.get_annotation("b_factor"), np.nansum)

    # clip so the mask can be recognized by the model
    # divide by 100 to get smaller values -> better for the model gradient 
    res_sasa = res_sasa.clip(0.00001)
    res_bfactor = res_bfactor.clip(0.00001)

    # mask the residues that are not in the PDB files, due to disorder
    disorder_residues = list("".join(map_missing_res))
    non_disorder_indices = [i for i, x in enumerate(disorder_residues) if x == "-"]
    
    if len(disorder_residues) != res_sasa.shape[0]:
        res_sasa_masked = np.zeros(len(disorder_residues))
        res_bfactor_masked = np.zeros(len(disorder_residues))

        """
        see if the per residue SASA and B-factor scores can be mapped to the primary 
        sequence when using only non-disordered residues
        """
        try:
            
            res_sasa_masked[non_disorder_indices] = res_sasa
            res_bfactor_masked[non_disorder_indices] = res_bfactor
        except ValueError:
            seq_chain_a_single = []
            for aa in biostruc.get_residues(struct)[1]:
                try:
                    seq_chain_a_single.append(ProteinSequence.convert_letter_3to1(aa))
                except KeyError:
                    try:
                        seq_chain_a_single.append(aa_dict[aa])
                    except KeyError:
                        seq_chain_a_single.append('X')
            # TODO shouldn't this be just seq against seq_chain_a_single ?
            alignment = align_sequences_nw(seq, "".join(seq_chain_a_single))
            primary_seq_overlap = np.array(list(alignment[0])) != '-'
            seq_chain_overlap = np.array(list(alignment[1])) != '-'

            """
            Case 1 - the alignment has gaps in the primary sequence and in the sequence from the PDB file
            Solution - remove the part of the sequence that causes the gap in the primary sequence, 
            afterwards treat it like case 2

            Case 2 - the alignment has gaps in the sequence from the PDB file, but not in the primary sequence
            Solution - mask the residues that are not in the PDB file, but are in the primary sequence    
            
            Case 3 - the alignment has gaps in the primary sequence, but not in the sequence from the PDB file
            Solution - mask the residues that are not in the primary sequence, but are in the PDB file
            Not sure if this case is possible, but it's here just in case   
            """
            if np.any(primary_seq_overlap == False) and np.any(seq_chain_overlap == False):
                seq_chain_overlap_cut = seq_chain_overlap[primary_seq_overlap]
                # TODO remove the part that causes the gap from SASA and B-factor arrays
                sasa_index = []
                counter = 0
                for i in range(len(primary_seq_overlap)):
                    if seq_chain_overlap[i] == True and primary_seq_overlap[i] != False:
                        sasa_index.append(counter)
                        counter += 1
                try:
                    res_sasa_masked[seq_chain_overlap_cut] = res_sasa[sasa_index]
                except IndexError as i:
                    print(f'Skipping protein {protein}...\n')
                    print(i)
                    return protein, None, None
                res_bfactor_masked[seq_chain_overlap_cut] = res_bfactor[sasa_index]

            elif np.any(seq_chain_overlap == False):
                try:
                    res_sasa_masked[seq_chain_overlap] = res_sasa
                except IndexError as i:
                    print(f'Skipping protein {protein}...\n')
                    print(i)
                    return protein, None, None
                res_bfactor_masked[seq_chain_overlap] = res_bfactor
            
            elif np.any(primary_seq_overlap == False):
                try:
                    res_sasa_masked = res_sasa[primary_seq_overlap]
                except IndexError as i:
                    print(f'Skipping protein {protein}...\n')
                    print(i)
                    return protein, None, None
                res_bfactor_masked = res_bfactor[primary_seq_overlap]
            
            else:
                print(f"Investiage protein {protein}")
                return protein, None, None
        return protein, res_sasa_masked.tolist(), res_bfactor_masked.tolist()
    return protein, res_sasa.tolist(), res_bfactor.tolist()


def calculate_scores(fasta_file: Fasta, pdb_path: str, nprocesses: int, mapping_fasta: Fasta) -> tuple[dict, dict]:
    """
    Calculates the SASA and B-factor scores for every protein in the fasta file. 
    The SASA and B-factor scores are calculated for each residue in the protein.
    
    Parameters
    ----------
    fasta_file (Fasta):  the fasta file containing the protein sequences
    pdb_path (str):  the path to the PDB files
    nprocesses (int):  the number of processes to use for multiprocessing
    mapping_fasta (Fasta):  the fasta file containing the mapping between the primary sequence and disorder/ordered residues. In addition contains the sequence and secondary structure of the protein.
    
    Returns
    -------
    sasa_scores (dict):  the SASA scores for each residue in the protein
    bfactor_scores (dict):  the B-factor scores for each residue in the protein
    """
    
    proteins = fasta_file.get_headers()
    with mp.Pool(int(nprocesses)) as pool:
        results = [pool.apply_async(calculate_scores_for_protein, 
                                    args=(protein, pdb_path, 
                                          mapping_fasta[":".join((protein.upper() + "-disorder").split("-"))], 
                                          mapping_fasta[":".join((protein.upper() + "-sequence").split("-"))])) for protein in proteins]
        results = [r.get() for r in tqdm(results)]
    sasa_scores = {protein: [sasa_scores] for protein, sasa_scores, _ in results}
    bfactor_scores = {protein: [bfactor_scores] for protein, _, bfactor_scores in results}
    return sasa_scores, bfactor_scores



def main(args: Optional[list] = None):
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--fasta_files', nargs='+', help='Path(s) to fasta files')
    parser.add_argument('-p', '--pdb_path', required=True, help='Path to PDB structures')
    parser.add_argument("-m", "--mapping_file", required=True, help="Path to mapping file, which is required to fill in missing residues")
    parser.add_argument('-o', '--output_path', required=True, help='Output path')
    parser.add_argument('-n', '--n_processes', default=16, help='Number of processes to use', type=int)

    # Parse arguments
    if args is None:
        args = parser.parse_args()
    # Access arguments

    fasta_files = args.fasta_files
    pdb_path = args.pdb_path
    mapping_file = args.mapping_file
    output_path = args.output_path
    mapping_fasta = Fasta(mapping_file)
    global aa_dict
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/substitution_dict.json"), "r") as f:
        aa_dict = json.load(f)
        
    for fasta_path in fasta_files:
        fasta = Fasta(fasta_path)
        sasa_scores, bfactor_scores = calculate_scores(fasta, pdb_path, args.n_processes, mapping_fasta)
        deepcopy(fasta).append(bfactor_scores).write_fasta(f'{output_path}/bfactor/{Path(fasta_path).stem}_bfactor.fasta', overwrite=True)
        fasta.append(sasa_scores).write_fasta(f'{output_path}/sasa/{Path(fasta_path).stem}_sasa.fasta', overwrite=True)




if __name__ == '__main__':
    main()
