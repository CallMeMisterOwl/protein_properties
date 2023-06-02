from copy import copy
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

aa_dict = {"PYL": "K", "SEC": "C", "AIB": "A", "PHL": "F", "DPR": "P", "DBZ": "A", "DAL": "A", "MLY": "K"}
def generate_sub_dict(protein: str, pdb_path: str, map_missing_res: list, protein_seq: str):
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
    struct = get_structure(pdbx, model=1)
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

    seq_chain_a_single = []
    for aa in biostruc.get_residues(struct)[1]:
        try:
            seq_chain_a_single.append(ProteinSequence.convert_letter_3to1(aa))
        except KeyError:
            try:
                seq_chain_a_single.append(aa_dict[aa])
            except KeyError:
                seq_chain_a_single.append('X')
    
    three_letter_seq = biostruc.get_residues(struct)[1]

    if len(biostruc.get_residues(struct)[1]) == len(seq):
        for i, aa in enumerate(seq_chain_a_single):
            if aa != seq[i] and three_letter_seq[i] not in aa_dict.keys():
                aa_dict[three_letter_seq[i]] = seq[i]
    elif len(biostruc.get_residues(struct)[1]) == len(seq[[i for i, x in enumerate(list("".join(map_missing_res))) if x == "-"]]):
        non_disorder_indices = [i for i, x in enumerate(list("".join(map_missing_res))) if x == "-"]
        sub_seq = seq[non_disorder_indices]
        for i, aa in enumerate(seq_chain_a_single):
            if aa != sub_seq[i] and three_letter_seq[i] not in aa_dict.keys():
                aa_dict[three_letter_seq[i]] = sub_seq[i]
    else:
        print(f"Could not find a match for {protein}")
        

def calculate_scores(fasta_file: Fasta, pdb_path: str, nprocesses: int, mapping_fasta) -> tuple[dict, dict]:
    
    proteins = fasta_file.get_headers()
    with mp.Pool(int(nprocesses)) as pool:
        results = [pool.apply_async(generate_sub_dict, 
                                    args=(protein, pdb_path, 
                                          mapping_fasta[":".join((protein.upper() + "-disorder").split("-"))], 
                                          mapping_fasta[":".join((protein.upper() + "-sequence").split("-"))])) for protein in proteins]
        results = [r.get() for r in tqdm(results)]
    sasa_scores = {protein: sasa_scores for protein, sasa_scores, _ in results}
    bfactor_scores = {protein: bfactor_scores for protein, _, bfactor_scores in results}
    return sasa_scores, bfactor_scores

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--fasta_files', required=True, help='Path to fasta files')
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
    nprocesses = args.n_processes
    calculate_scores(fasta_files, pdb_path, nprocesses, mapping_fasta)
    with open(os.path.join(output_path, 'substitution_dict.json'), 'w') as f:
        json.dump(aa_dict, f)