from copy import deepcopy
import json
from pathlib import Path
import sys
import pandas as pd
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
import tempfile



def create_dataset_ala_pandey(protein: str, 
                                 pdb_path: str): 
                                 
    cif_header: str = protein.split('-')[0]
    try:
        pdbx = PDBxFile.read(os.path.join(pdb_path, f'{cif_header}.cif'))
    except FileNotFoundError:
        print(f'Could not find PDBx file for {protein}\nFetching from RCSB...')
        file_path = rcsb.fetch(cif_header, "cif")
        pdbx = PDBxFile.read(file_path)
    struct = get_structure(pdbx, model=1, extra_fields=["b_factor"])

    chain_id = protein.split('-')[1]
    chain_starts = biostruc.get_chain_starts(struct).tolist()
    chain_ids = biostruc.get_chains(struct).tolist()
    if biostruc.get_chain_count(struct) == 1 or chain_starts[chain_ids.index(chain_id)] == chain_starts[-1]:
        struct = struct[chain_starts[chain_ids.index(chain_id)]:]
    else:
        struct = struct[chain_starts[chain_ids.index(chain_id)]:chain_starts[chain_ids.index(chain_id) + 1]]
    
    struct = struct[biostruc.filter_amino_acids(struct)]
    
    bfactor = biostruc.apply_residue_wise(struct, struct.get_annotation("b_factor"), np.nanmean)
    bfactor[bfactor != 0.0] = (bfactor[bfactor != 0.0] - np.nanmean(bfactor, where=bfactor != 0.0)) / np.nanstd(bfactor, where=bfactor != 0.0)
    
    struct_seq = []
    for aa in biostruc.get_residues(struct)[1]:
        try:
            struct_seq.append(ProteinSequence.convert_letter_3to1(aa))
        except KeyError:
            try:
                struct_seq.append(aa_dict[aa])
            except KeyError:
                struct_seq.append('X')

    ca_list = np.array([atom.coord for atom in struct if atom.atom_name == "CA"])
    struct_ss = biostruc.annotate_sse(struct)
    try:
        assert len(ca_list) == len(struct_seq) == len(struct_ss), f"Length of PDB sequence ({len(struct_seq)}) and CA atoms ({len(ca_list)}) and len SS ({len(struct_ss)}) do not match. Protein: {protein}"
        # TODO -> 115 protein affected by this -> not worth the time
        # I hate my life
    except AssertionError as e:
        print(e)
        print(f"Skipping protein {protein}...")
        return protein, None, None

    ca_coord_norm = (ca_list - np.mean(ca_list, axis=0)) / np.std(ca_list, axis=0)
    struct_seq = [x if x in codes else "-" for x in struct_seq]
    one_hot_seq = np.array(one_hot.merge(pd.DataFrame(data={"AA": struct_seq}), how="right", on="AA").drop("AA", axis=1))
    struct_ss = np.array([one_hot_ss[x] for x in struct_ss])

    assert one_hot_seq.shape[0] == len(struct_ss) == len(ca_coord_norm), f"Length of one-hot sequence ({one_hot_seq.shape[0]}), secondary structure ({len(struct_ss)}) and CA coordinates ({len(ca_coord_norm)}) do not match"
    final_features = np.stack(np.concatenate([one_hot_seq, struct_ss, ca_coord_norm], axis=1))

    start_end_pp = np.zeros(final_features.shape[0])
    start_end_pp[0], start_end_pp[-1] = 1, 1
    final_features = np.concatenate([final_features, start_end_pp[:, np.newaxis]], axis=1)    
    prot_length = final_features.shape[0]
    final_features = np.pad(final_features, ((0, 500 - final_features.shape[0]), (0, 0)), mode='constant', constant_values=0)
    final_features = np.pad(final_features, ((0, 0), (0, 1)), mode='constant', constant_values=prot_length)

    #!Assert no empty residues
    
    assert np.where(~final_features.any(axis=1))[0].shape[0] == 0
    
    return protein, final_features, bfactor


def calculate_scores(ids: list, pdb_path: str, nprocesses: int = None) -> tuple[dict, dict]:
    """
    Calculates the SASA and B-factor scores for every protein in the fasta file. 
    The SASA and B-factor scores are calculated for each residue in the protein.
    
    Parameters
    ----------
    fasta_file (Fasta):  the fasta file containing the protein sequences
    pdb_path (str):  the path to the PDB files
    nprocesses (int):  the number of processes to use for multiprocessing
    
    Returns
    -------
    sasa_scores (dict):  the SASA scores for each residue in the protein
    bfactor_scores (dict):  the B-factor scores for each residue in the protein
    """
    
    proteins = ids
    with mp.Pool() as pool:
        results = [pool.apply_async(create_dataset_ala_pandey, 
                                    args=(protein, pdb_path)) for protein in proteins]
        results = [r.get() for r in tqdm(results)]
    print(len(results[0]))
    full_features = np.array([result[1] for result in results], dtype=object)
    protein_list = [result[0] for result in results]
    bfactors = np.array([result[2] for result in results], dtype=object)
    return full_features, protein_list, bfactors



def main(args: Optional[list] = None):
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--fasta_path', help='Path to fasta files')
    parser.add_argument('-p', '--pdb_path', required=False, help='Path to PDB structures')
    parser.add_argument('-o', '--output_path', required=True, help='Output path')
    parser.add_argument('-n', '--n_processes', required=False, help='Number of processes to use', type=int)

    # Parse arguments
    if args is None:
        args = parser.parse_args()
    # Access arguments

    global one_hot_ss
    one_hot_ss = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}
    global codes
    codes = ['A', 'V', 'F', 'I', 'L','D','E','K','S','T','Y','C','N','Q', 'P','M', 'R', 'H', 'W', 'G', '-']
    # create matrix with diagonal 1
    global one_hot
    one_hot = np.identity(len(codes))
    # create dataframe with one-hot encoding for each amino acid
    one_hot = pd.DataFrame(one_hot, index=codes, columns=codes)
    one_hot["AA"] = one_hot.index

    fasta_path = str(args.fasta_path)
    if args.pdb_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        pdb_path = temp_dir.name
    else:   
        pdb_path = args.pdb_path
    output_path = args.output_path

    global aa_dict
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/substitution_dict.json"), "r") as f:
        aa_dict = json.load(f)

     
    test_fasta = Fasta(path=fasta_path)
    test_fasta = {k: v for k, v in test_fasta.items() if len(v[0]) <= 500}

    all_ids = list(test_fasta.keys()) 
    # all_ids = [x for xs in all_ids for x in xs]
    bfactor_full_features, protein_list, bfactor = calculate_scores(all_ids, pdb_path, args.n_processes if args.n_processes else None)
    np.save(os.path.join(output_path, "bfactor_full_features.npy"), bfactor_full_features)
    np.save(os.path.join(output_path, "bfactor_y.npy"), bfactor)
    with open(os.path.join(output_path, "protein_list.txt"), "w") as f:
        f.write("\n".join(protein_list))
    print("Done")
    temp_dir.cleanup()


if __name__ == '__main__':
    main()