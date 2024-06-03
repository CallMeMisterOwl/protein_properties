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

one_hot_ss = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}

def create_dataset_ala_pandey(protein: str, 
                                 pdb_path: str,
                                 map_missing_res: list[str],
                                 protein_seq: list): 
                                 
    cif_header: str = protein.split('-')[0]
    try:
        pdbx = PDBxFile.read(os.path.join(pdb_path, f'{cif_header}.cif'))
    except FileNotFoundError:
        print(f'Could not find PDBx file for {protein}\nFetching from RCSB...')
        file_path = rcsb.fetch(cif_header, "cif")
        pdbx = PDBxFile.read(file_path)
    struct = get_structure(pdbx, model=1, extra_fields=["b_factor"])

    # Thank you biotite for this wonderful class, NOT!
    try:
        seq = ProteinSequence(list(("".join(protein_seq)).replace('U', 'C').replace("O", "K")))
    except AlphabetError:
        print(f"Protein {protein} contains non-canonical amino acids, skipping...")
        print(f"Protein sequence: {protein_seq}")
        sys.exit(1)

    seq_length = len(seq)
    chain_id = protein.split('-')[1]
    chain_starts = biostruc.get_chain_starts(struct).tolist()
    chain_ids = biostruc.get_chains(struct).tolist()
    if biostruc.get_chain_count(struct) == 1 or chain_starts[chain_ids.index(chain_id)] == chain_starts[-1]:
        struct = struct[chain_starts[chain_ids.index(chain_id)]:]
    else:
        struct = struct[chain_starts[chain_ids.index(chain_id)]:chain_starts[chain_ids.index(chain_id) + 1]]
    
    struct = struct[biostruc.filter_amino_acids(struct)]
    struct_seq = []
    for aa in biostruc.get_residues(struct)[1]:
        try:
            struct_seq.append(ProteinSequence.convert_letter_3to1(aa))
        except KeyError:
            try:
                struct_seq.append(aa_dict[aa])
            except KeyError:
                struct_seq.append('X')
    struct_ss = biostruc.annotate_sse(struct)
    ca_list = np.array([atom.coord for atom in struct if atom.atom_name == "CA"])
    try:
        assert len(ca_list) == len(struct_seq) == len(struct_ss), f"Length of PDB sequence ({len(seq)}) and CA atoms ({len(ca_list)}) and len SS ({len(struct_ss)}) do not match. Protein: {protein}"
        # TODO f this man, assert is triggered for 1 protein, need to investigate -> 115 protein affected by this -> not worth the time< 
        # I hate my life
    except AssertionError as e:
        print(e)
        print(f"Skipping protein {protein}...")
        return protein, None

    ca_coord_norm = (ca_list - np.mean(ca_list, axis=0)) / np.std(ca_list, axis=0)
    struct_seq = [x if x in codes else "-" for x in struct_seq]
    one_hot_seq = np.array(one_hot.merge(pd.DataFrame(data={"AA": struct_seq}), how="right", on="AA").drop("AA", axis=1))
    struct_ss = np.array([one_hot_ss[x] for x in struct_ss])

    # mask the residues that are not in the PDB files, due to disorder
    disorder_residues = list("".join(map_missing_res))
    non_disorder_indices = [i for i, x in enumerate(disorder_residues) if x == "-"]
    assert one_hot_seq.shape[0] == len(struct_ss) == len(ca_coord_norm), f"Length of one-hot sequence ({one_hot_seq.shape[0]}), secondary structure ({len(struct_ss)}) and CA coordinates ({len(ca_coord_norm)}) do not match"
    final_features = np.stack(np.concatenate([one_hot_seq, struct_ss, ca_coord_norm], axis=1))
    if len(disorder_residues) != final_features.shape[0]:
        final_features_masked = np.zeros((len(disorder_residues), 27))

        """
        see if the per residue SASA and B-factor scores can be mapped to the primary 
        sequence when using only non-disordered residues
        """
        try:
            
            final_features_masked[non_disorder_indices] = final_features
        except ValueError:
            
            alignment = align_sequences_nw(seq, "".join(struct_seq))
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
                    final_features_masked[seq_chain_overlap_cut, :] = final_features[sasa_index]
                except IndexError as i:
                    print(f'Skipping protein {protein}...\n')
                    print(i)
                    return protein, None
                

            elif np.any(seq_chain_overlap == False):
                try:
                    final_features_masked[seq_chain_overlap, :] = final_features
                except IndexError as i:
                    print(f'Skipping protein {protein}...\n')
                    print(i)
                    return protein, None
                
            
            elif np.any(primary_seq_overlap == False):
                try:
                    final_features_masked = final_features[primary_seq_overlap, :]
                except IndexError as i:
                    print(f'Skipping protein {protein}...\n')
                    print(i)
                    return protein, None
                
            
            else:
                print(f"Investiage protein {protein}")
                return protein, None, None
                
        start_end_pp = np.zeros(final_features_masked.shape[0])
        start_end_pp[0], start_end_pp[-1] = 1, 1
        prot_length = np.array([final_features_masked.shape[0]] * final_features_masked.shape[0])
        final_features_masked = np.concatenate([final_features_masked, start_end_pp[:, np.newaxis]], axis=1)
        prot_length = final_features_masked.shape[0]
        final_features_masked = np.pad(final_features_masked, ((0, 500 - final_features_masked.shape[0]), (0, 0)), mode='constant', constant_values=0)
        final_features_masked = np.pad(final_features_masked, ((0, 0), (0, 500)), mode='constant', constant_values=prot_length)
        return protein, final_features_masked

    start_end_pp = np.zeros(final_features.shape[0])
    start_end_pp[0], start_end_pp[-1] = 1, 1
    final_features = np.concatenate([final_features, start_end_pp[:, np.newaxis]], axis=1)    
    prot_length = final_features.shape[0]
    final_features = np.pad(final_features, ((0, 500 - final_features.shape[0]), (0, 0)), mode='constant', constant_values=0)
    final_features = np.pad(final_features, ((0, 0), (0, 500)), mode='constant', constant_values=prot_length)
    return protein, final_features



def calculate_scores(ids: list, pdb_path: str, nprocesses: int, mapping_fasta: Fasta) -> tuple[dict, dict]:
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
    
    proteins = ids
    with mp.Pool(int(nprocesses)) as pool:
        results = [pool.apply_async(create_dataset_ala_pandey, 
                                    args=(protein, pdb_path, 
                                          mapping_fasta[":".join((protein.upper() + "-disorder").split("-"))], 
                                          mapping_fasta[":".join((protein.upper() + "-sequence").split("-"))])) for protein in proteins]
        results = [r.get() for r in tqdm(results)]
    full_features = np.array([features for _, features in results], dtype=object)
    protein_list = [protein for protein, _ in results]
    return full_features, protein_list



def main(args: Optional[list] = None):
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--fasta_path', help='Path to fasta files')
    parser.add_argument('-p', '--pdb_path', required=True, help='Path to PDB structures')
    parser.add_argument("-m", "--mapping_file", required=True, help="Path to mapping file, which is required to fill in missing residues")
    parser.add_argument('-o', '--output_path', required=True, help='Output path')
    parser.add_argument('-n', '--n_processes', default=16, help='Number of processes to use', type=int)

    # Parse arguments
    if args is None:
        args = parser.parse_args()
    # Access arguments

    map_missing_res = Fasta(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/substitution_dict.json"))

    global codes
    codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    # create matrix with diagonal 1
    global one_hot
    one_hot = np.identity(len(codes))
    # create dataframe with one-hot encoding for each amino acid
    one_hot = pd.DataFrame(one_hot, index=codes, columns=codes)
    one_hot["AA"] = one_hot.index

    fasta_path = str(args.fasta_path)
    pdb_path = args.pdb_path
    mapping_file = args.mapping_file
    output_path = args.output_path
    mapping_fasta = Fasta(mapping_file)

    global aa_dict
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/substitution_dict.json"), "r") as f:
        aa_dict = json.load(f)

    fasta_path = os.path.join(fasta_path, "test_norm.o") 
    test_fasta = Fasta(path=fasta_path)
    test_fasta = {k: v for k, v in test_fasta.items() if len(v[0]) <= 500}

    all_ids = list(test_fasta.keys()) 
    # all_ids = [x for xs in all_ids for x in xs]
    bfactor_full_features, protein_list = calculate_scores(all_ids, pdb_path, args.n_processes, mapping_fasta)
    np.save(os.path.join(output_path, "bfactor_full_features.npy"), bfactor_full_features)
    with open(os.path.join(output_path, "protein_list.txt"), "w") as f:
        f.write("\n".join(protein_list))
    print("Done")


if __name__ == '__main__':
    main()