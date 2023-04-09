import cProfile
import gzip
import io
import json
import pstats
import sys
from copy import deepcopy
from os import path
from pathlib import Path
from pstats import SortKey
from tempfile import gettempdir
from typing import Optional

import Bio.PDB as PDB
import biotite.database.rcsb as rcsb
import biotite.structure as biostruc
import h5py
import numpy as np
from Bio import SeqIO
from Bio.PDB import MMCIFParser
from biotite.sequence import AlphabetError, ProteinSequence
from biotite.structure.io.pdbx import CIFFile, get_sequence, get_structure
import biotite.database.rcsb as rcsb
from tqdm import tqdm

from src.data.fasta import Fasta

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import multiprocessing as mp
import os
import warnings
from multiprocessing.pool import Pool

from utils import (
    HOA_TIEN,
    SUBSTITUTION_DICT,
    TO_RSA,
    align_sequences_nw,
    # calculate_b_sasa_scores,
    fetch_pdb_sequence,
    get_auth_to_label_asym_mapping,
    get_relative_sa,
    get_pdb_structure,
)

mapping_h5 = None

# TODO find a way to automatically substitute non-generic amino acid with generic ones


def debug_calculate_scores_for_protein(
    fasta_file: Fasta, nprocesses: int, mapping_dict: dict, cif_dir, upper: bool = True
) -> tuple[dict, dict]:
    results = []
    for idx, record in tqdm(enumerate(fasta_file)):

        try:
            pr = cProfile.Profile()
            pr.enable()
            results.append(
                calculate_b_sasa_scores(
                    record.id,
                    mapping_dict[record.id.split("_")[0]],
                    record.seq,
                    cif_dir,
                )
            )
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            break
        except KeyError:
            print(f"Could not find mapping for {record.id}")
            continue
    sasa_scores = {
        protein: [sasa_scores]
        for protein, sasa_scores, _, _ in results
        if sasa_scores is not None
    }
    bfactor_scores = {
        protein: [bfactor_scores]
        for protein, _, bfactor_scores, _ in results
        if bfactor_scores is not None
    }
    protein_seq = {
        protein: protein_seq
        for protein, _, _, protein_seq in results
        if protein_seq is not None
    }
    return sasa_scores, bfactor_scores, protein_seq


def calculate_b_sasa_scores(protein, protein_seq, cif_dir, split_symbol):
    global mapping_h5
    cif_header, chain_id = protein.split(split_symbol)
    struct, cif = get_pdb_structure(protein, cif_dir, split_symbol)
    if struct is None or struct.coord.size == 0:
        return protein, None, None, None

    # well this is a mess, the mappings were created using sifts which uses the label_asym_id, but the structure uses the auth_asym_id
    asym_mappping = get_auth_to_label_asym_mapping(cif)
    try:
        uniprot = mapping_h5[cif_header][asym_mappping[chain_id]]["uniprot"][()]
        pdb = mapping_h5[cif_header][asym_mappping[chain_id]]["pdb"][()]
    except KeyError:
        return protein, None, None, None
    mapping_mask = pdb != -1
    #mapping = {k: v for k, v in zip(pdb[mapping_mask], uniprot[mapping_mask])}
    pdb_filtered = pdb[mapping_mask]
    uniprot_filtered = uniprot[mapping_mask]
    mapping = np.full(np.max(pdb_filtered) + 1, -1, dtype=int)
    mapping[pdb_filtered] = uniprot_filtered
    
    '''chain_starts = biostruc.get_chain_starts(struct).tolist()
    chain_ids = biostruc.get_chains(struct).tolist()
    try:
        assert chain_id in chain_ids, f"Chain {chain_id} not found for {protein}"
    except AssertionError as e:
        print(e)
        return protein, None, None, None
    if (
        biostruc.get_chain_count(struct) == 1
        or chain_starts[chain_ids.index(chain_id)] == chain_starts[-1]
    ):
        struct = struct[chain_starts[chain_ids.index(chain_id)] :]
    else:
        struct = struct[
            chain_starts[chain_ids.index(chain_id)] : chain_starts[
                chain_ids.index(chain_id) + 1
            ]
        ]'''
    chain_starts = biostruc.get_chain_starts(struct).tolist()
    chain_ids = biostruc.get_chains(struct).tolist()
    if chain_id not in chain_ids:
        print(f"Chain {chain_id} not found for {protein}")
        return protein, None, None, None

    chain_index = chain_ids.index(chain_id)
    if biostruc.get_chain_count(struct) == 1 or chain_starts[chain_index] == chain_starts[-1]:
        struct = struct[chain_starts[chain_index]:]
    else:
        struct = struct[chain_starts[chain_index]:chain_starts[chain_index + 1]]
    struct = struct[biostruc.filter_amino_acids(struct)]
    sasa = np.full(len(protein_seq), np.nan)
    bfactor = np.full(len(protein_seq), np.nan)

    try:
        atom_sasa = biostruc.sasa(struct, vdw_radii="ProtOr", point_number=500)
    except (ValueError, KeyError):
        return protein, None, None, None

    res_sasa = biostruc.apply_residue_wise(struct, atom_sasa, np.nansum)

    # TODO make this pretty and efficient
    res_ids = biostruc.get_residues(struct)[0]
    res_mask = (res_ids != None) & (res_ids < len(mapping))
    # assert res_ids are positive
    # got triggered by a negative residue id
    # assert all(res_ids > 0), f"Negative residue id found for {protein}"

    '''mapping_vec = np.vectorize(mapping.get)  # vectorize the mapping function
    try:
        np_mapping = mapping_vec(res_ids[res_mask])
    except:
        return protein, None, None, None'''
    # remove res_ids not in mapping
    try:
        np_mapping = mapping[res_ids[res_mask]]
    except IndexError:
        print('fuck you')
    mask = np_mapping != -1
    if np.any(np_mapping[mask] >= len(sasa)):
        print(f"Index out of bounds: max index {np.max(np_mapping[mask])}, sasa length {len(sasa)}, protein {protein}")
        return protein, None,None,None
    sasa[np_mapping[mask]] = res_sasa[res_mask][mask]
    for atom in struct:
        if atom.res_id >= len(mapping):
            continue
        if atom.atom_name == "CA" and mapping[atom.res_id] != -1:
            pos_idx = mapping[atom.res_id]
            bfactor[pos_idx] = atom.b_factor
    if all(bfactor == 0):
        return protein, None, None, None
    bfactor = np.clip(bfactor, 0.00001, None)
    # normalize B-factor
    bfactor = (bfactor - np.nanmean(bfactor)) / np.nanstd(bfactor)
    bfactor = np.nan_to_num(bfactor, nan=-1)

    sasa = np.clip(sasa, 0.00001, None)
    sasa = get_relative_sa(protein_seq, sasa)
    sasa = np.nan_to_num(sasa, nan=-1)

    return protein.replace("_", "-"), sasa.tolist(), bfactor.tolist(), protein_seq


# Initialization function to be called once in each worker process
def init_worker(h5_file_path):
    global mapping_h5
    # Open the HDF5 file once for this process
    mapping_h5 = h5py.File(h5_file_path, "r")


def calculate_scores(
    fasta_file: Fasta, nprocesses: int, cif_dir: str, mapping_file, upper: bool = True
) -> tuple[dict, dict]:
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
    # warnings.filterwarnings("error")
    # you are beyond stupid if you are reading this
    #! record = next(fasta_file) why skip the first record? Are you mental or something?
    fasta_file = list(fasta_file)
    record = fasta_file[0]

    if "-" in record.id:
        split_symbol = "-"
    else:
        split_symbol = "_"
    pids = [record.id.split(split_symbol)[0] for record in fasta_file]
    rcsb.fetch(pids, "cif", gettempdir())
    with Pool(
        int(nprocesses), initializer=init_worker, initargs=(mapping_file,)
    ) as pool:
        results = []
        for record in fasta_file:
            results.append(
                pool.apply_async(
                    calculate_b_sasa_scores,
                    args=(
                        record.id,
                        record.seq,
                        cif_dir,
                        split_symbol,
                    ),
                )
            )

        # Collect results
        results = [r.get() for r in tqdm(results)]
    sasa_scores = {
        protein: [sasa_scores]
        for protein, sasa_scores, _, _ in results
        if sasa_scores is not None
    }
    bfactor_scores = {
        protein: [bfactor_scores]
        for protein, _, bfactor_scores, _ in results
        if bfactor_scores is not None
    }
    protein_seq = {
        protein: protein_seq
        for protein, _, _, protein_seq in results
        if protein_seq is not None
    }
    return sasa_scores, bfactor_scores, protein_seq


def main(args: Optional[list] = None):
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--fasta_files", nargs="+", help="Path(s) to fasta files")

    parser.add_argument(
        "-p", "--pdb_path", required=False, help="Path to PDB structures"
    )
    parser.add_argument(
        "-m",
        "--mapping_file",
        required=True,
        help="Path to mapping file, which is required to fill in missing residues",
    )
    parser.add_argument("-o", "--output_path", required=True, help="Output path")

    parser.add_argument(
        "-n", "--n_processes", default=16, help="Number of processes to use", type=int
    )
    parser.add_argument(
        "-u", "--upper", action="store_true", help="Use uppercase protein names"
    )

    # Parse arguments
    if args is None:
        args = parser.parse_args()
    # Access arguments

    fasta_files = args.fasta_files
    pdb_path = args.pdb_path
    mapping_file = args.mapping_file
    output_path = args.output_path
    # mapping_fasta = Fasta(mapping_file)
    upper = args.upper

    for fasta_path in fasta_files:
        fasta_path = Path(fasta_path)
        fasta = SeqIO.parse(fasta_path, "fasta")
        # fasta = Fasta(fasta_path)

        sasa_scores, bfactor_scores, seqs = calculate_scores(
            fasta, args.n_processes, pdb_path, mapping_file, upper
        )

        with open(f"{output_path}/{fasta_path.stem}_bfactor.tsv", "w") as bf, open(
            f"{output_path}/{fasta_path.stem}_sasa.tsv", "w"
        ) as sasa:
            bf.write("Protein\tPosition\tAA\tnorm_Bfactor\n")
            sasa.write("Protein\tPosition\tAA\tRSA\n")
            for protein, scores in bfactor_scores.items():
                seq = seqs[protein]
                for i, score in enumerate(scores[0]):
                    bf.write(f"{protein}\t{i}\t{seq[i]}\t{score}\n")
            for protein, scores in sasa_scores.items():
                seq = seqs[protein]
                for i, score in enumerate(scores[0]):
                    sasa.write(f"{protein}\t{i}\t{seq[i]}\t{score}\n")


if __name__ == "__main__":
    main()
