import gzip
import json
import sys
from copy import deepcopy
from os import path
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

import Bio.PDB as PDB
import biotite.database.rcsb as rcsb
import biotite.structure as biostruc
import numpy as np
import cProfile, pstats, io
from pstats import SortKey
from Bio import SeqIO
from Bio.PDB import MMCIFParser
from biotite.sequence import AlphabetError, ProteinSequence
from biotite.structure.io.pdbx import CIFFile, get_sequence, get_structure
from tqdm import tqdm

from src.data.fasta import Fasta

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import multiprocessing as mp
from multiprocessing.pool import Pool
import os

import warnings



from utils import (
    HOA_TIEN,
    TO_RSA,
    SUBSTITUTION_DICT,
    align_sequences_nw,
    fetch_pdb_sequence,
    get_auth_to_label_asym_mapping,
    get_relative_sa,
    calculate_b_sasa_scores,
)

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


def calculate_scores(
    fasta_file: Fasta, nprocesses: int, mapping_dict: dict, cif_dir: str, upper: bool = True
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
    with Pool(int(nprocesses)) as pool:
        results = []
        for record in fasta_file:
            try:
                results.append(
                    pool.apply_async(
                        calculate_b_sasa_scores,
                        args=(
                            record.id,
                            mapping_dict[record.id.split("_")[0]],
                            record.seq,
                            cif_dir,
                        ),
                    )
                )
            except KeyError:
                print(f"Could not find mapping for {record.id}")
                continue
        """results = [pool.apply_async(calculate_b_sasa_scores, 
                                    args=(record.id, 
                                          mapping_dict[record.id.split('_')[0]],
                                          record.seq)) 
                                          for record in fasta_file]"""
        print(results)
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
    mapping_dict = json.load(open(mapping_file))
    # mapping_fasta = Fasta(mapping_file)
    upper = args.upper

    for fasta_path in fasta_files:
        fasta = SeqIO.parse(fasta_path, "fasta")
        # fasta = Fasta(fasta_path)
        
        
        sasa_scores, bfactor_scores, seqs = calculate_scores(
            fasta, args.n_processes, mapping_dict, pdb_path, upper
        )
        
        
        with open(f"{output_path}/pdb_all_bfactor.tsv", "w") as bf, open(
            f"{output_path}/pdb_all_sasa.tsv", "w"
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
