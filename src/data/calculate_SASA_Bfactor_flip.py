from copy import copy
from pathlib import Path
from typing import Optional

import numpy
from tqdm import tqdm
from biotite.structure.io.pdbx import PDBxFile, get_structure
from biotite.structure import sasa
from src.data.fasta import Fasta
import argparse
import multiprocessing as mp


def calculate_scores_for_protein(protein: str, pdb_path: str) -> tuple:
    cif_header: str = protein.split('-')[0]
    try:
        pdbx = PDBxFile.read(f'{pdb_path}/{cif_header}.cif')
    except FileNotFoundError:
        raise FileNotFoundError(f'Could not find PDB structure for {protein}')

    struct = get_structure(pdbx, model=1, extra_fields=["b_factor"])
    sasa_scores = sasa(struct).tolist()
    bfactor_scores = struct.get_annotation("b_factor").tolist()
    return protein, sasa_scores, bfactor_scores


def calculate_scores(fasta_file: Fasta, pdb_path: str, nprocesses: int) -> tuple[dict, dict]:
    proteins = fasta_file.get_headers()
    with mp.Pool(nprocesses) as pool:
        results = [pool.apply_async(calculate_scores_for_protein, args=(protein, pdb_path)) for protein in proteins]
        results = [r.get() for r in tqdm(results)]
    sasa_scores = {protein: sasa_scores for protein, sasa_scores, _ in results}
    bfactor_scores = {protein: bfactor_scores for protein, _, bfactor_scores in results}
    return sasa_scores, bfactor_scores



def main(args: Optional[list] = None):
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--split_paths', nargs='+', help='Path(s) to split files')
    parser.add_argument('-p', '--pdb_path', required=True, help='Path to PDB structures')
    parser.add_argument('-o', '--output_path', required=True, help='Output path')
    parser.add_argument('-n', '--n_processes', default=16, help='Number of processes to use')

    # Parse arguments
    if args is None:
        args = parser.parse_args()
    # Access arguments
    split_paths = args.split_paths
    pdb_path = args.pdb_path
    output_path = args.output_path
    for fasta_path in split_paths:
        fasta = Fasta(fasta_path)
        sasa_scores, bfactor_scores = calculate_scores(fasta, pdb_path, args.n_processes)
        copy(fasta).append(bfactor_scores).write_fasta(f'{output_path}/bfactor/{Path(fasta_path).stem}_bfactor.fasta')
        fasta.append(sasa_scores).write_fasta(f'{output_path}/sasa/{Path(fasta_path).stem}_sasa.fasta')




if __name__ == '__main__':
    main()
