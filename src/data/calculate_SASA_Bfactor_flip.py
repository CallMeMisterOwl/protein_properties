from copy import copy
from pathlib import Path
from typing import Optional
from tempfile import gettempdir
import numpy as np
from tqdm import tqdm
from biotite.structure.io.pdbx import PDBxFile, get_structure, get_sequence
import biotite.structure as biostruc
import biotite.database.rcsb as rcsb
from fasta import Fasta
import argparse
import multiprocessing as mp
import os

def calculate_scores_for_protein(protein: str, pdb_path: str, map_missing_res: list[str]) -> tuple:
    cif_header: str = protein.split('-')[0]
    try:
        pdbx = PDBxFile.read(os.path.join(pdb_path, f'{cif_header}.cif'))
    except FileNotFoundError:
        print(f'Could not find PDBx file for {protein}\nFetching from RCSB...')
        file_path = rcsb.fetch(protein, "cif")
        pdbx = PDBxFile.read(file_path)
    struct = get_structure(pdbx, model=1, extra_fields=["b_factor"])
    seq_length = len(get_sequence(pdbx)[0])
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

    res_sasa_masked = np.zeros(len(disorder_residues))
    res_sasa_masked[non_disorder_indices] = res_sasa

    res_bfactor_masked = np.zeros(len(disorder_residues))
    res_bfactor_masked[non_disorder_indices] = res_bfactor

    assert len(res_sasa_masked) == seq_length, f'Length of primary sequence {protein} does not match length of SASA scores'

    return protein, res_sasa_masked, res_bfactor_masked


def calculate_scores(fasta_file: Fasta, pdb_path: str, nprocesses: int, mapping_fasta) -> tuple[dict, dict]:
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

    parser.add_argument('-s', '--fasta-files', nargs='+', help='Path(s) to fasta files')
    parser.add_argument('-p', '--pdb_path', required=True, help='Path to PDB structures')
    parser.add_argument("-m", "--mapping-file", required=True, help="Path to mapping file, which is required to fill in missing residues")
    parser.add_argument('-o', '--output_path', required=True, help='Output path')
    parser.add_argument('-n', '--n_processes', default=16, help='Number of processes to use')

    # Parse arguments
    if args is None:
        args = parser.parse_args()
    # Access arguments

    split_paths = args.split_paths
    pdb_path = args.pdb_path
    mapping_file = args.mapping_file
    output_path = args.output_path
    mapping_fasta = Fasta(mapping_file)
    for fasta_path in split_paths:
        fasta = Fasta(fasta_path)
        sasa_scores, bfactor_scores = calculate_scores(fasta, pdb_path, args.n_processes, mapping_fasta)
        copy(fasta).append(bfactor_scores).write_fasta(f'{output_path}/bfactor/{Path(fasta_path).stem}_bfactor.fasta')
        fasta.append(sasa_scores).write_fasta(f'{output_path}/sasa/{Path(fasta_path).stem}_sasa.fasta')




if __name__ == '__main__':
    main()
