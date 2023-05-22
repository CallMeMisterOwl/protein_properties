"""
Read in the 3 fasta files: "/mnt/home/mheinzinger/deepppi1tb/oculus_backup/per_residue_prediction/data_set/Train_HHblits.fasta",
"/mnt/home/mheinzinger/deepppi1tb/oculus_backup/per_residue_prediction/data_set/new_test_final_pisces.fasta" and
"/mnt/home/mheinzinger/deepppi1tb/oculus_backup/per_residue_prediction/data_set/CASP12_HHblits.fasta" using the Fasta class
Afterwards for each fasta file: iterate over all headers and get the sequence from the corresponding mmcif file in
"/mnt/home/mheinzinger/deepppi1tb/oculus_backup/per_residue_prediction/distance_maps/structures/cif" using the PDBxReader class

Now align the seq_chain_a_single to the sequence from the fasta file using brute force alignment described here https://johnlekberg.com/blog/2020-10-25-seq-align.html
Finally return a fasta file that contains only sequences that have a sequence identity of below 100%
"""

import argparse
import sys
from os import path
from pathlib import Path

from biotite.structure.io.pdbx import PDBxFile
import biotite.structure as struc
from fasta import Fasta
from biotite.sequence import ProteinSequence

# Ugly but necessary to import utils
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import utils


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--fasta', nargs='+', help='Path(s) to fasta files')
    parser.add_argument('-p', '--pdb_path', required=True, help='Path to PDB structures')
    parser.add_argument('-o', '--output_path', required=True, help='Output path')

    args = parser.parse_args()

    fasta_files = args.fasta
    pdb_path = args.pdb_path
    output_path = args.output_path

    final_fasta = Fasta()

    for fasta in fasta_files:
        fasta = Fasta(fasta)
        for header in fasta.get_headers():
            pdbx = PDBxFile.read(Path(pdb_path) / (header.split('-')[0] + ".cif"))
            struct = struc.io.pdbx.get_structure(pdbx, extra_fields=["b_factor"], model=1)
            chain_starts = struc.get_chain_starts(struct).tolist()
            chain_ids = struc.get_chains(struct).tolist()
            chain_id = header.split('-')[1]
            if struc.get_chain_count(struct) == 1 or chain_starts[chain_ids.index(chain_id)] == chain_starts[-1]:
                struct_chain = struct[chain_starts[chain_ids.index(chain_id)]:]
            else:
                struct_chain = struct[
                               chain_starts[chain_ids.index(chain_id)]:chain_starts[chain_ids.index(chain_id) + 1]]
            num_res = struc.get_residue_count(struct_chain)
            struct_chain = struct_chain[struc.filter_amino_acids(struct_chain)]
            if num_res == fasta.get_sequence_lengths(header):
                break
            seq_chain_a = []
            for res in struc.residue_iter(struct_chain):
                seq_chain_a.append(res[0].res_name)
            seq_chain_a_single = []
            for aa in seq_chain_a:
                seq_chain_a_single.append(ProteinSequence.convert_letter_3to1(aa))
            fasta[header] = utils.align_sequences_nw("".join(seq_chain_a_single), fasta.get_sequence(header)[0])
        final_fasta.append(fasta)

    final_fasta.write_fasta(str(Path(output_path) / "aligned.fasta"), overwrite=True)


if __name__ == '__main__':
    main()
