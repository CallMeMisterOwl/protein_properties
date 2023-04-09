from urllib.request import urlopen
import gzip
from datetime import datetime
from pathlib import Path
import argparse
from Bio import SwissProt
import numpy as np
from tqdm import tqdm
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
import multiprocessing
from fasta import Fasta


# data types
def calculate_sequence_similarity(query_seq: str, lookup_seqs: list[str]) -> list[float]:
    """
    Calculate sequence similarity between query protein and lookup proteins. 
    Sequence similarity and identity are used interchangeably, even though they are not the same.
    Uniqueprot uses sequence identity -> we should be fine 
    """
    # data types
    scores = []
    with multiprocessing.Pool() as pool:
        results = pool.starmap(calculate_similarity, [(query_seq, seq) for seq in lookup_seqs])
        scores = [result[0] for result in results]
    return scores


def calculate_similarity(query_seq: str, seq: str) -> float:
    alignments = aligner.align(query_seq, seq)
    a = alignments[0]
    seq_similarity = a.counts().identities / len(a.indices[0]) * 100
    return seq_similarity, seq


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output_path', required=True, help='Output path')
    parser.add_argument('-i', '--input')

    args = parser.parse_args()
    output_path = Path(args.output_path)
    input_path = Path(args.input)
    
    blosum_matrix = substitution_matrices.load("BLOSUM62")
    global aligner
    aligner = PairwiseAligner(scoring="blastp", mode = 'global')
    aligner.substitution_matrix = blosum_matrix

    fasta = Fasta(path=input_path)

    # calculate sequence similarity
    for i, query_seq in tqdm(enumerate(fasta.get_headers())):
        scores = calculate_sequence_similarity(query_seq, lookup_seqs)
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        # save scores to file
        with open(output_path / 'scores.txt', 'a') as f:
            f.write(f'{query_ids[i]}\t{mean_score}\t{median_score}\t{max_score}\t{min_score}\n')
        lookup_seqs.append(query_seq)



if __name__ == '__main__':
    main()