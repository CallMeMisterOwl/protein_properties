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


def get_data(url, output_path) -> None:
    aas = ['A', 'V', 'F', 'I', 'L','D','E','K','S','T','Y','C','N','Q', 'P','M', 'R', 'H', 'W', 'G']
    # check if data is already downloaded
    if (output_path / 'id_seqs.fasta').exists() and (output_path / 'dates.npy').exists():
        print(f'Data already exists at {output_path}')
        return
    handle = urlopen(url)
    handle = gzip.open(handle, "rt")
    dates = []
    p_ids = []
    seqs = []
    for ref in tqdm(SwissProt.parse(handle)):
        dates.append(datetime.strptime(ref.created[0], "%d-%b-%Y"))
        p_ids.append(ref.accessions[0])
        seqs.append(''.join(['*' if aa not in aas else aa for aa in ref.sequence]))
    
    dates = np.array(dates, dtype='datetime64[M]')
    sorted_indices = np.argsort(dates)
    dates = dates[sorted_indices]
    p_ids = np.array(p_ids)[sorted_indices].tolist()
    # memory footprint is too large to use numpy arrays
    seqs = [seqs[i] for i in sorted_indices]
    with open(output_path / "id_seqs.fasta", 'w') as f:
        for i in range(len(dates)):
            f.write(f'>{p_ids[i]}\n{seqs[i]}\n')
    
    print(f'Wrote {len(dates)} sequences to {output_path}')
    np.save(output_path / 'dates.npy', dates)

# TODO fix alpahbet issue -> ValueError: sequence contains letters not in the alphabet
def calculate_sequence_similarity(query_seq: str, lookup_seqs: list[str]) -> list[float]:
        """
        Calculate sequence similarity between query protein and lookup proteins. 
        Sequence similarity and identity are used interchangeably, even though they are not the same.
        Uniqueprot uses sequence identity -> we should be fine 
        """
        # data types
        scores = []
        for idx, seq in enumerate(lookup_seqs):
            alignments = aligner.align(query_seq, seq)
            best_ss_score = 0
            for a in alignments: # find best alignment -> highest sequence similarity score
                seq_similarity = a.counts[1] / len(a.indices[0]) * 100
                best_ss_score = seq_similarity if seq_similarity > best_ss_score else best_ss_score
                    
            scores.append(best_ss_score)
        assert len(scores) == len(lookup_ids) # check if all lookup proteins have a score
        return scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output_path', required=True, help='Output path')
    parser.add_argument('-u', '--url', help='URL to download data from', 
    default='ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz')
    parser.add_argument('-c', '--cutoff_year')

    args = parser.parse_args()
    output_path = Path(args.output_path)
    url = args.url
    cutoff_year = args.cutoff_year
    get_data(url, output_path)

    
    blosum_matrix = substitution_matrices.load("BLOSUM62")
    global aligner
    aligner = PairwiseAligner(scoring="blastp", mode = 'global')
    aligner.substitution_matrix = blosum_matrix

    # all dates smaller than cutoff_year
    dates = np.load(output_path / 'dates.npy')
    seqs = []
    ids = []
    with open(output_path / 'id_seqs.fasta', 'r') as f:
        for line in f:
            if line.startswith('>'):
                ids.append(line[1:-1])
            else:
                seqs.append(line)

    cutoff_date = np.datetime64(cutoff_year)
    mask = dates < cutoff_date
    lookup_seqs = [seq for i, seq in enumerate(seqs) if mask[i]]
    query_seqs = [seq for i, seq in enumerate(seqs) if not mask[i]]
    query_ids = [id for i, id in enumerate(ids) if not mask[i]]
    with open(output_path / 'scores.txt', 'w') as f:
        f.write("ID\tMean\tMedian\tMax\tMin\n")
    # calculate sequence similarity
    for i, query_seq in enumerate(query_seqs):
        scores = calculate_sequence_similarity(query_seq, lookup_seqs)
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        # save scores to file
        with open(output_path / 'scores.txt', 'w') as f:
            f.write(f'{query_ids[i]}\t{mean_score}\t{median_score}\t{max_score}\t{min_score}\n')
        lookup_seqs.append(query_seq)



if __name__ == '__main__':
    main()