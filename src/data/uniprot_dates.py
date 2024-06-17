from urllib.request import urlopen
import gzip
from datetime import datetime
from pathlib import Path
import argparse


def main():


    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output_path', required=True, help='Output path')
    args = parser.parse_args()
    output_path = Path(args.output_path)
    url = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.dat.gz"
    handle = urlopen(url)
    handle = gzip.open(handle, "rt")
    dates = []
    p_ids = []
    seqs = []
    for ref in SwissProt.parse(handle):
        dates.append(datetime.strptime(ref.created[0], "%Y-%b-%d"))
        p_ids.append(ref.entry_name)
        seqs.append(ref.sequence)
    
    dates = np.array(dates, dtype='datetime64')
    sorted_indices = np.argsort(dates)
    dates = dates[sorted_indices]
    p_ids = np.array(p_ids)[sorted_indices].tolist()
    seqs = np.array(seqs)[sorted_indices].tolist()

    with open(output_path / "id_seqs.fasta", 'w') as f:
        for i in range(len(dates)):
            f.write(f'>{p_ids[i]}\n{seqs[i]}\n')
    
    print(f'Wrote {len(dates)} sequences to {output_path}')

    np.save(dates, output_path / 'dates.npy')


if __name__ == '__main__':
    main()