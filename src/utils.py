import random
import numpy as np
from pytorch_lightning import seed_everything
import torch  
from tqdm import tqdm
import math
from Bio import Entrez
from Bio.Seq import Seq
import concurrent.futures
import requests

def seed_all(seed=13):
    """
    Seed function to guarantee the reproducibility of the code.

    See https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    """
    seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)

# https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5
def align_sequences_nw(x: str, y: str, match=2, mismatch=100, gap=1):
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    F[0,:] = np.linspace(0, -ny * gap, ny + 1)
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            rx.append(x[i-1])
            ry.append('-')
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(y[j-1])
            j -= 1
    # Reverse the strings.
    rx = ''.join(rx)[::-1]
    ry = ''.join(ry)[::-1]
    return [rx, ry]

def kaiming_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.startswith("layers.0"):  # The first layer does not have ReLU applied on its input
            param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))


def get_protein_sequences(protein_ids):
    sequences = {}
    protein_ids = set(protein_ids)
    # Separate UniProt and NCBI IDs
    uniprot_ids = [id for id in protein_ids if not id.startswith('NP_')]
    ncbi_ids = [id for id in protein_ids if id.startswith('NP_')]
    
    # Fetch sequences for UniProt IDs
    if uniprot_ids:
        uniprot_sequences = fetch_uniprot_sequences(uniprot_ids)  # Fetch UniProt sequences
        sequences.update(uniprot_sequences)
    
    # Fetch sequences for NCBI IDs
    if ncbi_ids:
        ncbi_sequences = fetch_ncbi_sequences(ncbi_ids)  # Fetch NCBI sequences
        sequences.update(ncbi_sequences)
    
    return sequences

def fetch_uniprot_sequences(uniprot_ids):
    sequences = {}
    
    for uniprot_id in tqdm(uniprot_ids):
        
        # Make a request to UniProt for the FASTA sequence
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        #url = f'https://www.uniprot.org/uniprot/{uniprot_id}.fasta'
        response = requests.get(url)
        
        if response.ok:
            sequences[uniprot_id] = [''.join(response.text.split('\n')[1:])]
    
    return sequences

def fetch_ncbi_sequences(ncbi_ids):
    Entrez.email = 'd.hasler@campus.lmu.de'  # Set your email address here
    Entrez.api_key = "d0e67a06b3f5a139e29f787339fae9b3d008"
    sequences = {}
    
    def fetch_sequence(ncbi_id):
        handle = Entrez.efetch(db='protein', id=ncbi_id, rettype='fasta', retmode='text')
        record = handle.read()
        handle.close()
        sequences[ncbi_id] = [record.split('\n', 1)[1].replace('\n', '')]
    
    # Fetch sequences using concurrent futures
    for ncbi_id in tqdm(ncbi_ids):
        fetch_sequence(ncbi_id)
    
    return sequences