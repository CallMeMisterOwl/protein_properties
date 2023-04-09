import concurrent.futures
import gzip
import json
import math
import random
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
import pandas as pd
import requests
import scipy.stats
import torch
from Bio import Entrez, SeqIO
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Seq import Seq
from biotite.sequence import AlphabetError, ProteinSequence
from biotite.structure.io.pdbx import CIFFile, get_sequence, get_structure
from pytorch_lightning import seed_everything
from tqdm import tqdm

from src.data.fasta import Fasta

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import multiprocessing as mp
import os
from multiprocessing.pool import Pool

QUERY = """
query {
  entry(entry_id: "YOUR_PDB_ID_HERE") {
    polymer_entities {
      entity_poly {
        pdbx_seq_one_letter_code
      }
      rcsb_polymer_entity_container_identifiers {
        auth_asym_ids
      }
    }
  }
}
"""
URL_PDB = "https://data.rcsb.org/graphql"
HOA_TIEN = {'undefined': -1, "A": 121, "R": 265, "N": 187, "D": 187, "C": 148, "E": 214, "Q": 214, "G": 97, "H": 216, "I": 195, "L": 191, "K": 230, "M": 203, "F": 228, "P": 154, "S": 143, "T": 163, "W": 264, "X": 180,"Y": 255, "V": 165, "U": 148, "O": 230}
TO_RSA = np.vectorize(HOA_TIEN.get)
SUBSTITUTION_DICT = {"CYG": "C", "TRN": "W", "IAS": "D", "CSD": "C", "CSO": "C", "TRO": "W", "CSS": "C", "SEP": "S", "DDZ": "A", 
                     "PCA": "E", "CGU": "E", "OCS": "C", "TYI": "Y", "LLP": "K", "CXM": "M", "KCX": "K", "AYA": "A", "TRW": "W", 
                     "CME": "C", "NEP": "H", "CAS": "C", "CSX": "C", "TPQ": "Y", "NLE": "L", "LYZ": "K", "SEB": "S", "LED": "L", 
                     "CAF": "C", "MCS": "C", "CS4": "C", "TYS": "Y", "SVY": "S", "MLZ": "K", "DAH": "F", "TY2": "Y", "KYN": "W", 
                     "LP6": "K", "FME": "M", "ALY": "K", "LCK": "K", "LA2": "K", "MSO": "M", "KPI": "K", "MEQ": "Q", "HIC": "H", 
                     "LVN": "V", "MHO": "M", "SNN": "N", "TPO": "T", "IYR": "Y", "TRQ": "W", "QCS": "C", "SME": "M", "ASB": "D", 
                     "0AF": "W", "CCS": "C", "SCH": "C", "GPL": "K", "LYR": "K", "MHS": "H", "AGM": "R", "MGN": "Q", "GL3": "G", 
                     "DYA": "D", "SMC": "C", "SAC": "S", "YCM": "C", "OCY": "C", "PHI": "F", "ALS": "A", "PTR": "Y", "ALO": "T", 
                     "GLU": "N", "ALA": "H", "HYP": "P", "SNC": "C", "6V1": "C", "BFD": "D", "PYL": "K", "SEC": "C", "AIB": "A", 
                     "PHL": "F", "DPR": "P", "DBZ": "A", "DAL": "A", "MLY": "K"}

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
    Entrez.email = ''  # Set your email address here
    Entrez.api_key = ""
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

def read_vespag(pid: str, data_dir: str):
    try: 
        vespag_df = pd.read_csv(f"{data_dir}/{pid}.csv")
    except FileNotFoundError:
        print(f"File {data_dir}/{pid}.csv not found")
        return None
    
    vespag_df[['AA', 'Position', 'Mutation']] = pd.DataFrame(vespag_df['Mutation']
                                                             .apply(lambda x: [x[0], int(x[1:-1]), x[-1]]).tolist(), 
                                                              columns=['AA', 'Position', 'Mutation'])
    return vespag_df

def fetch_pdb_sequence(pdb_id, label_asym_id) -> str:
    query = QUERY.replace("YOUR_PDB_ID_HERE", pdb_id)
    payload = {
    "query": query
    }
    # Send the request
    response_pdb = requests.post(URL_PDB, json=payload)
    if response_pdb.status_code == 200:
        data = response_pdb.json()
        
        # Check if data is not None and has the expected structure
        if data['data'] and data['data']['entry'] and data['data']['entry']['polymer_entities']:
            # Extract and process data for each polymer entity
            for entity in data['data']['entry']['polymer_entities']:
                sequence = entity['entity_poly']['pdbx_seq_one_letter_code']
                chain_ids = entity['rcsb_polymer_entity_container_identifiers']['auth_asym_ids']
                
                for chain_id in chain_ids:
                    if chain_id != label_asym_id:
                        continue
                    # Prepare header for the chain
                    return sequence
    print(f"Could not fetch sequence for {pdb_id} and {label_asym_id}")
    print(response_pdb.text)

def get_auth_to_label_asym_mapping(cif_file):
    # Parse the mmCIF file
    # if gzipped file
    if isinstance(cif_file, str):
        cif_file = Path(cif_file)
    if '.gz' in cif_file.suffixes:
        with gzip.open(cif_file, 'rt') as f:
            cif_dict = MMCIF2Dict(f)
    else:
        cif_dict = MMCIF2Dict(cif_file)
    
    # Access the mmCIF dictionary from the structure

    # Extract the label_asym_id and auth_asym_id mapping
    label_asym_ids = cif_dict["_atom_site.label_asym_id"]
    auth_asym_ids = cif_dict["_atom_site.auth_asym_id"]
    
    # Create a dictionary to store unique mappings
    mapping = {}
    
    # Iterate over atoms and map label_asym_id to auth_asym_id
    for label_asym, auth_asym in zip(label_asym_ids, auth_asym_ids):
        if auth_asym not in mapping:  # Avoid duplicates
            mapping[auth_asym] = label_asym
    
    return mapping

def get_relative_sa(seq, sasa):
    seq = np.array(list(seq))
    try:
        hoa = TO_RSA(seq)
    except TypeError:
        print(
            f"No max SA value for the residue {set(list(seq)).difference(HOA_TIEN.keys())}"
        )
        seq[seq == None] = "undefined"
        try:
            hoa = TO_RSA(seq)
        except TypeError:
            print(
                f"No max SA value for the residue {set(list(seq)).difference(HOA_TIEN.keys())}"
            )
            return None
    return sasa / hoa

def get_pdb_structure(protein, cif_dir, split_symbol="_"):
    cif_header, chain_id = protein.split(split_symbol)
    cif = None
    if cif_dir is not None:
        cif_dir = Path(cif_dir)
        cif = cif_dir / cif_header.lower()[1:3] / f"{cif_header.lower()}.cif"
        if cif.with_suffix(".cif.gz").exists():
            cif = cif.with_suffix(".cif.gz")
            
        elif not cif.exists():
            cif = None
            
    if cif is None:
        try:
            cif = PDB.PDBList().retrieve_pdb_file(cif_header.upper(), pdir=gettempdir(), file_format='mmCif')
        except Exception as e:
            print(f"Could not fetch PDB file for {protein}\n{e}")
            return None, None
        
    if '.gz' in Path(cif).suffixes:
        try:
            with gzip.open(
                cif, "rt"
            ) as f:
                cif_obj = CIFFile().read(f)
                struct = get_structure(cif_obj, model=1, extra_fields=["b_factor"])
        except ValueError:
            print(f"Skipping protein {protein}...\nCould not parse structure")
            return None, None 
    else:
        try:
            with open(cif, "r") as f:
                cif_obj = CIFFile().read(f)
                struct = get_structure(cif_obj, model=1, extra_fields=["b_factor"])
        except ValueError:
            print(f"Skipping protein {protein}...\nCould not parse structure")
            return None, None 
        except FileNotFoundError:
            print(f"Skipping protein {protein}...\nCould not find CIF file")
            return None, None
    return struct, cif

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h
