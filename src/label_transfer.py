from dataclasses import dataclass
from typing import Any, Literal
import h5py
import lightning.pytorch as pl
import torch.nn.functional as F
import torch
from torch import nn
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics.functional.classification import f1_score, matthews_corrcoef, accuracy
from typing import Optional
import pytorch 
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices



#cosine vs euclidean https://medium.com/@sasi24/cosine-similarity-vs-euclidean-distance-e5d9a9375fc8

class LabelTransferBaseline(nn.Module):
    """
    Model for label transfer using HBI and EAT.
    """
    def __init__(self, pids: Tensor, per_prot_emb: Tensor, rr_30: Tensor, rr_50: Tensor, rr_80: Tensor, seqs_path: str):
        """
        Args:
            pids: Tensor of protein ids.
            per_prot_emb: Tensor of protein embeddings.
            rr_30: Tensor with indices of 30% redundancy reduced proteins.
            rr_50: Tensor with indices of 50% redundancy reduced proteins.
            rr_80: Tensor with indices of 80% redundancy reduced proteins.
            seqs_path: Path to h5py file containing protein sequences.
        """
        super().__init__()
        self.pids = pids
        self.per_prot_emb = per_prot_emb
        self.rr_30 = rr_30
        self.rr_50 = rr_50
        self.rr_80 = rr_80
        self.seqs = h5py.File(seqs_path, 'r')
        self.aligner = self.initialize_aligner()
        
    
    def forward(self, x: Tensor, pid: str) -> Tensor:
        pass

    def calculate_euclidean(self, query: Tensor, lookup: Tensor) -> Tensor:
        lookup = torch.unsqueeze(lookup, 0)
        return (lookup - query).pow(2).sum(1).sqrt()
    
    def calculate_sequence_similarity(self, query: str, lookup: str) -> float:
        # data types
        query_seq = self.seqs[query][:]
        lookup_seqs = self.seqs[lookup][:]
        scores = []
        for idx, seq in enumerate(lookup_seqs):
            alignments = self.aligner.align(query_seq, seq)
            local_score = []
            for a in alignments: 
                # this is currently sequence identity -> sequence similiarity might be better
                # for this one would need to check the blosum matrix 
                seq_similarity = a.counts[1] / len(a.indices[0]) * 100
                


                
    
    def initialize_aligner(self) -> PairwiseAligner:
        with open("BLOSUM62") as handle:
            blosum_matrix = substitution_matrices.read(handle)
        aligner = PairwiseAligner(scoring="blastp", mode = 'global')
        


        

    
        
