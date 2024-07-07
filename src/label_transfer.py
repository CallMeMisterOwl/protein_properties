import multiprocessing
from dataclasses import dataclass
from typing import Any, Literal, Optional

import h5py
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from Bio.Align import PairwiseAligner, substitution_matrices
from torch import Tensor, nn
from torchmetrics.functional.classification import accuracy, f1_score, matthews_corrcoef
from torchmetrics.functional.regression import mean_absolute_error, pearson_corrcoef

#cosine vs euclidean https://medium.com/@sasi24/cosine-similarity-vs-euclidean-distance-e5d9a9375fc8

class LabelTransferBaseline(nn.Module):
    """
    Model for label transfer using HBI and EAT.
    """
    def __init__(self, lookup_pids: Tensor, lookup_per_prot_emb: Tensor, rsa_fasta_path: str, top_k: int = 10):
        """
        Args:
            lookup_pids: Tensor of protein ids.
            lookup_per_prot_emb: Tensor of protein embeddings. Same order as pids.
            
            rsa_fasta_path: Path to fasta file containing protein sequences and RSA score per residue (in that order).
        """
        super().__init__()
        self.lookup_pids = lookup_pids
        self.lookup_per_prot_emb = lookup_per_prot_emb
        self.rsa_fasta = Fasta(rsa_fasta_path)
        self.top_k = top_k
        self.aligner = self.initialize_aligner()
        
    
    def forward(self, x: Tensor, pid: str, rr: int) -> Tensor:
        """
        Transfers per-protein labels to query protein using HBI and EAT.
        Also it uses 3 methods to generate labels for the query protein:
        1. Top 1 protein with highest sequence similarity score. Simply align and tranfer label for similar parts
        2. Top k proteins with highest sequence similarity score. Align and transfer label for similar parts. Average the labels.
        Hence every query protein will have 9 possible labels transfers.
        """
        
        # calculate similarity scores
        seq_sim= self.calculate_sequence_similarity(pid, lookup_pids)
        
        # calculate distances
        euclidean = self.calculate_euclidean(x, lookup_per_prot_emb)
        
        
        # transfer labels using HBI
        def hbi():
            top_k_idx = torch.topk(seq_sim, self.top_k).indices
            top_label: Tensor = None
            top_k_labels = []
            for i, top in enumerate(lookup_pids[top_k_idx]):
                label = self.transfer_labels(pid, top)
                if i == 0:
                    top_label = label
                top_k_labels.append(label)

            top_k_labels = torch.stack(top_k_labels)
            top_k_avg_label = torch.mean(top_k_labels, dim=0)
            top_k_median_label = torch.median(top_k_labels, dim=0)
            return top_label, top_k_avg_label, top_k_median_label
        
        # transfer labels using EAT
        def eat():
            top_k_idx = torch.topk(euclidean, self.top_k).indices
            top_label: Tensor = None
            top_k_labels = []
            for i, top in enumerate(lookup_pids[top_k_idx]):
                label = self.transfer_labels(pid, top)
                if i == 0:
                    top_label = label
                top_k_labels.append(label)

            top_k_labels = torch.stack(top_k_labels)
            top_k_avg_label = torch.mean(top_k_labels, dim=0)
            top_k_median_label = torch.median(top_k_labels, dim=0)
            return top_label, top_k_avg_label, top_k_median_label

        true_label = self.rsa_fasta[pid][1]
        metrics = []
        trans_labels = hbi() + eat()
        for trans_label in trans_labels:
            metrics.append((pearson_corrcoef(trans_label, true_label), mean_absolute_error(trans_label, true_label)))
        return trans_labels, metrics
    

    def transfer_labels(self, query_id: str, lookup_id: str) -> Tensor:
        """
        Transfer labels from lookup proteins to query protein using HBI and EAT.
        """
        # get sequences
        query_seq = self.rsa_fasta[query_id][0]
        seq = self.seqs[lookup_id][0]
        alignments = self.aligner.align(query_seq, seq)
        best_alignment = None
        best_ss_score = 0
        for a in alignments: # find best alignment -> highest sequence similarity score
            seq_similarity = a.counts[1] / len(a.indices[0]) * 100
            best_ss_score = seq_similarity if seq_similarity > best_ss_score else best_ss_score
            best_alignment = a if seq_similarity == best_ss_score else best_alignment
        assert best_alignment is not None

        # transfer labels
        labels = self.rsa_fasta[lookup_id][1]
        trans_labels = tensor.zeros(len(labels))
        for idx in zip(best_alignment.aligned[0], best_alignment.aligned[1]):
            trans_labels[idx[0]] = labels[idx[1]]

        return trans_labels
        

    def calculate_euclidean(self, query: Tensor, lookup: Tensor) -> Tensor:
        lookup = torch.unsqueeze(lookup, 0)
        return (lookup - query).pow(2).sum(1).sqrt()
    
    def calculate_sequence_similarity(self, query_id: str, lookup_ids: list[str]) -> list[float]:
        """
        Calculate sequence similarity between query protein and lookup proteins. 
        Sequence similarity and identity are used interchangeably, even though they are not the same.
        Uniqueprot uses sequence identity -> we should be fine 
        """
        # data types
        query_seq = self.rsa_fasta[query_id][0]
        lookup_seqs = [self.seqs[id][0] for id in lookup_ids]
        scores = []
        
        def calculate_similarity(seq):
            alignments = self.aligner.align(query_seq, seq)
            best_ss_score = 0
            a = alignments[0]
            seq_similarity = a.counts().identities / len(a.indices[0]) * 100
            return seq_similarity
        
        with multiprocessing.Pool() as pool:
            scores = pool.map(calculate_similarity, lookup_seqs)
        
        assert len(scores) == len(lookup_ids) # check if all lookup proteins have a score
        return scores
      
    def initialize_aligner(self) -> PairwiseAligner:
        with open("BLOSUM62") as handle:
            blosum_matrix = substitution_matrices.read(handle)
        aligner = PairwiseAligner(scoring="blastp", mode = 'global')
        aligner.substitution_matrix = blosum_matrix
        


        

    
        
