import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .fasta import Fasta


@dataclass
class BFactorDataConfig:
    """
    Data configuration for Bfactor dataset
    """
    data_dir: str = 'data/e_prsa/bfactor'
    embedding_path: str = 'data/e_prsa/prott5_sasa_bfactor.h5'
    esm_embedding_path: str = 'data//e_prsa/esm_sasa_bfactor.h5'
    np_path: str = 'data/e_prsa/bfactor/np'
    num_workers: int = 4


class DynamicBatchDataLoader(DataLoader):
    def __init__(self, dataset, max_batch_length,*args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.max_batch_length = max_batch_length

    def batch_size_fn(self, batch):
    # Calculate the total length of all proteins in the batch
        total_length = sum(len(protein) for protein in batch)

        # Determine the batch size based on the total length
        if total_length <= self.max_batch_length:
            return len(batch)  # Use all proteins in the batch
        else:
            # Find the largest subset of proteins that fits within the length limit
            cumulative_length = 0
            for idx, protein in enumerate(batch):
                cumulative_length += len(protein)
                if cumulative_length > self.max_batch_length:
                    return idx  # Return the index to split the batch

        return len(batch) 

    def __iter__(self):
        self.batch_sampler = self.batch_sampler.__iter__()
        return self

    def __next__(self):
        indices = next(self.batch_sampler)
        batch = [self.dataset[i] for i in indices]
        batch_size = self.batch_size_fn(batch)

        # Adjust batch size dynamically
        self.batch_sampler.batch_size = batch_size
        self.batch_sampler.drop_last = False  # Optional: Adjust drop_last based on batch size

        return batch

# One could implement a more memory efficient version of this, but this is fine for now
class BFactorDataset(Dataset):
    """
    Bfactor dataset class for PyTorch Lightning module data loaders and data module. 
    """
    def __init__(self, split: Literal["train", "test", "val", "blind_test"], config: BFactorDataConfig):
    
        super().__init__()
        self.split = split
        self.data_dir = Path(config.data_dir)
        self.embedding_path = config.embedding_path
        self.esm_embedding_path = config.esm_embedding_path
        self.np_path = Path(config.np_path)

        self.X = None
        self.y = None
        self.pids = None
        self.load_data()

    def load_data(self):
        if self.split == "blind_test":
            raise NotImplementedError("Blind test set not implemented yet!")
        
        try:
            self.X = np.load(str(self.np_path / f"{self.split}_X.npy"), allow_pickle=True)
            self.y = np.load(str(self.np_path / f"{self.split}_y.npy"), allow_pickle=True)
            self.pids = np.load(str(self.np_path / f"{self.split}_pids.npy"), allow_pickle=True)
            return
        except:
            print("Creating numpy arrays...")

        label_df = pd.read_csv(self.data_dir / f"{self.split}.tsv", sep="\t")
        
        X = []
        y = []
        pids = []
        for pid in tqdm(set(label_df['Protein'])):
            # masking the 0.0 values, so I can remove them later before calculating the loss
            bfactor = label_df[label_df['Protein'] == pid]['norm_Bfactor'].values
            
            y.append(bfactor.astype(np.float32))
            embedding = None
            with h5py.File(self.embedding_path, "r") as embeddings:
                try:
                    embedding = embeddings[pid][()]
                except KeyError:
                    print(f"Protein {pid} not found in ProtT5 embeddings!")
                    continue
                except IndexError:
                    print(f"Position {pos} not found in protein {pid}!")
                    continue
            
            with h5py.File(self.esm_embedding_path, "r") as esm_embeddings:
                
                try:
                    embedding = np.concatenate([embedding, esm_embeddings[pid.replace("_", "-")][()]], axis=1)
                except KeyError:
                    print(f"Protein {pid} not found in ESM embeddings!")
                    continue
                except IndexError:
                    print(f"Position {pos} not found in protein {pid}!")
                    continue
            
            assert len(embedding) == len(bfactor), f"Length of embedding and RSA is not equal for {pid}"
            X.append(embedding)
            pids.append(pid)
           
        self.X = np.array(X, dtype=object)
        self.y = np.array(y, dtype=object)
        self.pids = np.array(pids, dtype=object)
        np.save(str(self.np_path / f"{self.split}_X.npy"), self.X)
        np.save(str(self.np_path / f"{self.split}_y.npy"), self.y)
        np.save(str(self.np_path / f"{self.split}_pids.npy"), self.pids)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.pids[idx]

    def __len__(self):
        return len(self.X)

class BFactorDataModule(pl.LightningDataModule):
    """
    Data module for the Bfactor dataset. 
    Manages the data loaders and data splits.
    """
    def __init__(self, config: BFactorDataConfig):
        super().__init__()
        self.data_dir = Path(config.data_dir)
        self.np_path = Path(config.np_path)
        self.config = config

        self.class_weights = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.shuffled_ids = None

    def prepare_data(self):
        """
        Prepares the data for the data module. In this case, simply checks if the data is available and that the validation set is split.
        Else, it splits the data into train, val.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError("The data directory that was provided does not exist! Please download the data first!")
        
        if not (self.data_dir / "train.tsv").exists():
            raise FileNotFoundError("The data directory that was provided is missing the tsv files! Please download the data first!")
        
        print("Data preparation done!")
        

    def setup(self, stage=None):
        if stage != 'fit':
            return
        self.prepare_data()
        
        self.train_dataset = BFactorDataset("train", self.config)
        self.val_dataset = BFactorDataset("val", self.config)

        '''if self.config.num_classes < 3:
            self.train_dataset.y = np.array([arr.astype(np.float16) 
                                             if arr.dtype != np.float16 
                                             else arr 
                                             for arr in self.train_dataset.y], dtype=object)
            self.val_dataset.y = np.array([arr.astype(np.float16)
                                           if arr.dtype != np.float16
                                           else arr
                                           for arr in self.val_dataset.y], dtype=object)'''
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)
    



    