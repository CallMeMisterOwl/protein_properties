from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal
from sklearn.model_selection import train_test_split
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split, Dataset
import h5py
import numpy as np
from tqdm import tqdm

from .fasta import Fasta

# I love copilot <3
# substituted O for K
# substituted U for C
hoa_tien = {"A": 121, "R": 265, "N": 187, "D": 187, "C": 148, "E": 214, "Q": 214, "G": 97, "H": 216, "I": 195, "L": 191, "K": 230, "M": 203, "F": 228, "P": 154, "S": 143, "T": 163, "W": 264, "X": 180,"Y": 255, "V": 165, "U": 148, "O": 230}

@dataclass
class SASADataConfig:
    """
    Data configuration for SASA dataset
    """
    data_dir: str = '../../data'
    embedding_path: str = '../../data/sasaembeddings.h5'
    np_path: str = '../../data/'
    num_classes: Literal[1,2,3,10] = 3
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
class SASADataset(Dataset):
    """
    SASA dataset class for PyTorch Lightning module data loaders and data module. 
    Uses relative solvent accessibility (RSA) as labels.
    """
    def __init__(self, split: Literal["train", "test", "val", "blind_test"], config: SASADataConfig):
        """
        
        :param split: str
            Split to use. One of 'train', 'val', 'test'
        :param config: SASADataConfig
            Data configuration
        """
        super().__init__()
        self.split = split
        self.data_dir = Path(config.data_dir)
        self.embedding_path = config.embedding_path
        self.np_path = Path(config.np_path)
        self.num_classes = config.num_classes

        self.to_rsa = np.vectorize(hoa_tien.get)

        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        if self.split == "blind_test":
            raise NotImplementedError("Blind test set not implemented yet!")
        
        try:
            self.X = np.load(str(self.np_path / f"{self.split}_X.npy"), allow_pickle=True)
            self.y = np.load(str(self.np_path / f"{self.split}_y_c{self.num_classes}.npy"), allow_pickle=True)
            return
        except:
            print("Creating numpy arrays...")

        fasta = Fasta(self.data_dir / f"{self.split}.o")
        embeddings = h5py.File(self.embedding_path, 'r')
        X = []
        y = []
        for pid, seqs in tqdm(fasta.items()):
            rsa = self.get_relative_sa(seqs[0], seqs[1]).astype(np.float32)
            # masking the 0.0 values, so I can remove them later before calculating the loss
            rsa = np.where(rsa == 0.0, -1, rsa)
            # Regression task
            if self.num_classes == 1:
                pass
            # class 0 if below 16%, class 1 above or equal 16% (as described in the paper Rost and Sander (1994))
            elif self.num_classes == 2:
                rsa[rsa != -1] = np.where(rsa[rsa != -1] >= 0.16, 1, 0)
            # class 0 if below 9%, class 1 between 9% and 36%, class 2 above or equal 36%
            elif self.num_classes == 3:
                rsa[rsa != -1] = np.where(rsa[rsa != -1] >= 0.36, 2, np.where(rsa[rsa != -1] >= 0.09, 1, 0))
            # Ten-state class
            elif self.num_classes == 10:
                # clipping the values to 1.0 -> it can happen that the rsa is larger than 1.0 since the highest observed values per aa are not 100% accurate
                # this messes with the formular and produces more than 10 classes -> 
                rsa[rsa != -1] = np.clip(rsa[rsa != -1], 0.0, 1.0)
                rsa[rsa != -1] = np.clip(np.trunc(np.sqrt(rsa[rsa != -1] * 100)), 0, 9)
            else:
                raise ValueError("Invalid number of classes!\nValid values are 2, 3 and 10.")
            y.append(rsa.astype(np.int64))
            # vespa replaces "-" with "_" in the ids -.-
            e = embeddings[pid.replace("-", "_") if "-" in pid else pid][()]
            assert len(e) == len(rsa), f"Length of embedding and RSA is not equal for {pid}"
            X.append(e)

        
        self.X = np.array(X, dtype=object)
        self.y = np.array(y, dtype=object)
        np.save(str(self.np_path / f"{self.split}_X.npy"), self.X)
        np.save(str(self.np_path / f"{self.split}_y_c{self.num_classes}.npy"), self.y)
        
    def get_relative_sa(self, seq, sasa):
        sasa = np.array(sasa)
        try:
            hoa = self.to_rsa(np.array(list(seq)))
        except TypeError:
            print(f"No max SA value for the residue {set(list(seq)).difference(hoa_tien.keys())}")
            raise
        return sasa / hoa

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

class SASADataModule(pl.LightningDataModule):
    """
    Data module for the SASA dataset. 
    Manages the data loaders and data splits.
    """
    def __init__(self, config: SASADataConfig):
        super().__init__()
        self.data_dir = Path(config.data_dir)
        self.config = config

        self.class_weights = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Prepares the data for the data module. In this case, simply checks if the data is available and that the validation set is split.
        Else, it splits the data into train, val.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError("The data directory that was provided does not exist! Please download the data first!")
        
        if not (self.data_dir / "train.o").exists() or not (self.data_dir / "blind_test.o").exists() or not (self.data_dir / "test.o").exists():
            raise FileNotFoundError("The data directory that was provided is missing the .o files! Please download the data first!")
        
        if (self.data_dir / "val.o").exists():
            print("Data preparation already done!")
            return
        
        print("Preparing data...")
        # Splitting the data into train, val and test set
        fasta = Fasta(self.data_dir / "train.o")
        train, val = train_test_split(fasta.get_headers(), test_size=0.1, random_state=13, shuffle=True)
        filterByKey = lambda keys: {x: fasta[x] for x in keys}
        train_dict = filterByKey(train)
        val_dict = filterByKey(val)
        Fasta(sequences=train_dict).write_fasta(self.data_dir / "train.o", overwrite=True)
        Fasta(sequences=val_dict).write_fasta(self.data_dir / "val.o", overwrite=True)
        print("Data preparation done!")
        # IDs to shuffle
        """random.shuffle(self.vids["train"])
        np.save(OUT_PATH / "shuffle", self.vids["train"])"""
            

    def setup(self, stage=None):
        self.prepare_data()
        
        self.train_dataset = SASADataset("train", self.config)
        self.val_dataset = SASADataset("val", self.config)
        self.test_dataset = SASADataset("test", self.config)

        if self.config.num_classes < 3:
            
            self.train_dataset.y = np.array([arr.astype(np.float16) 
                                             if arr.dtype != np.float16 
                                             else arr 
                                             for arr in self.train_dataset.y], dtype=object)
            self.val_dataset.y = np.array([arr.astype(np.float16)
                                           if arr.dtype != np.float16
                                           else arr
                                           for arr in self.val_dataset.y], dtype=object)
            self.test_dataset.y = np.array([arr.astype(np.float16)
                                            if arr.dtype != np.float16
                                            else arr
                                            for arr in self.test_dataset.y], dtype=object)
            
                    
        if (Path(self.data_dir) / f"class_weights_c{self.config.num_classes}.pt").exists():
            self.class_weights = torch.load(Path(self.data_dir) / f"class_weights_c{self.config.num_classes}.pt")
            return
        
        # calculate class weights for the loss function
        ys = np.concatenate((np.apply_along_axis(np.concatenate, 0, self.train_dataset.y),
                             np.apply_along_axis(np.concatenate, 0, self.val_dataset.y),
                             np.apply_along_axis(np.concatenate, 0, self.test_dataset.y)), axis=0)
        counts = np.unique(ys, return_counts=True)[1]
        # check if class weights are already calculated

        if self.config.num_classes < 3:
            # For binary predictions set only positive weight
            self.class_weights = torch.tensor([counts[0] / counts[1]], dtype=torch.float16)
        else:
            self.class_weights = torch.tensor([max(counts) / counts[i] for i in range(self.config.num_classes)], dtype=torch.float32)
        torch.save(self.class_weights, self.data_dir / f"class_weights_c{self.config.num_classes}.pt")


        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)

    def blind_test_dataloader(self):
        raise NotImplementedError("Blind test set not implemented yet!")
        
    