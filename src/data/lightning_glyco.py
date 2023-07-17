import os
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import h5py
from tqdm import tqdm
from src.data.fasta import Fasta
from dataclasses import dataclass


@dataclass
class GlycoDataConfig():
    data_dir: str = '../../data'
    embedding_path: str = '../../data/glyco/glyco_embeddings.h5'
    np_path: str = '../../data/'
    num_workers: int = 4
    batch_size: int = 32
    classes = {"T": 0, "N": 1, "O": 2}


import torch
from torch.utils.data import BatchSampler

"""
Create a custom batch sampler that groups samples by their protein ID.
This ensures that all samples from the same protein are in the same batch.
The batch should contain the indices of the samples in the batch.
Additionally, the groups should be shuffled and so should the samples within the groups.
You need to fill the batch with groups of samples until the batch is full. A batch can contain samples from multiple proteins.
"""
class GroupedBatchSampler(BatchSampler):
    def __init__(self, protein_ids, batch_size, drop_last=False):
        self.protein_ids = protein_ids
        self.batch_size = batch_size
        self.drop_last = False
        self.batches = None
        self._create_batches()

    def _create_batches(self):
        self.batches = []

        protein_ids, counts = np.unique(self.protein_ids, return_counts=True)
        shuffle_mask = numpy.random.permutation(len(protein_ids))
        protein_ids, counts = protein_ids[shuffle_mask], counts[shuffle_mask]

        batch = np.array([])
        for pid, count in zip(protein_ids, counts):
            indices = np.where(self.protein_ids == pid)[0]
            np.random.shuffle(indices)
            if len(batch) + len(indices) > self.batch_size:
                self.batches.append(batch)
                batch = indices
                
            batch = np.append(batch, indices)
            if len(batch) >= self.batch_size:
                self.batches.append(batch)
                
        np.random.shuffle(self.batches)
    
    def __iter__(self):
        for batch in self.batches:
            yield batch


class GlycoDataModule(pl.LightningDataModule):
    def __init__(self, config: GlycoDataConfig) -> None:
        super().__init__()
        super().__init__()
        self.data_dir = Path(config.data_dir)
        self.np_path = Path(config.np_path)
        self.config = config

        self.class_weights = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.O_test_dataset = None
        self.N_test_dataset = None
        self.class_weights = None
        self.shuffled_ids = None
        

    def prepare_data(self):
        """
        Prepares the data for the data module. In this case, simply checks if the data is available and that the validation set is split.
        Else, it splits the data into train, val.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError("The data directory that was provided does not exist! Please download the data first!")
        
        if not (self.data_dir / "train.o").exists() or not (self.data_dir / "val.o").exists() or not (self.data_dir / "O_test.o").exists():
            raise FileNotFoundError("The data directory that was provided is missing the .o files! Please download the data first!")
        

    def setup(self, stage=None):
        self.prepare_data()
        
        self.train_dataset = GlycoDataset("train", self.config)
        self.val_dataset = GlycoDataset("val", self.config)
        self.O_test_dataset = GlycoDataset("O_test", self.config)
        self.N_test_dataset = GlycoDataset("N_test", self.config)
        self.test_dataset = ConcatDataset([self.O_test_dataset, self.N_test_dataset])

        # Shuffel train data #reproducibility #science
        
        # Create an array of indices that correspond to the order of IDs in 'b'
        if not (self.np_path / "shuffle.npy").exists():
            self.shuffled_ids = np.random.permutation(self.train_dataset.pids)
            np.save(self.np_path / "shuffle.npy", self.shuffled_ids)
        else:
            self.shuffled_ids = np.load(self.np_path / "shuffle.npy", allow_pickle=True)

        index_mapping = {id: index for index, id in enumerate(self.shuffled_ids)}
        sorted_indices = np.zeros(len(self.shuffled_ids), dtype=int)
        for items in index_mapping.items():
            sorted_indices[items[1]] = np.where(items[0] == self.train_dataset.pids)[0][0]
        
        self.train_dataset.X = self.train_dataset.X[sorted_indices]
        self.train_dataset.y = self.train_dataset.y[sorted_indices]
        self.train_dataset.pids = self.shuffled_ids

        if len(self.config.classes) < 3:
            self.train_dataset.y = np.array([arr.astype(np.float16) 
                                             if arr.dtype != np.float16 
                                             else arr 
                                             for arr in self.train_dataset.y], dtype=object)
            self.val_dataset.y = np.array([arr.astype(np.float16)
                                           if arr.dtype != np.float16
                                           else arr
                                           for arr in self.val_dataset.y], dtype=object)
            self.O_test_dataset.y = np.array([arr.astype(np.float16)
                                            if arr.dtype != np.float16
                                            else arr
                                            for arr in self.O_test_dataset.y], dtype=object)
            self.N_test_dataset.y = np.array([arr.astype(np.float16)
                                            if arr.dtype != np.float16
                                            else arr
                                            for arr in self.N_test_dataset.y], dtype=object)
            
                    
        if (self.np_path / f"class_weights_c{len(self.config.classes)}.pt").exists():
            self.class_weights = torch.load(self.np_path / f"class_weights_c{len(self.config.classes)}.pt")
            return
        
        # calculate class weights for the loss function
        ys = np.concatenate((np.apply_along_axis(np.concatenate, 0, self.train_dataset.y),
                             np.apply_along_axis(np.concatenate, 0, self.val_dataset.y),
                             np.apply_along_axis(np.concatenate, 0, self.O_test_dataset.y),
                             np.apply_along_axis(np.concatenate, 0, self.N_test_dataset.y)), axis=0)
        counts = np.unique(ys, return_counts=True)[1]
        # check if class weights are already calculated

        if len(self.config.classes) < 3:
            # For binary predictions set only positive weight
            self.class_weights = torch.tensor([counts[0] / counts[1]], dtype=torch.float16)
        else:
            self.class_weights = torch.tensor([max(counts) / counts[i] for i in range(len(self.config.classes))], dtype=torch.float32)
        torch.save(self.class_weights, self.np_path / f"class_weights_c{len(self.config.classes)}.pt")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)
    
    def O_test_dataloader(self):
        return DataLoader(self.O_test_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)
    
    def N_test_dataloader(self):
        return DataLoader(self.N_test_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)

    

# TODO add multiple dataset classes 
# one for the baseline -> simply use the embedding of the glyco site 
# one for the baseline + dialated mean embedding -> use the embedding of the glyco site and the dialated mean embedding e.g. 101010N010101
class GlycoDataset(Dataset):
    def __init__(self, split: str, config: GlycoDataConfig) -> None:
        super().__init__()
        self.split = split
        self.data_dir = Path(config.data_dir)
        self.embedding_path = config.embedding_path
        self.np_path = Path(config.np_path)
        self.num_classes = len(config.classes)
        self.to_classes = np.vectorize(config.classes.get)
        self.config = config

        self.X = None
        self.y = None
        self.pids = None
        self.load_data()
    
    def load_data(self):
        if self.split == "blind_test":
            raise NotImplementedError("Blind test set not implemented yet!")
        try:
            self.X = np.load(str(self.np_path / f"{self.split}_X.npy"), allow_pickle=True)
            self.y = np.load(str(self.np_path / f"{self.split}_y_c{self.num_classes}.npy"), allow_pickle=True)
            self.pids = np.load(str(self.np_path / f"{self.split}_pids.npy"), allow_pickle=True)
            return
        except:
            print("Creating numpy arrays...")

        fasta = Fasta(self.data_dir / f"{self.split}.o")
        embeddings = h5py.File(self.embedding_path, 'r')
        X = []
        y = []
        pids = []
        classes = np.array(list(self.config.classes.keys()))
        for pid, seqs in tqdm(fasta.items()):
            labels = np.array(list(seqs[1]))
            samples = np.isin(labels, classes)
            try:
                embedding = embeddings[pid.replace("-", "_").replace(".", "_")][()]
            except:
                print(f"Protein {pid}  not found in embeddings!")
                continue
            X.append(embedding[samples])
            y.append(self.to_classes(labels[samples]))
            pids.append(pid)

        self.X = np.array(X, dtype=object)
        self.y = np.array(y, dtype=object)
        self.pids = np.array(pids, dtype=object)
        np.save(str(self.np_path / f"{self.split}_X.npy"), self.X)
        np.save(str(self.np_path / f"{self.split}_y_c{self.num_classes}.npy"), self.y)
        np.save(str(self.np_path / f"{self.split}_pids.npy"), self.pids)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.pids[index]
    
    

            