import os
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import h5py
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class GlycoDataConfig():
    classes: dict
    data_dir: str = '../../data'
    embedding_path: str = '../../data/glyco/glyco_embeddings.h5'
    np_path: str = '../../data/np'
    num_workers: int = 4
    batch_size: int = 64
    add_neg_sites: bool = False
    use_neg_diff: bool = False
    

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
    def __init__(self, protein_ids, batch_size: int, drop_last: bool = False, shuffle: bool = False):
        self.protein_ids = protein_ids
        self.batch_size = batch_size
        if drop_last:
            NotImplementedError("Drop last not implemented yet!")
        self.drop_last = False
        self.batches = None
        self.shuffle = shuffle
        self._create_batches()

    def _create_batches(self):
        self.batches = []
        protein_ids, counts = np.unique(self.protein_ids, return_counts=True)
        if self.shuffle:
            shuffle_mask = np.random.permutation(len(protein_ids))
            protein_ids, counts = protein_ids[shuffle_mask], counts[shuffle_mask]

        batch = []
        for pid, count in zip(protein_ids, counts):
            indices = np.where(self.protein_ids == pid)[0].astype(int).tolist()
            np.random.shuffle(indices)
            if len(batch) + len(indices) > self.batch_size:
                
                if not len(indices) >= self.batch_size:
                    self.batches.append(batch)
                    batch = indices
                
            batch.extend(indices)
            if len(batch) >= self.batch_size:
                self.batches.append(batch)
                
        np.random.shuffle(self.batches)
    
    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)

# !TODO: Subset doesn't support the custom class variables like pids or classes etc.
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
        
        """if not (self.data_dir / "train.csv").exists() or not (self.data_dir / "val.csv").exists() or not (self.data_dir / "O_test.csv").exists():
            raise FileNotFoundError("The data directory that was provided is missing the .o files! Please download the data first!")"""
        

    def setup(self, stage=None):
        self.prepare_data()
        
        og_train_dataset = GlycoDataset("train", self.config, 42, False)
        # split the train dataset into train and val
        labels = []
        for l, a in zip(og_train_dataset.y, og_train_dataset.AAs):
            labels.append(l if a == "N" and l == 0 else 3)
        train_idx, val_idx = next(GroupShuffleSplit(test_size=0.1, random_state=42).split(og_train_dataset.X, labels, og_train_dataset.pids)) 
        self.train_dataset = Subset(og_train_dataset, train_idx)
        self.val_dataset = Subset(og_train_dataset, val_idx) 
        
        #self.val_dataset = GlycoDataset("val", self.config)
        """self.O_test_dataset = GlycoDataset("O_test", self.config, 370381)
        self.N_test_dataset = GlycoDataset("N_test", self.config)
        self.test_dataset = ConcatDataset([self.O_test_dataset, self.N_test_dataset])"""


        if len(self.config.classes) < 3:
            og_train_dataset.y = np.array([arr.astype(np.float16) 
                                             if arr.dtype != np.float16 
                                             else arr 
                                             for arr in og_train_dataset.y])

            
                    
        if (self.np_path / f"class_weights_c{len(self.config.classes)}_{'_'.join(self.config.classes.keys())}.pt").exists():
            self.class_weights = torch.load(self.np_path / f"class_weights_c{len(self.config.classes)}_{'_'.join(self.config.classes.keys())}.pt")
            return
        
        # calculate class weights for the loss function
        #ys = np.apply_along_axis(np.concatenate, 0, og_train_dataset.y)
        counts = np.unique(og_train_dataset.y, return_counts=True)[1]
        # check if class weights are already calculated

        if len(self.config.classes) < 3:
            # For binary predictions set only positive weight
            self.class_weights = torch.tensor([counts[0] / counts[1]], dtype=torch.float16)
        else:
            self.class_weights = torch.tensor([max(counts) / counts[i] for i in range(len(self.config.classes))], dtype=torch.float32)
        torch.save(self.class_weights, self.np_path / f"class_weights_c{len(self.config.classes)}_{'_'.join(self.config.classes.keys())}.pt")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.config.num_workers, batch_sampler=GroupedBatchSampler(self.train_dataset.dataset.pids[self.train_dataset.indices], self.config.batch_size))
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)
    
    """def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)
    
    def O_test_dataloader(self):
        return DataLoader(self.O_test_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)
    
    def N_test_dataloader(self):
        return DataLoader(self.N_test_dataset, batch_size=1, shuffle=False, num_workers=self.config.num_workers)
"""
    

# TODO add multiple dataset classes 
# one for the baseline -> simply use the embedding of the glyco site 
# one for the baseline + dialated mean embedding -> use the embedding of the glyco site and the dialated mean embedding e.g. 101010N010101
class GlycoDataset(Dataset):
    def __init__(self, split: str, config: GlycoDataConfig, seed: int = None, generate_numpy: bool = True) -> None:
        super().__init__()
        self.split = split
        self.data_dir = Path(config.data_dir)
        self.embedding_path = config.embedding_path
        self.np_path = Path(config.np_path)
        self.num_classes = len(config.classes)
        self.to_classes = np.vectorize(config.classes.get)
        self.config = config
        self.add_neg_sites = config.add_neg_sites
        if self.add_neg_sites and self.num_classes < 3 and self.split == "train":
            self.split = "train_more_neg"
        self.use_neg_diff = config.use_neg_diff if self.split == "train" else False

        self.seed = seed if seed else random.randint(0, 100000)
        self.numpy = generate_numpy
        
        self.X = None
        self.y = None
        self.pids = None
        self.load_data()
    

    def load_data(self):
        if self.split == "blind_test":
            raise NotImplementedError("Blind test set not implemented yet!")
        try:
            self.X = np.load(str(self.np_path / f"{self.split}_X_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"), allow_pickle=True)
            self.y = np.load(str(self.np_path / f"{self.split}_y_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"), allow_pickle=True)
            self.pids = np.load(str(self.np_path / f"{self.split}_pids_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"), allow_pickle=True)
            return
        except:
            print("Creating numpy arrays...")
            

        df = pd.read_csv(self.data_dir / f"{self.split}_rr_df.csv")
        if self.num_classes < 3:
            df = df[df["label"].isin(self.config.classes.keys())]
        #undersample for O and N seperately
        O_df = df.loc[(df["AA"] == "S") | (df["AA"] == "T")]
        N_df = df[df["AA"] == "N"]
        
        ros = RandomUnderSampler(random_state=self.seed)
        
        O_df, _ = ros.fit_resample(O_df, O_df["label"])
        N_df, _ = ros.fit_resample(N_df, N_df["label"])
        df = pd.concat([O_df, N_df])                   
        with h5py.File(self.embedding_path, 'r') as embeddings:
            y = df["label"].values
            pids = df["PID"].values
            positions = df["Position"].values
            AAs = df["AA"].values 
            classes = np.array(list(self.config.classes.keys()))

            # Preallocate memory for X if the size is known
            X = np.empty((len(pids), embeddings[list(embeddings.keys())[0]].shape[1]))

            # Perform string replacements once
            processed_pids = [pid.replace("-", "_").replace(".", "_") for pid in pids]

            for idx, (pid, pos) in enumerate(tqdm(zip(processed_pids, positions), total=len(processed_pids))):
                try:
                    embedding = embeddings[pid][()]
                    X[idx] = embedding[pos - 1]
                except KeyError:
                    print(f"Protein {pid} not found in embeddings!")
                    continue
                except IndexError:
                    print(f"Position {pos} not found in protein {pid}!")
                    continue

        self.X = X
 
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.pids = np.array(pids)
        self.AAs = np.array(AAs)
        if self.numpy:
            np.save(str(self.np_path / f"{self.split}_X_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"), self.X)
            np.save(str(self.np_path / f"{self.split}_y_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"), self.y)
            np.save(str(self.np_path / f"{self.split}_pids_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"), self.pids)
            np.save(str(self.np_path / f"{self.split}_AAs_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"), self.AAs)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.pids[index]
    
    

            