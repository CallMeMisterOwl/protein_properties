import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    random_split,
)
from tqdm import tqdm

from src.utils import read_vespag


@dataclass
class GlycoDataConfig:
    classes: dict
    data_dir: str = "data/glyco/"
    embedding_path: str = "data/glyco/glyco_embeddings.h5"
    esm_embedding_path: str = "data/glyco/glyco_esm_embeddings.h5"
    np_path: str = "data/glyco/np"
    num_workers: int = 4
    batch_size: int = 64
    add_esm: bool = True
    add_vespag: bool = True
    neg_sample_ratio: float = 1.3
    pos_sample_ratio: float = 1.0
    add_neg_op_sites: bool = False
    model_type: Literal["FNN", "CNN"] = "FNN"


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
    def __init__(
        self,
        protein_ids,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
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
                    continue

            batch.extend(indices)
            if len(batch) >= self.batch_size:
                self.batches.append(batch)

        np.random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)



class ImpGroupedBatchSampler(BatchSampler):
    def __init__(
        self,
        protein_ids,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        bin_size: int = 10  # New parameter to control binning
    ):
        self.protein_ids = protein_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.bin_size = bin_size
        self.batches = []
        self._create_batches()

    def _create_batches(self):
        # Get unique protein IDs and their counts
        protein_ids, counts = np.unique(self.protein_ids, return_counts=True)
        
        # Create bins based on count ranges
        bin_indices = np.digitize(counts, np.arange(0, counts.max() + self.bin_size, self.bin_size))
        
        # Group protein IDs by bin
        bin_groups = defaultdict(list)
        for bin_idx, pid, count in zip(bin_indices, protein_ids, counts):
            bin_groups[bin_idx].append((pid, count))
        
        # Shuffle bins if required
        if self.shuffle:
            for bin_id in bin_groups:
                np.random.shuffle(bin_groups[bin_id])

        # Create batches from bin groups
        current_batch = []
        for bin_id, proteins in bin_groups.items():
            for pid, count in proteins:
                indices = np.where(self.protein_ids == pid)[0].tolist()
                if self.shuffle:
                    np.random.shuffle(indices)

                for idx in indices:
                    current_batch.append(idx)
                    if len(current_batch) == self.batch_size:
                        self.batches.append(current_batch)
                        current_batch = []

        # Handle last batch if drop_last is False
        if current_batch and not self.drop_last:
            self.batches.append(current_batch)
        
        # Shuffle batches if required
        if self.shuffle:
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
            raise FileNotFoundError(
                "The data directory that was provided does not exist! Please download the data first!"
            )

        """if not (self.data_dir / "train.csv").exists() or not (self.data_dir / "val.csv").exists() or not (self.data_dir / "O_test.csv").exists():
            raise FileNotFoundError("The data directory that was provided is missing the .o files! Please download the data first!")"""

    def setup(self, stage=None):
        if stage == 'test':
            return
        self.prepare_data()

         
        train_dataset = GlycoDataset("train", self.config, 42, True)
        val_dataset = GlycoDataset("val", self.config, 42, True)
        #og_train_dataset = GlycoDataset("train", self.config, 42, False)
        # split the train dataset into train and val
        '''labels = []
        for l, a in zip(og_train_dataset.y, og_train_dataset.AAs):
            labels.append(l if a == "N" and l == 0 else 3)
        train_idx, val_idx = next(
            GroupShuffleSplit(train_size=0.9, random_state=42, n_splits=2).split(
                og_train_dataset.X, labels, og_train_dataset.pids
            )
        )
        self.train_dataset = Subset(og_train_dataset, train_idx)
        self.val_dataset = Subset(og_train_dataset, val_idx)'''

        # self.val_dataset = GlycoDataset("val", self.config)
        """self.O_test_dataset = GlycoDataset("O_test", self.config, 370381)
        self.N_test_dataset = GlycoDataset("N_test", self.config)
        self.test_dataset = ConcatDataset([self.O_test_dataset, self.N_test_dataset])"""

        if len(self.config.classes) < 3:
            train_dataset.y = np.array(
                [
                    arr.astype(np.float16) if arr.dtype != np.float16 else arr
                    for arr in train_dataset.y
                ]
            )
            val_dataset.y = np.array(
                [
                    arr.astype(np.float16) if arr.dtype != np.float16 else arr
                    for arr in val_dataset.y
                ]
            )


        # calculate class weights for the loss function
        # ys = np.apply_along_axis(np.concatenate, 0, og_train_dataset.y)
        counts = np.unique(train_dataset.y, return_counts=True)[1]
        # this will be used for cross validation 
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # check if class weights are already calculated
        if len(self.config.classes) < 3:
            # For binary predictions set only positive weight
            self.class_weights = torch.tensor(
                [counts[0] / counts[1]], dtype=torch.float16
            )
        else:
            self.class_weights = torch.tensor(
                [max(counts) / counts[i] for i in range(len(self.config.classes))],
                dtype=torch.float32,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.config.num_workers,
            batch_sampler=ImpGroupedBatchSampler(
                self.train_dataset.pids,
                self.config.batch_size,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

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
    def __init__(
        self,
        split: str,
        config: GlycoDataConfig,
        seed: int = None,
        generate_numpy: bool = True,
    ) -> None:
        super().__init__()
        self.split = split
        self.data_dir = Path(config.data_dir)
        self.embedding_path = config.embedding_path
        self.esm_embedding_path = config.esm_embedding_path
        self.np_path = Path(config.np_path)
        self.num_classes = len(config.classes)
        self.to_classes = np.vectorize(config.classes.get)
        self.add_esm = config.add_esm
        self.add_vespag = config.add_vespag
        self.neg_sample_ratio = config.neg_sample_ratio
        self.pos_sample_ratio = config.pos_sample_ratio
        self.model_type = config.model_type
        self.config = config
        '''self.add_neg_sites = config.add_neg_sites
        if self.add_neg_sites and self.num_classes < 3 and self.split == "train":
            self.split = "train_more_neg"
        self.use_neg_diff = config.use_neg_diff if self.split == "train" else False'''

        self.seed = seed if seed else random.randint(0, 100000)
        self.numpy = generate_numpy

        self.X = None
        self.y = None
        self.pids = None
        self.load_data()

    
    def create_embeddings(self, pids, input_feat_dim, positions):
        if self.model_type == 'CNN':
            X = np.zeros((len(pids), 31, input_feat_dim))
            embedding_dims = (31, input_feat_dim)
            half_window = 15

        else:
            X = np.zeros((len(pids), input_feat_dim), dtype=np.float32)
            embedding_dims = input_feat_dim

        # Perform string replacements once
        processed_pids = [pid.replace("-", "_").replace(".", "_") for pid in pids]

        for idx, (pid, pos) in enumerate(tqdm(zip(processed_pids, positions), total=len(processed_pids))):
            input_feature = np.zeros(embedding_dims)
            with h5py.File(self.embedding_path, "r") as embeddings:
                try:
                    if self.model_type == 'CNN':
                        emb = embeddings[pid][()]
                        start = max(0, pos - half_window - 1)
                        end = min(emb.shape[0], pos + half_window)
                        input_feature[:, 1024] = emb[start:end]
                    else:
                        input_feature[:1024] = embeddings[pid][()][pos - 1]
                except KeyError:
                    print(f"Protein {pid} not found in ProtT5 embeddings!")
                    continue
                except IndexError:
                    print(f"Position {pos} not found in protein {pid}!")
                    continue
                
            if self.add_esm:
                with h5py.File(self.esm_embedding_path, "r") as esm_embeddings:
                    input_pos = 1024 if not self.add_vespag else 1024 + 20
                    try:
                        if self.model_type == 'CNN':
                            emb = esm_embeddings[pid][()]
                            start = max(0, pos - half_window - 1)
                            end = min(emb.shape[0], pos + half_window)
                            input_feature[:, input_pos] = emb[start:end]
                        else:
                            input_feature[input_pos:] = esm_embeddings[pid.replace("_", "-")][()][pos - 1]
                        
                    except KeyError:
                        print(f"Protein {pid} not found in ESM embeddings!")
                        continue
                    except IndexError:
                        print(f"Position {pos} not found in protein {pid}!")
                        continue
            X[idx] = input_feature
            
        return X
    
    def load_data(self):
        if self.split == "blind_test":
            raise NotImplementedError("Blind test set not implemented yet!")
        
        try:
            self.X = np.load(
                str(
                    self.np_path
                    / f"{self.split}_X_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"
                ),
                allow_pickle=True,
            )
            self.y = np.load(
                str(
                    self.np_path
                    / f"{self.split}_y_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"
                ),
                allow_pickle=True,
            )
            self.pids = np.load(
                str(
                    self.np_path
                    / f"{self.split}_pids_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"
                ),
                allow_pickle=True,
            )
            return
        except:
            print("Creating numpy arrays...")

        ros = RandomUnderSampler(random_state=self.seed)
        prefix = 'N' if 'N' in self.config.classes else 'O'
        prefix = prefix if self.num_classes < 3 else ''
        df = pd.read_csv(self.data_dir /f'{prefix}'/ f"{prefix}_{self.split}_RR.csv" if prefix != '' else self.data_dir / 'combined' /f"{self.split}_RR.csv")
        if self.num_classes < 3 and self.split == "train":
            #df = df[df["label"].isin(self.config.classes.values())]
            #df["label"] = df["label"].apply(lambda x: 1 if x >= 1 else 0)
            if not self.config.add_neg_op_sites:
                #df = df[df["AA"] == "N"] if 'N' in self.config.classes else df[(df["AA"] == "S") | (df["AA"] == "T")]
                sampling_strat = {
                    0: int(df["label"].value_counts()[1] * self.neg_sample_ratio),
                    1: int(df["label"].value_counts()[1]),
                } 
                over_sampling_strat = {
                    0: int(df["label"].value_counts()[1] * self.neg_sample_ratio),
                    1: int(df["label"].value_counts()[1] * self.pos_sample_ratio),
                } 
                labels = df["label"].values 
                
            ros = RandomUnderSampler(random_state=self.seed, 
                                    sampling_strategy=sampling_strat)
            over_samp = RandomOverSampler(random_state=self.seed, 
                                    sampling_strategy=over_sampling_strat)
            df, labels = ros.fit_resample(df, labels)
            df, _ = over_samp.fit_resample(df, labels)
        # undersample for O and N seperately
        elif self.split == "train":
            O_df = df.loc[(df["AA"] == "S") | (df["AA"] == "T")]
            N_df = df[df["AA"] == "N"]
            O_df, _ = ros.fit_resample(O_df, O_df["label"])
            N_df, _ = ros.fit_resample(N_df, N_df["label"])
            df = pd.concat([O_df, N_df])

            
        y = df["label"].values
        pids = df["PID"].values
        positions = df["Position"].values
        AAs = df["AA"].values
        classes = np.array(list(self.config.classes.keys()))
        input_feat_dim = 1024 
        if self.add_vespag:
            input_feat_dim += 20
        if self.add_esm:
            input_feat_dim += 1280
        
        # Preallocate memory for X if the size is known
        
        X = np.empty((len(pids), input_feat_dim), dtype=np.float32)

        # Perform string replacements once
        processed_pids = [pid.replace("-", "_").replace(".", "_") for pid in pids]

        for idx, (pid, pos) in enumerate(tqdm(zip(processed_pids, positions), total=len(processed_pids))):
            input_feature = np.empty(input_feat_dim)
            with h5py.File(self.embedding_path, "r") as embeddings:
                try:
                    input_feature[:1024] = embeddings[pid][()][pos - 1]
                except KeyError:
                    print(f"Protein {pid} not found in ProtT5 embeddings!")
                    continue
                except IndexError:
                    print(f"Position {pos} not found in protein {pid}!")
                    continue
            if self.add_vespag:
                vespag = read_vespag(pid, self.data_dir / "vespag")
                if vespag is not None:
                    vespag = vespag[vespag["Position"] == pos]
                    if not vespag.empty:
                        vespag = vespag["VespaG"].values[0]
                        input_feature[1024:1044] = vespag
                else:
                    # add zeros if vespag is not found
                    input_feature[1024:1044] = np.zeros(20)
            if self.add_esm:
                with h5py.File(self.esm_embedding_path, "r") as esm_embeddings:
                    input_pos = 1024 if not self.add_vespag else 1024 + 20
                    try:
                        input_feature[input_pos:] = esm_embeddings[pid.replace("_", "-")][()][pos - 1]
                    except KeyError:
                        print(f"Protein {pid} not found in ESM embeddings!")
                        continue
                    except IndexError:
                        print(f"Position {pos} not found in protein {pid}!")
                        continue
            X[idx] = input_feature
                                            
        # remove samples with missing embeddings
        mask = np.all(X != 0, axis=1)
        X = X[mask]
        y = y[mask]
        pids = pids[mask]
        AAs = AAs[mask]

        print(f"Number of samples removed: {np.sum(~mask)}")

        self.X = X
        self.X = np.array(self.X)
        self.y = np.array(y)
        self.pids = np.array(pids)
        self.AAs = np.array(AAs)
        
        if self.numpy:
            np.save(str(self.np_path / f"{self.split}_X_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"),
                    self.X
            )
            np.save(str(self.np_path / f"{self.split}_y_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"),
                    self.y
            )
            np.save(str(self.np_path / f"{self.split}_pids_c{self.num_classes}_{'_'.join(self.config.classes.keys())}.npy"),
                    self.pids
            )
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.pids[index]

