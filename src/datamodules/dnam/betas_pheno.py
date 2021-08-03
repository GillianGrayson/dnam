import torch
from typing import Optional, Tuple
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
import numpy as np
import pandas as pd
from collections import Counter
from src.utils import utils
import matplotlib.pyplot as plt
import os


log = utils.get_logger(__name__)

class BetasPhenoDataset(Dataset):

    def __init__(
            self,
            betas: pd.DataFrame,
            pheno: pd.DataFrame,
            outcome: str = 'Status'
    ):
        self.betas = betas
        self.pheno = pheno
        self.outcome = outcome
        self.num_subjects = self.betas.shape[0]
        self.num_features = self.betas.shape[1]
        self.ys = self.pheno.loc[:, self.outcome].values

    def __getitem__(self, idx: int):
        x = self.betas.iloc[idx, :].to_numpy()
        y = self.ys[idx]
        return (x, y, idx)

    def __len__(self):
        return self.num_subjects


class BetasPhenoDataModule(LightningDataModule):

    def __init__(
            self,
            path: str = "E:/YandexDisk/Work/pydnameth/datasets/GPL13534/meta/BrainDiseases/variance(0.005)",
            outcome: str = "Status",
            train_val_test_split: Tuple[float, float, float] = (0.8, 0.2, 0.2),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 1337,
            weighted_sampler = False,
            **kwargs,
    ):
        super().__init__()

        self.path = path
        self.outcome = outcome
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.weighted_sampler = weighted_sampler

        self.dataset_train: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.betas = pd.read_pickle(f"{self.path}/betas.pkl")
        self.pheno = pd.read_pickle(f"{self.path}/pheno.pkl")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.betas.shape[1])

        self.dataset = BetasPhenoDataset(self.betas, self.pheno, self.outcome)

        self.ids_all = np.arange(len(self.dataset))

        self.ids_train_val, self.ids_test = train_test_split(
            self.ids_all,
            test_size=self.train_val_test_split[-1],
            stratify=self.dataset.ys[self.ids_all],
            random_state=self.seed
        )

        self.ids_train, self.ids_val = train_test_split(
            self.ids_train_val,
            test_size=self.train_val_test_split[-2],
            stratify=self.dataset.ys[self.ids_train_val],
            random_state=self.seed
        )

        for name, ids in {"train_val": self.ids_train_val,
                         "train": self.ids_train,
                         "val": self.ids_val,
                         "test": self.ids_test}.items():
            if not os.path.exists(f"{self.path}/figs"):
                os.makedirs(f"{self.path}/figs")
            status_counts = pd.DataFrame(Counter(self.dataset.ys[ids]), index=[0])
            status_counts = status_counts.reindex(sorted(status_counts.columns), axis=1)
            plot = status_counts.plot.bar()
            plt.xlabel("Status", fontsize=15)
            plt.ylabel("Count", fontsize=15)
            plt.xticks([])
            plt.axis('auto')
            fig = plot.get_figure()
            fig.savefig(f"{self.path}/figs/bar_{name}.pdf")
            fig.savefig(f"{self.path}/figs/bar_{name}.png")
            plt.close()

        self.dataset_train = Subset(self.dataset, self.ids_train)
        self.dataset_val = Subset(self.dataset, self.ids_val)
        self.dataset_test = Subset(self.dataset, self.ids_test)

        log.info(f"total_count: {len(self.dataset)}")
        log.info(f"train_count: {len(self.dataset_train)}")
        log.info(f"val_count: {len(self.dataset_val)}")
        log.info(f"test_count: {len(self.dataset_test)}")

    def get_train_val_dataset_and_labels(self):
        return Subset(self.dataset, self.ids_train_val), self.dataset.ys[self.ids_train_val]

    def get_weighted_sampler(self):
        return self.weighted_sampler

    def train_dataloader(self):
        ys_train = self.dataset.ys[self.ids_train]
        class_counter = Counter(ys_train)
        class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
        weights = torch.FloatTensor([class_weights[y] for y in ys_train])
        if self.weighted_sampler:
            weighted_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
            return DataLoader(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=weighted_sampler
            )
        else:
            return DataLoader(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
