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
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
import plotly.express as px
from scripts.python.routines.plot.layout import add_layout
import plotly.graph_objects as go


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
            path: str = "",
            cpgs_fn: str = "",
            statuses_fn: str = "",
            dnam_fn: str = "betas.pkl",
            pheno_fn: str = "pheno.pkl",
            outcome: str = "Status",
            train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 1337,
            weighted_sampler = False,
            **kwargs,
    ):
        super().__init__()

        self.path = path
        self.cpgs_fn = cpgs_fn
        self.statuses_fn = statuses_fn
        self.dnam_fn = dnam_fn
        self.pheno_fn = pheno_fn
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
        self.betas = pd.read_pickle(f"{self.path}/{self.dnam_fn}")
        self.pheno = pd.read_pickle(f"{self.path}/{self.pheno_fn}")
        cpgs_df = pd.read_excel(self.cpgs_fn)
        self.cpgs = cpgs_df.loc[:, 'CpG'].values

        statuses_df = pd.read_excel(self.statuses_fn)
        self.statuses = {}
        for st_id, st in enumerate(statuses_df.loc[:, self.outcome].values):
            self.statuses[st] = st_id
        self.pheno = self.pheno.loc[self.pheno[self.outcome].isin(self.statuses)]
        self.pheno['Status_Origin'] = self.pheno[self.outcome]
        self.pheno[self.outcome].replace(self.statuses, inplace=True)

        self.betas = self.betas.loc[self.pheno.index.values, self.cpgs]
        if not list(self.pheno.index.values) == list(self.betas.index.values):
            log.info(f"Error! In pheno and betas subjects have different order")
            raise ValueError(f"Error! In pheno and betas subjects have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.betas.shape[1])

        self.dataset = BetasPhenoDataset(self.betas, self.pheno, self.outcome)

        self.ids_all = np.arange(len(self.dataset))

        assert abs(1.0 - sum(self.train_val_test_split)) < 1.0e-8, "Sum of train_val_test_split must be 1"

        if self.train_val_test_split[-1] < 1e-6:
            self.ids_train, self.ids_val = train_test_split(
                self.ids_all,
                test_size=self.train_val_test_split[1],  # self.train_val_test_split[-2]
                stratify=self.dataset.ys[self.ids_all],
                random_state=self.seed
            )
            self.ids_test = []
            dict_to_plot = {
                "all": self.ids_all,
                "train": self.ids_train,
                "val": self.ids_val,
            }
        else:
            self.ids_train_val, self.ids_test = train_test_split(
                self.ids_all,
                test_size=self.train_val_test_split[-1],
                stratify=self.dataset.ys[self.ids_all],
                random_state=self.seed
            )
            corrected_val_size = self.train_val_test_split[1] / (self.train_val_test_split[0] + self.train_val_test_split[1])
            self.ids_train, self.ids_val = train_test_split(
                self.ids_train_val,
                test_size=corrected_val_size, # self.train_val_test_split[-2]
                stratify=self.dataset.ys[self.ids_train_val],
                random_state=self.seed
            )
            dict_to_plot = {
                "all": self.ids_all,
                "train_val": self.ids_train_val,
                "train": self.ids_train,
                "val": self.ids_val,
                "test": self.ids_test
            }

        for name, ids in dict_to_plot.items():
            if not os.path.exists(f"{self.path}/figs"):
                os.makedirs(f"{self.path}/figs")

            status_counts = pd.DataFrame(Counter(self.pheno['Status_Origin'].values[ids]), index=[0])
            status_counts = status_counts.reindex(statuses_df.loc[:, self.outcome].values, axis=1)
            fig = go.Figure()
            for st, st_id in self.statuses.items():
                add_bar_trace(fig, x=[st], y=[status_counts.at[0, st]], text=[status_counts.at[0, st]], name=st)
            add_layout(fig, f"", f"Count", "")
            fig.update_layout({'colorway': px.colors.qualitative.Set1})
            fig.update_xaxes(showticklabels=False)
            save_figure(fig, f"bar_{name}")

        self.pheno.loc[self.pheno.index[self.ids_train], 'Part'] = "train"
        self.pheno.loc[self.pheno.index[self.ids_val], 'Part'] = "val"
        self.pheno.loc[self.pheno.index[self.ids_test], 'Part'] = "test"

        self.pheno.to_excel(f"pheno.xlsx", index=True)

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


class DNAmPhenoInferenceDataModule(LightningDataModule):

    def __init__(
            self,
            path: str = "",
            cpgs_fn: str = "",
            statuses_fn: str = "",
            dnam_fn: str = "betas.pkl",
            pheno_fn: str = "pheno.pkl",
            outcome: str = "Status",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.path = path
        self.cpgs_fn = cpgs_fn
        self.statuses_fn = statuses_fn
        self.dnam_fn = dnam_fn
        self.pheno_fn = pheno_fn
        self.outcome = outcome
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.betas = pd.read_pickle(f"{self.path}/{self.dnam_fn}")
        self.pheno = pd.read_pickle(f"{self.path}/{self.pheno_fn}")
        cpgs_df = pd.read_excel(self.cpgs_fn)
        self.cpgs = cpgs_df.loc[:, 'CpG'].values

        statuses_df = pd.read_excel(self.statuses_fn)
        self.statuses = {}
        for st_id, st in enumerate(statuses_df.loc[:, self.outcome].values):
            self.statuses[st] = st_id
        self.pheno = self.pheno.loc[self.pheno[self.outcome].isin(self.statuses)]
        self.pheno['Status_Origin'] = self.pheno[self.outcome]
        self.pheno[self.outcome].replace(self.statuses, inplace=True)

        self.betas = self.betas.loc[self.pheno.index.values, self.cpgs]
        if not list(self.pheno.index.values) == list(self.betas.index.values):
            log.info(f"Error! In pheno and betas subjects have different order")
            raise ValueError(f"Error! In pheno and betas subjects have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.betas.shape[1])

        self.dataset = BetasPhenoDataset(self.betas, self.pheno, self.outcome)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )