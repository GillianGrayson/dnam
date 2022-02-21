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

class DNAmDataset(Dataset):

    def __init__(
            self,
            dnam: pd.DataFrame,
            pheno: pd.DataFrame,
            outcome: str = 'Status'
    ):
        self.dnam = dnam
        self.pheno = pheno
        self.outcome = outcome
        self.num_subjects = self.dnam.shape[0]
        self.num_features = self.dnam.shape[1]
        self.ys = self.pheno.loc[:, self.outcome].values

    def __getitem__(self, idx: int):
        x = self.dnam.iloc[idx, :].to_numpy()
        y = self.ys[idx]
        return (x, y, idx)

    def __len__(self):
        return self.num_subjects


class DNAmSingleDataModule(LightningDataModule):

    def __init__(
            self,
            path: str = "",
            cpgs_fn: str = "",
            statuses_fn: str = "",
            dnam_fn: str = "",
            pheno_fn: str = "",
            outcome: str = "Status",
            trn_val_tst_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
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
        self.trn_val_tst_split = trn_val_tst_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.weighted_sampler = weighted_sampler

        self.dataset_trn: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_tst: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        self.dnam = pd.read_pickle(f"{self.path}/{self.dnam_fn}")
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

        self.dnam = self.dnam.loc[self.pheno.index.values, self.cpgs]
        if not list(self.pheno.index.values) == list(self.dnam.index.values):
            log.info(f"Error! In pheno and betas subjects have different order")
            raise ValueError(f"Error! In pheno and betas subjects have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.dnam.shape[1])

        self.dataset = DNAmDataset(self.dnam, self.pheno, self.outcome)

        self.ids_all = np.arange(len(self.dataset))

        assert abs(1.0 - sum(self.trn_val_tst_split)) < 1.0e-8, "Sum of trn_val_tst_split must be 1"

        if self.trn_val_tst_split[-1] < 1e-6:
            self.ids_trn, self.ids_val = train_test_split(
                self.ids_all,
                test_size=self.trn_val_tst_split[1],
                stratify=self.dataset.ys[self.ids_all],
                random_state=self.seed
            )
            self.ids_tst = []
            dict_to_plot = {
                "all": self.ids_all,
                "trn": self.ids_trn,
                "val": self.ids_val,
            }
        else:
            self.ids_trn_val, self.ids_tst = train_test_split(
                self.ids_all,
                test_size=self.trn_val_tst_split[-1],
                stratify=self.dataset.ys[self.ids_all],
                random_state=self.seed
            )
            corrected_val_size = self.trn_val_tst_split[1] / (self.trn_val_tst_split[0] + self.trn_val_tst_split[1])
            self.ids_trn, self.ids_val = train_test_split(
                self.ids_trn_val,
                test_size=corrected_val_size,
                stratify=self.dataset.ys[self.ids_trn_val],
                random_state=self.seed
            )
            dict_to_plot = {
                "all": self.ids_all,
                "trn_val": self.ids_trn_val,
                "trn": self.ids_trn,
                "val": self.ids_val,
                "tst": self.ids_tst
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

        self.pheno.loc[self.pheno.index[self.ids_trn], 'Part'] = "trn"
        self.pheno.loc[self.pheno.index[self.ids_val], 'Part'] = "val"
        self.pheno.loc[self.pheno.index[self.ids_tst], 'Part'] = "tst"

        self.pheno.to_excel(f"pheno.xlsx", index=True)

        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)
        self.dataset_tst = Subset(self.dataset, self.ids_tst)

        log.info(f"total_count: {len(self.dataset)}")
        log.info(f"trn_count: {len(self.dataset_trn)}")
        log.info(f"val_count: {len(self.dataset_val)}")
        log.info(f"tst_count: {len(self.dataset_tst)}")

    def get_trn_val_dataset_and_labels(self):
        return Subset(self.dataset, self.ids_trn_val), self.dataset.ys[self.ids_trn_val]

    def get_weighted_sampler(self):
        return self.weighted_sampler

    def train_dataloader(self):
        ys_trn = self.dataset.ys[self.ids_trn]
        class_counter = Counter(ys_trn)
        class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
        weights = torch.FloatTensor([class_weights[y] for y in ys_trn])
        if self.weighted_sampler:
            weighted_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
            return DataLoader(
                dataset=self.dataset_trn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=weighted_sampler
            )
        else:
            return DataLoader(
                dataset=self.dataset_trn,
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
            dataset=self.dataset_tst,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class DNAmDoubleDataModule(LightningDataModule):

    def __init__(
            self,
            path: str = "",
            cpgs_fn: str = "",
            statuses_fn: str = "",
            dnam_trn_val_fn: str = "",
            pheno_trn_val_fn: str = "",
            dnam_tst_fn: str = "",
            pheno_tst_fn: str = "",
            outcome: str = "Status",
            trn_val_split: Tuple[float, float, float] = (0.8, 0.2),
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
        self.dnam_trn_val_fn = dnam_trn_val_fn
        self.pheno_trn_val_fn = pheno_trn_val_fn
        self.dnam_tst_fn = dnam_tst_fn
        self.pheno_tst_fn = pheno_tst_fn
        self.outcome = outcome
        self.trn_val_split = trn_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.weighted_sampler = weighted_sampler

        self.dataset_trn: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_tst: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        self.dnam_trn_val = pd.read_pickle(f"{self.path}/{self.dnam_trn_val_fn}")
        self.pheno_trn_val = pd.read_pickle(f"{self.path}/{self.pheno_trn_val_fn}")
        self.dnam_tst = pd.read_pickle(f"{self.path}/{self.dnam_tst_fn}")
        self.pheno_tst = pd.read_pickle(f"{self.path}/{self.pheno_tst_fn}")
        cpgs_df = pd.read_excel(self.cpgs_fn)
        self.cpgs = cpgs_df.loc[:, 'CpG'].values

        statuses_df = pd.read_excel(self.statuses_fn)
        self.statuses = {}
        for st_id, st in enumerate(statuses_df.loc[:, self.outcome].values):
            self.statuses[st] = st_id
        self.pheno_trn_val = self.pheno_trn_val.loc[self.pheno_trn_val[self.outcome].isin(self.statuses)]
        self.pheno_trn_val['Status_Origin'] = self.pheno_trn_val[self.outcome]
        self.pheno_trn_val[self.outcome].replace(self.statuses, inplace=True)
        self.pheno_tst = self.pheno_tst.loc[self.pheno_tst[self.outcome].isin(self.statuses)]
        self.pheno_tst['Status_Origin'] = self.pheno_tst[self.outcome]
        self.pheno_tst[self.outcome].replace(self.statuses, inplace=True)

        self.dnam_trn_val = self.dnam_trn_val.loc[self.pheno_trn_val.index.values, self.cpgs]
        if not list(self.pheno_trn_val.index.values) == list(self.dnam_trn_val.index.values):
            log.info(f"Error! In pheno and betas subjects have different order")
            raise ValueError(f"Error! In pheno and betas subjects have different order")

        self.dnam_tst = self.dnam_tst.loc[self.pheno_tst.index.values, self.cpgs]
        if not list(self.pheno_tst.index.values) == list(self.dnam_tst.index.values):
            log.info(f"Error! In pheno and betas subjects have different order")
            raise ValueError(f"Error! In pheno and betas subjects have different order")

        self.dnam = pd.concat([self.dnam_trn_val, self.dnam_tst])
        self.pheno = pd.concat([self.pheno_trn_val, self.pheno_tst])
        if not list(self.pheno.index.values) == list(self.dnam.index.values):
            log.info(f"Error! In pheno and betas subjects have different order")
            raise ValueError(f"Error! In pheno and betas subjects have different order")

        self.ids_trn_val = np.arange(self.pheno_trn_val.shape[0])
        self.ids_tst = np.arange(self.pheno_tst.shape[0]) + self.pheno_trn_val.shape[0]

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.dnam.shape[1])

        self.dataset = DNAmDataset(self.dnam, self.pheno, self.outcome)

        assert abs(1.0 - sum(self.trn_val_split)) < 1.0e-8, "Sum of trn_val_split must be 1"

        self.ids_trn, self.ids_val = train_test_split(
            self.ids_trn_val,
            test_size=self.trn_val_split[1],
            stratify=self.dataset.ys[self.ids_trn_val],
            random_state=self.seed
        )
        dict_to_plot = {
            "trn": self.ids_trn,
            "val": self.ids_val,
            "tsr": self.ids_tst
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

        self.pheno.loc[self.pheno.index[self.ids_trn], 'Part'] = "trn"
        self.pheno.loc[self.pheno.index[self.ids_val], 'Part'] = "val"
        self.pheno.loc[self.pheno.index[self.ids_tst], 'Part'] = "tst"

        self.pheno.to_excel(f"pheno.xlsx", index=True)

        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)
        self.dataset_tst = Subset(self.dataset, self.ids_tst)

        log.info(f"total_count: {len(self.dataset)}")
        log.info(f"trn_count: {len(self.dataset_trn)}")
        log.info(f"val_count: {len(self.dataset_val)}")
        log.info(f"tst_count: {len(self.dataset_tst)}")

    def get_trn_val_dataset_and_labels(self):
        return Subset(self.dataset, self.ids_trn_val), self.dataset.ys[self.ids_trn_val]

    def get_weighted_sampler(self):
        return self.weighted_sampler

    def train_dataloader(self):
        ys_trn = self.dataset.ys[self.ids_trn]
        class_counter = Counter(ys_trn)
        class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
        weights = torch.FloatTensor([class_weights[y] for y in ys_trn])
        if self.weighted_sampler:
            weighted_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
            return DataLoader(
                dataset=self.dataset_trn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=weighted_sampler
            )
        else:
            return DataLoader(
                dataset=self.dataset_trn,
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
            dataset=self.dataset_tst,
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
            dnam_fn: str = "",
            pheno_fn: str = "",
            outcome: str = "Status",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            imputation: str = "median",
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
        self.imputation = imputation

        self.dataset: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        self.dnam = pd.read_pickle(f"{self.path}/{self.dnam_fn}")
        self.pheno = pd.read_pickle(f"{self.path}/{self.pheno_fn}")
        cpgs_df = pd.read_excel(self.cpgs_fn)
        self.cpgs = cpgs_df.loc[:, 'CpG'].values
        cpgs_df.set_index('CpG', inplace=True)

        statuses_df = pd.read_excel(self.statuses_fn)
        self.statuses = {}
        for st_id, st in enumerate(statuses_df.loc[:, self.outcome].values):
            self.statuses[st] = st_id
        self.pheno = self.pheno.loc[self.pheno[self.outcome].isin(self.statuses)]
        self.pheno['Status_Origin'] = self.pheno[self.outcome]
        self.pheno[self.outcome].replace(self.statuses, inplace=True)

        missed_cpgs = list(set(self.cpgs) - set(self.dnam.columns.values))
        if len(missed_cpgs) > 0:
            log.info(f"Perform imputation for {len(missed_cpgs)} CpGs with {self.imputation}")
            if self.imputation in ["mean", "median"]:
                for cpg in  missed_cpgs:
                    self.dnam.loc[:, cpg] = cpgs_df.at[cpg, self.imputation]
            else:
                raise ValueError(f"Unsupported imputation: {self.imputation}")

        self.dnam = self.dnam.astype('float32')

        self.dnam = self.dnam.loc[self.pheno.index.values, self.cpgs]
        if not list(self.pheno.index.values) == list(self.dnam.index.values):
            log.info(f"Error! In pheno and betas subjects have different order")
            raise ValueError(f"Error! In pheno and betas subjects have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.dnam.shape[1])
        self.ids = np.arange(self.pheno.shape[0])

        self.dataset = DNAmDataset(self.dnam, self.pheno, self.outcome)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )