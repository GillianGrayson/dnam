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
            data: pd.DataFrame,
            output: pd.DataFrame,
            outcome: str
    ):
        self.data = data
        self.output = output
        self.outcome = outcome
        self.num_subjects = self.data.shape[0]
        self.num_features = self.data.shape[1]
        self.ys = self.output.loc[:, self.outcome].values

    def __getitem__(self, idx: int):
        x = self.data.iloc[idx, :].to_numpy()
        y = self.ys[idx]
        return (x, y, idx)

    def __len__(self):
        return self.num_subjects


class DNAmDataModuleNoTest(LightningDataModule):

    def __init__(
            self,
            path: str = "",
            task: str = "",
            features_fn: str = "",
            classes_fn: str = "",
            trn_val_fn: str = "",
            outcome: str = "",
            trn_val_split: Tuple[float, float] = (0.8, 0.2),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 1337,
            weighted_sampler = False,
            **kwargs,
    ):
        super().__init__()

        self.path = path
        self.task = task
        self.features_fn = features_fn
        self.classes_fn = classes_fn
        self.trn_val_fn = trn_val_fn
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
        self.trn_val = pd.read_excel(f"{self.path}/{self.trn_val_fn}", index_col="index")
        features_df = pd.read_excel(self.features_fn)
        self.features_names = features_df.loc[:, 'features'].values

        if self.task in ['binary', 'multiclass']:
            self.classes_df = pd.read_excel(self.classes_fn)
            self.classes_dict = {}
            for cl_id, cl in enumerate(self.classes_df.loc[:, self.outcome].values):
                self.classes_dict[cl] = cl_id

            self.trn_val = self.trn_val.loc[self.trn_val[self.outcome].isin(self.classes_dict)]
            self.trn_val[f'{self.outcome}_origin'] = self.trn_val[self.outcome]
            self.trn_val[self.outcome].replace(self.classes_dict, inplace=True)

        self.data = self.trn_val.loc[:, self.features_names]
        self.data = self.data.astype('float32')
        self.output = self.trn_val.loc[:, [self.outcome]]
        if self.task == 'regression':
            self.output = self.output.astype('float32')

        if not list(self.data.index.values) == list(self.output.index.values):
            log.info(f"Error! Indexes have different order")
            raise ValueError(f"Error! Indexes have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.data.shape[1])

        self.dataset = DNAmDataset(self.data, self.output, self.outcome)

        self.ids_trn_val = np.arange(self.trn_val.shape[0])

    def refresh_datasets(self):
        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)

    def perform_split(self):

        assert abs(1.0 - sum(self.trn_val_split)) < 1.0e-8, "Sum of trn_val_split must be 1"

        if self.task in ['binary', 'multiclass']:
            self.ids_trn, self.ids_val = train_test_split(
                self.ids_trn_val,
                test_size=self.trn_val_split[1],
                stratify=self.dataset.ys[self.ids_trn_val],
                random_state=self.seed
            )
        elif self.task == 'regression':
            ptp = np.ptp(self.dataset.ys[self.ids_trn_val])
            num_bins = 3
            bins = np.linspace(np.min(self.dataset.ys[self.ids_trn_val]) - 0.1 * ptp,
                               np.max(self.dataset.ys[self.ids_trn_val]) + 0.1 * ptp, num_bins + 1)
            binned = np.digitize(self.dataset.ys[self.ids_trn_val], bins) - 1
            unique, counts = np.unique(binned, return_counts=True)
            occ = dict(zip(unique, counts))
            self.ids_trn, self.ids_val = train_test_split(
                self.ids_trn_val,
                test_size=self.trn_val_split[1],
                stratify=binned,
                random_state=self.seed
            )

        self.ids_tst = None
        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)

    def plot_split(self, suffix=''):
        dict_to_plot = {
            "Train": self.ids_trn,
            "Val": self.ids_val
        }

        if not os.path.exists(f"{self.path}/figs"):
            os.makedirs(f"{self.path}/figs")
        if self.task in ['binary', 'multiclass']:
            for name, ids in dict_to_plot.items():
                classes_counts = pd.DataFrame(Counter(self.output[f'{self.outcome}_origin'].values[ids]), index=[0])
                classes_counts = classes_counts.reindex(self.classes_df.loc[:, self.outcome].values, axis=1)
                fig = go.Figure()
                for st, st_id in self.classes_dict.items():
                    add_bar_trace(fig, x=[st], y=[classes_counts.at[0, st]], text=[classes_counts.at[0, st]], name=st)
                add_layout(fig, f"", f"Count", "")
                fig.update_layout({'colorway': ["blue", "red", "green"]})
                fig.update_xaxes(showticklabels=False)
                save_figure(fig, f"bar_{name}{suffix}")

        elif self.task == 'regression':
            ptp = np.ptp(self.output[f'{self.outcome}'].values)
            bin_size = ptp / 15
            fig = go.Figure()
            for name, ids in dict_to_plot.items():
                fig.add_trace(
                    go.Histogram(
                        x=self.output[f'{self.outcome}'].values[ids],
                        name=name,
                        showlegend=True,
                        marker=dict(
                            opacity=0.7,
                            line=dict(
                                width=1
                            ),
                        ),
                        xbins=dict(size=bin_size)
                    )
                )
            add_layout(fig, f"{self.outcome}", "Count", "")
            fig.update_layout(margin=go.layout.Margin(l=90, r=20, b=75, t=50, pad=0))
            fig.update_layout(legend_font_size=20)
            fig.update_layout({'colorway': ["blue", "red", "green"]}, barmode='overlay')
            save_figure(fig, f"hist{suffix}")

        self.output.loc[self.output.index[self.ids_trn], 'Part'] = "trn"
        self.output.loc[self.output.index[self.ids_val], 'Part'] = "val"

        self.output.to_excel(f"output{suffix}.xlsx", index=True)

        log.info(f"total_count: {len(self.dataset)}")
        log.info(f"trn_count: {len(self.dataset_trn)}")
        log.info(f"val_count: {len(self.dataset_val)}")

    def get_trn_val_X_and_y(self):
        return Subset(self.dataset, self.ids_trn_val), self.dataset.ys[self.ids_trn_val]

    def get_weighted_sampler(self):
        return self.weighted_sampler

    def train_dataloader(self):
        ys_trn = self.dataset.ys[self.ids_trn]
        if self.task in ['binary', 'multiclass']:
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
        return None

    def get_feature_names(self):
        return self.data.columns.to_list()

    def get_outcome_name(self):
        return self.outcome

    def get_class_names(self):
        return list(self.classes_dict.keys())

    def get_df(self):
        df = pd.merge(self.output.loc[:, self.outcome], self.data, left_index=True, right_index=True)
        return df


class DNAmDataModuleTogether(LightningDataModule):

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

    def get_feature_names(self):
        return self.dnam.columns.to_list()

    def get_class_names(self):
        return list(self.statuses.keys())

    def get_raw_data(self):
        data = pd.merge(self.pheno.loc[:, self.outcome], self.dnam, left_index=True, right_index=True)
        train_data = data.iloc[self.ids_trn]
        val_data = data.iloc[self.ids_val]
        test_data = data.iloc[self.ids_tst]
        raw_data = {}
        raw_data['X_train'] = train_data.loc[:, self.dnam.columns.values].values
        raw_data['y_train'] = train_data.loc[:, self.outcome].values
        raw_data['X_val'] = val_data.loc[:, self.dnam.columns.values].values
        raw_data['y_val'] = val_data.loc[:, self.outcome].values
        raw_data['X_test'] = test_data.loc[:, self.dnam.columns.values].values
        raw_data['y_test'] = test_data.loc[:, self.outcome].values
        return raw_data


class DNAmDataModuleSeparate(LightningDataModule):

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

    def get_feature_names(self):
        return self.dnam.columns.to_list()

    def get_class_names(self):
        return list(self.statuses.keys())

    def get_raw_data(self):
        data = pd.merge(self.pheno.loc[:, self.outcome], self.dnam, left_index=True, right_index=True)
        train_data = data.iloc[self.ids_trn]
        val_data = data.iloc[self.ids_val]
        test_data = data.iloc[self.ids_tst]
        raw_data = {}
        raw_data['X_train'] = train_data.loc[:, self.dnam.columns.values].values
        raw_data['y_train'] = train_data.loc[:, self.outcome].values
        raw_data['X_val'] = val_data.loc[:, self.dnam.columns.values].values
        raw_data['y_val'] = val_data.loc[:, self.outcome].values
        raw_data['X_test'] = test_data.loc[:, self.dnam.columns.values].values
        raw_data['y_test'] = test_data.loc[:, self.outcome].values
        return raw_data


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

    def get_feature_names(self):
        return self.dnam.columns.to_list()

    def get_class_names(self):
        return list(self.statuses.keys())

    def get_raw_data(self):
        data = pd.merge(self.pheno.loc[:, self.outcome], self.dnam, left_index=True, right_index=True)
        test_data = data.iloc[self.ids]
        raw_data = {}
        raw_data['X_test'] = test_data.loc[:, self.dnam.columns.values].values
        raw_data['y_test'] = test_data.loc[:, self.outcome].values
        return raw_data