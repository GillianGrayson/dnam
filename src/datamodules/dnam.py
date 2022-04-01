import torch
from typing import Optional, Tuple
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
import numpy as np
import pandas as pd
from collections import Counter
from src.utils import utils
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
import plotly.express as px
from scripts.python.routines.plot.layout import add_layout
import plotly.graph_objects as go
from tqdm import tqdm
from impyute.imputation.cs import fast_knn, mean, median, random, mice, mode, em


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
        self.trn_val = pd.read_pickle(f"{self.trn_val_fn}")
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
        if self.task == 'regression':
            self.output = self.trn_val.loc[:, [self.outcome]]
            self.output = self.output.astype('float32')
        elif self.task in ['binary', 'multiclass']:
            self.output = self.trn_val.loc[:, [self.outcome, f'{self.outcome}_origin']]

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

        log.info(f"total_count: {len(self.dataset)}")
        log.info(f"trn_count: {len(self.dataset_trn)}")
        log.info(f"val_count: {len(self.dataset_val)}")

    def plot_split(self, suffix=''):
        dict_to_plot = {
            "Train": self.ids_trn,
            "Val": self.ids_val
        }
        if self.task in ['binary', 'multiclass']:
            for name, ids in dict_to_plot.items():
                classes_counts = pd.DataFrame(Counter(self.output[f'{self.outcome}_origin'].values[ids]), index=[0])
                classes_counts = classes_counts.reindex(self.classes_df.loc[:, self.outcome].values, axis=1)
                fig = go.Figure()
                for st, st_id in self.classes_dict.items():
                    add_bar_trace(fig, x=[st], y=[classes_counts.at[0, st]], text=[classes_counts.at[0, st]], name=st)
                add_layout(fig, f"", f"Count", "")
                fig.update_layout({'colorway': px.colors.qualitative.Set1})
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
            fig.update_layout({'colorway': px.colors.qualitative.Set1}, barmode='overlay')
            save_figure(fig, f"hist{suffix}")

        self.output.loc[self.output.index[self.ids_trn], 'Part'] = "trn"
        self.output.loc[self.output.index[self.ids_val], 'Part'] = "val"

        self.output.to_excel(f"output{suffix}.xlsx", index=True)

    def get_trn_val_y(self):
        return self.dataset.ys[self.ids_trn_val]

    def train_dataloader(self):
        ys_trn = self.dataset.ys[self.ids_trn]
        if self.task in ['binary', 'multiclass'] and self.weighted_sampler:
            class_counter = Counter(ys_trn)
            class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
            weights = torch.FloatTensor([class_weights[y] for y in ys_trn])
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


class DNAmDataModuleSeparate(LightningDataModule):

    def __init__(
            self,
            task: str = "",
            features_fn: str = "",
            classes_fn: str = "",
            trn_val_fn: str = "",
            tst_fn: str = "",
            outcome: str = "",
            trn_val_split: Tuple[float, float] = (0.8, 0.2),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 1337,
            weighted_sampler = False,
            imputation: str = "median",
            **kwargs,
    ):
        super().__init__()

        self.task = task
        self.features_fn = features_fn
        self.classes_fn = classes_fn
        self.trn_val_fn = trn_val_fn
        self.tst_fn = tst_fn
        self.outcome = outcome
        self.trn_val_split = trn_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.weighted_sampler = weighted_sampler
        self.imputation = imputation

        self.dataset_trn: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_tst: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        self.trn_val = pd.read_pickle(f"{self.trn_val_fn}")
        self.tst = pd.read_pickle(f"{self.tst_fn}")
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

            self.tst = self.tst.loc[self.tst[self.outcome].isin(self.classes_dict)]
            self.tst[f'{self.outcome}_origin'] = self.tst[self.outcome]
            self.tst[self.outcome].replace(self.classes_dict, inplace=True)

        missed_features = list(set(self.features_names) - set(self.tst.columns.values))
        exist_features = list(set(self.features_names) - set(missed_features))
        if len(missed_features) > 0:
            log.info(f"Perform imputation for {len(missed_features)} features with {self.imputation}")
            if self.imputation == "median":
                for f_id, f in enumerate(tqdm(missed_features, desc=f"{self.imputation} calculation")):
                    self.tst.loc[:, f] = self.trn_val[f].median()
            elif self.imputation == "mean":
                for f_id, f in enumerate(tqdm(missed_features, desc=f"{self.imputation} calculation")):
                    self.tst.loc[:, f] = self.trn_val[f].mean()
            else:
                raise ValueError(f"Unsupported imputation: {self.imputation}")

        self.ids_trn_val = np.arange(self.trn_val.shape[0])
        self.ids_tst =  np.arange(self.tst.shape[0]) + self.trn_val.shape[0]
        self.ids_all = np.concatenate([self.ids_trn_val, self.ids_tst])

        self.all = pd.concat((self.trn_val, self.tst))

        self.data = pd.concat((self.trn_val.loc[:, self.features_names], self.tst.loc[:, self.features_names]))
        self.data = self.data.astype('float32')
        if self.task == 'regression':
            self.output = self.all.loc[:, [self.outcome]]
            self.output = self.output.astype('float32')
        elif self.task in ['binary', 'multiclass']:
            self.output = self.all.loc[:, [self.outcome, f'{self.outcome}_origin']]

        if not list(self.data.index.values) == list(self.output.index.values):
            log.info(f"Error! Indexes have different order")
            raise ValueError(f"Error! Indexes have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.data.shape[1])

        self.dataset = DNAmDataset(self.data, self.output, self.outcome)

    def refresh_datasets(self):
        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)
        self.dataset_tst = Subset(self.dataset, self.ids_tst)

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

        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)
        self.dataset_tst = Subset(self.dataset, self.ids_tst)

        log.info(f"total_count: {len(self.dataset)}")
        log.info(f"trn_count: {len(self.dataset_trn)}")
        log.info(f"val_count: {len(self.dataset_val)}")
        log.info(f"tst_count: {len(self.dataset_tst)}")

    def plot_split(self, suffix=''):
        dict_to_plot = {
            "Train": self.ids_trn,
            "Val": self.ids_val,
            "Test": self.ids_tst
        }

        if self.task in ['binary', 'multiclass']:
            for name, ids in dict_to_plot.items():
                classes_counts = pd.DataFrame(Counter(self.output[f'{self.outcome}_origin'].values[ids]), index=[0])
                classes_counts = classes_counts.reindex(self.classes_df.loc[:, self.outcome].values, axis=1)
                fig = go.Figure()
                for st, st_id in self.classes_dict.items():
                    add_bar_trace(fig, x=[st], y=[classes_counts.at[0, st]], text=[classes_counts.at[0, st]], name=st)
                add_layout(fig, f"", f"Count", "")
                fig.update_layout({'colorway': px.colors.qualitative.Set1})
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
            fig.update_layout({'colorway': px.colors.qualitative.Set1}, barmode='overlay')
            save_figure(fig, f"hist{suffix}")

        self.output.loc[self.output.index[self.ids_trn], 'Part'] = "trn"
        self.output.loc[self.output.index[self.ids_val], 'Part'] = "val"
        self.output.loc[self.output.index[self.ids_tst], 'Part'] = "tst"

        self.output.to_excel(f"output{suffix}.xlsx", index=True)

    def get_trn_val_y(self):
        return self.dataset.ys[self.ids_trn_val]

    def train_dataloader(self):
        ys_trn = self.dataset.ys[self.ids_trn]
        if self.task in ['binary', 'multiclass'] and self.weighted_sampler:
            class_counter = Counter(ys_trn)
            class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
            weights = torch.FloatTensor([class_weights[y] for y in ys_trn])
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
        return self.data.columns.to_list()

    def get_outcome_name(self):
        return self.outcome

    def get_class_names(self):
        return list(self.classes_dict.keys())

    def get_df(self):
        df = pd.merge(self.output.loc[:, self.outcome], self.data, left_index=True, right_index=True)
        return df


class DNAmDataModuleTrainValNoSplit(LightningDataModule):

    def __init__(
            self,
            task: str = "",
            features_fn: str = "",
            classes_fn: str = "",
            trn_fn: str = "",
            val_fn: str = "",
            outcome: str = "",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 1337,
            weighted_sampler = False,
            imputation: str = "median",
            **kwargs,
    ):
        super().__init__()

        self.task = task
        self.features_fn = features_fn
        self.classes_fn = classes_fn
        self.trn_fn = trn_fn
        self.val_fn = val_fn
        self.outcome = outcome
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.weighted_sampler = weighted_sampler
        self.imputation = imputation

        self.dataset_trn: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_tst: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        self.trn = pd.read_pickle(f"{self.trn_fn}")
        self.val = pd.read_pickle(f"{self.val_fn}")
        features_df = pd.read_excel(self.features_fn)
        self.features_names = features_df.loc[:, 'features'].values

        if self.task in ['binary', 'multiclass']:
            self.classes_df = pd.read_excel(self.classes_fn)
            self.classes_dict = {}
            for cl_id, cl in enumerate(self.classes_df.loc[:, self.outcome].values):
                self.classes_dict[cl] = cl_id

            self.trn = self.trn.loc[self.trn[self.outcome].isin(self.classes_dict)]
            self.trn[f'{self.outcome}_origin'] = self.trn[self.outcome]
            self.trn[self.outcome].replace(self.classes_dict, inplace=True)

            self.val = self.val.loc[self.val[self.outcome].isin(self.classes_dict)]
            self.val[f'{self.outcome}_origin'] = self.val[self.outcome]
            self.val[self.outcome].replace(self.classes_dict, inplace=True)

        missed_features = list(set(self.features_names) - set(self.val.columns.values))
        exist_features = list(set(self.features_names) - set(missed_features))
        if len(missed_features) > 0:
            log.info(f"Perform imputation for {len(missed_features)} features with {self.imputation}")
            if self.imputation == "median":
                for f_id, f in enumerate(tqdm(missed_features, desc=f"{self.imputation} calculation")):
                    self.val.loc[:, f] = self.trn[f].median()
            elif self.imputation == "mean":
                for f_id, f in enumerate(tqdm(missed_features, desc=f"{self.imputation} calculation")):
                    self.val.loc[:, f] = self.trn[f].mean()
            else:
                raise ValueError(f"Unsupported imputation: {self.imputation}")

        self.ids_trn = np.arange(self.trn.shape[0])
        self.ids_val =  np.arange(self.val.shape[0]) + self.trn.shape[0]
        self.ids_tst = None
        self.ids_trn_val = np.concatenate([self.ids_trn, self.ids_val])

        self.all = pd.concat((self.trn, self.val))

        self.data = pd.concat((self.trn.loc[:, self.features_names], self.val.loc[:, self.features_names]))
        self.data = self.data.astype('float32')
        if self.task == 'regression':
            self.output = self.all.loc[:, [self.outcome]]
            self.output = self.output.astype('float32')
        elif self.task in ['binary', 'multiclass']:
            self.output = self.all.loc[:, [self.outcome, f'{self.outcome}_origin']]

        if not list(self.data.index.values) == list(self.output.index.values):
            log.info(f"Error! Indexes have different order")
            raise ValueError(f"Error! Indexes have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.data.shape[1])

        self.dataset = DNAmDataset(self.data, self.output, self.outcome)

    def refresh_datasets(self):
        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)

    def perform_split(self):
        self.dataset_trn = Subset(self.dataset, self.ids_trn)
        self.dataset_val = Subset(self.dataset, self.ids_val)

        log.info(f"total_count: {len(self.dataset)}")
        log.info(f"trn_count: {len(self.dataset_trn)}")
        log.info(f"val_count: {len(self.dataset_val)}")

    def plot_split(self, suffix=''):
        dict_to_plot = {
            "Train": self.ids_trn,
            "Val": self.ids_val,
        }

        if self.task in ['binary', 'multiclass']:
            for name, ids in dict_to_plot.items():
                classes_counts = pd.DataFrame(Counter(self.output[f'{self.outcome}_origin'].values[ids]), index=[0])
                classes_counts = classes_counts.reindex(self.classes_df.loc[:, self.outcome].values, axis=1)
                fig = go.Figure()
                for st, st_id in self.classes_dict.items():
                    add_bar_trace(fig, x=[st], y=[classes_counts.at[0, st]], text=[classes_counts.at[0, st]], name=st)
                add_layout(fig, f"", f"Count", "")
                fig.update_layout({'colorway': px.colors.qualitative.Set1})
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
            fig.update_layout({'colorway': px.colors.qualitative.Set1}, barmode='overlay')
            save_figure(fig, f"hist{suffix}")

        self.output.loc[self.output.index[self.ids_trn], 'Part'] = "trn"
        self.output.loc[self.output.index[self.ids_val], 'Part'] = "val"

        self.output.to_excel(f"output{suffix}.xlsx", index=True)

    def get_trn_val_y(self):
        return self.dataset.ys[self.ids_trn_val]

    def train_dataloader(self):
        ys_trn = self.dataset.ys[self.ids_trn]
        if self.task in ['binary', 'multiclass'] and self.weighted_sampler:
            class_counter = Counter(ys_trn)
            class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
            weights = torch.FloatTensor([class_weights[y] for y in ys_trn])
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


class DNAmDataModuleInference(LightningDataModule):

    def __init__(
            self,
            task: str = "",
            features_fn: str = "",
            classes_fn: str = "",
            trn_val_fn: str = "",
            inference_fn: str = "",
            outcome: str = "",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            imputation: str = "median",
            **kwargs,
    ):
        super().__init__()

        self.task = task
        self.features_fn = features_fn
        self.classes_fn = classes_fn
        self.trn_val_fn = trn_val_fn
        self.inference_fn = inference_fn
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

        self.trn_val = pd.read_pickle(f"{self.trn_val_fn}")
        self.inference = pd.read_pickle(f"{self.inference_fn}")

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

            self.inference = self.inference.loc[self.inference[self.outcome].isin(self.classes_dict)]
            self.inference[f'{self.outcome}_origin'] = self.inference[self.outcome]
            self.inference[self.outcome].replace(self.classes_dict, inplace=True)

        missed_features = list(set(self.features_names) - set(self.inference.columns.values))
        missed_features_df = pd.DataFrame(index=missed_features, columns=["index"])
        for mf in missed_features:
            missed_features_df.at[mf, "index"] = np.where(self.features_names == mf)
        missed_features_df.to_excel("missed_features.xlsx")
        exist_features = list(set(self.features_names) - set(missed_features))
        if len(missed_features) > 0:
            log.info(f"Perform imputation for {len(missed_features)} features with {self.imputation}")
            if self.imputation == "median":
                for f_id, f in enumerate(tqdm(missed_features, desc=f"{self.imputation} calculation")):
                    self.inference.loc[:, f] = self.trn_val[f].median()
            elif self.imputation == "mean":
                for f_id, f in enumerate(tqdm(missed_features, desc=f"{self.imputation} calculation")):
                    self.inference.loc[:, f] = self.trn_val[f].mean()
            else:
                raise ValueError(f"Unsupported imputation: {self.imputation}")

        self.data = self.inference.loc[:, self.features_names]
        self.data = self.data.astype('float32')
        if self.task == 'regression':
            self.output = self.inference.loc[:, [self.outcome]]
            self.output = self.output.astype('float32')
        elif self.task in ['binary', 'multiclass']:
            self.output = self.inference.loc[:, [self.outcome, f'{self.outcome}_origin']]

        if not list(self.data.index.values) == list(self.output.index.values):
            log.info(f"Error! Indexes have different order")
            raise ValueError(f"Error! Indexes have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.data.shape[1])

        self.dataset = DNAmDataset(self.data, self.output, self.outcome)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def get_feature_names(self):
        return self.data.columns.to_list()

    def get_outcome_name(self):
        return self.outcome

    def get_class_names(self):
        return list(self.classes_dict.keys())

    def get_df(self):
        df = pd.merge(self.output.loc[:, self.outcome], self.data, left_index=True, right_index=True)
        return df


class DNAmDataModuleImpute(LightningDataModule):

    def __init__(
            self,
            task: str = "",
            features_fn: str = "",
            features_impute_fn = "",
            classes_fn: str = "",
            trn_val_fn: str = "",
            inference_fn: str = "",
            outcome: str = "",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            imputation: str = "median",
            k: int = 1,
            **kwargs,
    ):
        super().__init__()

        self.task = task
        self.features_fn = features_fn
        self.features_impute_fn = features_impute_fn
        self.classes_fn = classes_fn
        self.trn_val_fn = trn_val_fn
        self.inference_fn = inference_fn
        self.outcome = outcome
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.imputation = imputation
        self.k = k

        self.dataset: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):

        self.trn_val = pd.read_pickle(f"{self.trn_val_fn}")
        self.inference = pd.read_pickle(f"{self.inference_fn}")

        features_df = pd.read_excel(self.features_fn)
        self.features_names = features_df.loc[:, 'features'].values

        features_impute_df = pd.read_excel(self.features_impute_fn)
        missed_features = features_impute_df.loc[:, 'features'].values

        if self.task in ['binary', 'multiclass']:
            self.classes_df = pd.read_excel(self.classes_fn)
            self.classes_dict = {}
            for cl_id, cl in enumerate(self.classes_df.loc[:, self.outcome].values):
                self.classes_dict[cl] = cl_id

            self.trn_val = self.trn_val.loc[self.trn_val[self.outcome].isin(self.classes_dict)]
            self.trn_val[f'{self.outcome}_origin'] = self.trn_val[self.outcome]
            self.trn_val[self.outcome].replace(self.classes_dict, inplace=True)

            self.inference = self.inference.loc[self.inference[self.outcome].isin(self.classes_dict)]
            self.inference[f'{self.outcome}_origin'] = self.inference[self.outcome]
            self.inference[self.outcome].replace(self.classes_dict, inplace=True)

        missed_features = list(set(missed_features).union(set(self.features_names) - set(self.inference.columns.values)))
        missed_features_df = pd.DataFrame(index=missed_features, columns=["index"])
        for mf in missed_features:
            missed_features_df.at[mf, "index"] = np.where(self.features_names == mf)
        missed_features_df.to_excel("missed_features.xlsx")
        exist_features = list(set(self.features_names) - set(missed_features))
        if len(missed_features) > 0:
            log.info(f"Perform imputation for {len(missed_features)} features with {self.imputation}")
            inference_index = self.inference.index.values
            df = pd.concat([self.trn_val.loc[:, self.features_names], self.inference.loc[:, exist_features]])
            df = df.astype(np.float)
            if self.imputation == "median":
                imputed_training = median(df.loc[:, self.features_names].values)
            elif self.imputation == "mean":
                imputed_training = mean(df.loc[:, self.features_names].values)
            elif self.imputation == "fast_knn":
                imputed_training = fast_knn(df.loc[:, self.features_names].values, k=self.k)
            elif self.imputation == "random":
                imputed_training = random(df.loc[:, self.features_names].values)
            elif self.imputation == "mice":
                imputed_training = mice(df.loc[:, self.features_names].values)
            elif self.imputation == "em":
                imputed_training = em(df.loc[:, self.features_names].values)
            elif self.imputation == "mode":
                imputed_training = mode(df.loc[:, self.features_names].values)
            else:
                raise ValueError(f"Unsupported imputation: {self.imputation}")
            df.loc[:, :] = imputed_training
            self.inference.loc[inference_index, self.features_names] = df.loc[inference_index, self.features_names]
        self.data = self.inference.loc[:, self.features_names]
        self.data = self.data.astype('float32')
        if self.task == 'regression':
            self.output = self.inference.loc[:, [self.outcome]]
            self.output = self.output.astype('float32')
        elif self.task in ['binary', 'multiclass']:
            self.output = self.inference.loc[:, [self.outcome, f'{self.outcome}_origin']]

        if not list(self.data.index.values) == list(self.output.index.values):
            log.info(f"Error! Indexes have different order")
            raise ValueError(f"Error! Indexes have different order")

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.data.shape[1])

        self.dataset = DNAmDataset(self.data, self.output, self.outcome)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def get_feature_names(self):
        return self.data.columns.to_list()

    def get_outcome_name(self):
        return self.outcome

    def get_class_names(self):
        return list(self.classes_dict.keys())

    def get_df(self):
        df = pd.merge(self.output.loc[:, self.outcome], self.data, left_index=True, right_index=True)
        return df
