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

class EEGDataset(Dataset):

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


class EEGDataModuleSeparate(LightningDataModule):

    def __init__(
            self,
            path: str = "",
            features_fn: str = "",
            classes_fn: str = "",
            trn_val_fn: str = "",
            tst_fn: str = "",
            outcome: str = "",
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

        self.dataset_trn: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_tst: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        self.trn_val = pd.read_excel(f"{self.path}/{self.trn_val_fn}", index_col="index")
        self.tst = pd.read_excel(f"{self.path}/{self.tst_fn}", index_col="index")
        features_df = pd.read_excel(self.features_fn)
        self.features_names = features_df.loc[:, 'features'].values
        classes_df = pd.read_excel(self.classes_fn)

        self.classes_dict = {}
        for cl_id, cl in enumerate(classes_df.loc[:, self.outcome].values):
            self.classes_dict[cl] = cl_id

        self.trn_val = self.trn_val.loc[self.trn_val[self.outcome].isin(self.classes_dict)]
        self.trn_val[f'{self.outcome}_origin'] = self.trn_val[self.outcome]
        self.trn_val[self.outcome].replace(self.classes_dict, inplace=True)

        self.tst = self.tst.loc[self.tst[self.outcome].isin(self.classes_dict)]
        self.tst[f'{self.outcome}_origin'] = self.tst[self.outcome]
        self.tst[self.outcome].replace(self.classes_dict, inplace=True)

        self.data_trn_val = self.trn_val.loc[:, self.features_names]
        self.output_trn_val = self.trn_val.loc[:, [self.outcome, f'{self.outcome}_origin']]
        self.data_tst = self.tst.loc[:, self.features_names]
        self.output_tst = self.tst.loc[:, [self.outcome, f'{self.outcome}_origin']]

        self.data = pd.concat([self.data_trn_val, self.data_tst])
        self.output = pd.concat([self.output_trn_val, self.output_tst])
        if not list(self.data.index.values) == list(self.output.index.values):
            log.info(f"Error! Indexes have different order")
            raise ValueError(f"Error! Indexes have different order")

        self.ids_trn_val = np.arange(self.trn_val.shape[0])
        self.ids_tst = np.arange(self.tst.shape[0]) + self.trn_val.shape[0]

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.data.shape[1])

        self.dataset = EEGDataset(self.data, self.output, self.outcome)

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

            classes_counts = pd.DataFrame(Counter(self.output[f'{self.outcome}_origin'].values[ids]), index=[0])
            classes_counts = classes_counts.reindex(classes_df.loc[:, self.outcome].values, axis=1)
            fig = go.Figure()
            for st, st_id in self.classes_dict.items():
                add_bar_trace(fig, x=[st], y=[classes_counts.at[0, st]], text=[classes_counts.at[0, st]], name=st)
            add_layout(fig, f"", f"Count", "")
            fig.update_layout({'colorway': px.colors.qualitative.Set1})
            fig.update_xaxes(showticklabels=False)
            save_figure(fig, f"bar_{name}")

        self.output.loc[self.output.index[self.ids_trn], 'Part'] = "trn"
        self.output.loc[self.output.index[self.ids_val], 'Part'] = "val"
        self.output.loc[self.output.index[self.ids_tst], 'Part'] = "tst"

        self.output.to_excel(f"output.xlsx", index=True)

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
        return self.data.columns.to_list()

    def get_class_names(self):
        return list(self.classes_dict.keys())

    def get_raw_data(self):
        data = pd.merge(self.output.loc[:, self.outcome], self.data, left_index=True, right_index=True)
        train_data = data.iloc[self.ids_trn]
        val_data = data.iloc[self.ids_val]
        test_data = data.iloc[self.ids_tst]
        raw_data = {}
        raw_data['X_train'] = train_data.loc[:, self.data.columns.values].values
        raw_data['y_train'] = train_data.loc[:, self.outcome].values
        raw_data['X_val'] = val_data.loc[:, self.data.columns.values].values
        raw_data['y_val'] = val_data.loc[:, self.outcome].values
        raw_data['X_test'] = test_data.loc[:, self.data.columns.values].values
        raw_data['y_test'] = test_data.loc[:, self.outcome].values
        return raw_data
