from abc import abstractmethod, ABC
from typing import Tuple
import torch
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from torch.utils.data import DataLoader, ConcatDataset, Subset, WeightedRandomSampler
import numpy as np


class CVSplitter(ABC):

    def __init__(self,
                 data_module,
                 n_splits: int = 5,
                 n_repeats: int = 1,
                 groups: str = "categorical",
                 random_state: int = 322,
                 shuffle: bool = False):
        self.data_module = data_module
        self._n_splits = n_splits
        self._n_repeats = n_repeats
        self._groups = groups
        self._random_state = random_state
        self._shuffle = shuffle


    @abstractmethod
    def split(self):
        pass


class RepeatedStratifiedKFoldCVSplitter(CVSplitter):
    """
        Stratified K-fold cross-validation data module

    Args:
        data_module: data module containing data to be split
        n_splits: number of k-fold iterations/data splits
    """

    def __init__(self,
                 data_module,
                 n_splits: int = 5,
                 n_repeats: int = 1,
                 groups: str = "categorical",
                 random_state: int = 322,
                 shuffle: bool = False
                 ):
        super().__init__(data_module, n_splits, n_repeats, groups, random_state, shuffle)
        self._k_fold = RepeatedStratifiedKFold(n_splits=self._n_splits, n_repeats=self._n_repeats, random_state=self._random_state)

    def get_test_dataloader(self):
        return self.data_module.test_dataloader()

    def split(self):
        """
            Split data into k-folds and yield each pair
        """
        # 0. Get data to split
        self.data_module.setup()
        train_val_X, train_val_y = self.data_module.get_trn_val_X_and_y()
        weighted_sampler = self.data_module.get_weighted_sampler()

        if self._groups == "categorical":
            splits = self._k_fold.split(X=range(len(train_val_X)), y=train_val_y, groups=train_val_y)
        elif self._groups == "continuous":
            ptp = np.ptp(train_val_y)
            num_bins = 3
            bins = np.linspace(np.min(train_val_y) - 0.1 * ptp, np.max(train_val_y) + 0.1 * ptp, num_bins + 1)
            binned = np.digitize(train_val_y, bins) - 1
            unique, counts = np.unique(binned, return_counts=True)
            occ = dict(zip(unique, counts))
            splits = self._k_fold.split(X=range(len(train_val_X)), y=binned, groups=binned)

        # 1. Iterate through splits
        for train_indexes, val_indexes in splits:
            y_train = train_val_y[train_indexes]
            if self._groups == "categorical" and weighted_sampler:
                class_counter = Counter(y_train)
                class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
                weights = torch.FloatTensor([class_weights[y] for y in y_train])
                weighted_sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True
                )
                train_dl = DataLoader(Subset(train_val_X, train_indexes),
                                      batch_size=self.data_module.batch_size,
                                      num_workers=self.data_module.num_workers,
                                      pin_memory=self.data_module.pin_memory,
                                      sampler=weighted_sampler)
            else:
                train_dl = DataLoader(Subset(train_val_X, train_indexes),
                                      batch_size=self.data_module.batch_size,
                                      num_workers=self.data_module.num_workers,
                                      pin_memory=self.data_module.pin_memory,
                                      shuffle=self._shuffle)
            val_dl = DataLoader(Subset(train_val_X, val_indexes),
                                batch_size=self.data_module.batch_size,
                                num_workers=self.data_module.num_workers,
                                pin_memory=self.data_module.pin_memory,
                                shuffle=self._shuffle)

            yield train_dl, train_indexes, val_dl, val_indexes
