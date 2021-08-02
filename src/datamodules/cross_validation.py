from abc import abstractmethod, ABC
from typing import Tuple
import torch
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, ConcatDataset, Subset, WeightedRandomSampler


class CVDataModule(ABC):

    def __init__(self,
                 data_module,
                 n_splits: int = 5,
                 shuffle: bool = True):
        self.data_module = data_module
        self._n_splits = n_splits
        self._shuffle = shuffle

    @abstractmethod
    def split(self):
        pass


class KFoldCVDataModule(CVDataModule):
    """
        K-fold cross-validation data module

    Args:
        data_module: data module containing data to be split
        n_splits: number of k-fold iterations/data splits
    """

    def __init__(self,
                 data_module,
                 n_splits: int = 5):
        super().__init__(data_module, n_splits)
        self._k_fold = KFold(n_splits=self._n_splits, shuffle=self._shuffle)

    def get_data(self):
        """
            Extract and concatenate training and validation datasets from data module.
        """
        self.data_module.setup()
        train_ds = self.data_module.train_dataloader().dataset
        val_ds = self.data_module.val_dataloader().dataset
        return ConcatDataset([train_ds, val_ds])

    def get_test_dataloader(self):
        return self.data_module.test_dataloader()

    def split(self) -> Tuple[DataLoader, DataLoader]:
        """
            Split data into k-folds and yield each pair
        """
        # 0. Get data to split
        data = self.get_data()

        # 1. Iterate through splits
        for train_idx, val_idx in self._k_fold.split(range(len(data))):

            train_dl = DataLoader(Subset(data, train_idx),
                                  batch_size=self.data_module.batch_size,
                                  num_workers=self.data_module.num_workers,
                                  pin_memory=self.data_module.pin_memory,
                                  shuffle=self._shuffle)
            val_dl = DataLoader(Subset(data, val_idx),
                                batch_size=self.data_module.batch_size,
                                num_workers=self.data_module.num_workers,
                                pin_memory=self.data_module.pin_memory,
                                shuffle=self._shuffle)

            yield train_dl, val_dl


class StratifiedKFoldCVDataModule(CVDataModule):
    """
        Stratified K-fold cross-validation data module

    Args:
        data_module: data module containing data to be split
        n_splits: number of k-fold iterations/data splits
    """

    def __init__(self,
                 data_module,
                 n_splits: int = 5):
        super().__init__(data_module, n_splits)
        self._k_fold = StratifiedKFold(n_splits=self._n_splits, shuffle=self._shuffle)

    def get_test_dataloader(self):
        return self.data_module.test_dataloader()

    def split(self) -> Tuple[DataLoader, DataLoader]:
        """
            Split data into k-folds and yield each pair
        """
        # 0. Get data to split
        self.data_module.setup()
        train_val_dataset, train_val_labels = self.data_module.get_train_val_dataset_and_labels()
        weighted_sampler = self.data_module.get_weighted_sampler()

        # 1. Iterate through splits
        for train_idx, val_idx in self._k_fold.split(range(len(train_val_dataset)), train_val_labels):
            y_train = train_val_labels[train_idx]
            class_counter = Counter(y_train)
            class_weights = {c: 1.0 / class_counter[c] for c in class_counter}
            weights = torch.FloatTensor([class_weights[y] for y in y_train])
            if weighted_sampler:
                weighted_sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True
                )
                train_dl = DataLoader(Subset(train_val_dataset, train_idx),
                                      batch_size=self.data_module.batch_size,
                                      num_workers=self.data_module.num_workers,
                                      pin_memory=self.data_module.pin_memory,
                                      sampler=weighted_sampler)
            else:
                train_dl = DataLoader(Subset(train_val_dataset, train_idx),
                                      batch_size=self.data_module.batch_size,
                                      num_workers=self.data_module.num_workers,
                                      pin_memory=self.data_module.pin_memory,
                                      shuffle=self._shuffle)
            val_dl = DataLoader(Subset(train_val_dataset, val_idx),
                                batch_size=self.data_module.batch_size,
                                num_workers=self.data_module.num_workers,
                                pin_memory=self.data_module.pin_memory,
                                shuffle=self._shuffle)

            yield train_dl, val_dl