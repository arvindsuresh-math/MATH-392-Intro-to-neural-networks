"""
Data loading, preprocessing, and utility classes for the election prediction project.

Provides:
- WeightedStandardScaler: For feature scaling considering sample weights.
- DataHandler: Manages dataset loading, splitting for CV/train/test,
               and formatting data for different model types.
"""
import os
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Tuple, Any

from .constants import DATA_DIR # 

# --- WeightedStandardScaler Class ---

class WeightedStandardScaler:
    """Weighted version of StandardScaler that uses sample weights to compute mean & var."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray, weights: np.ndarray):
        """Fits the scaler to the data X using sample weights."""
        w = weights.reshape(-1, 1)
        self.mean_ = (X * w).sum(axis=0) / w.sum(axis=0)
        var = (w * (X - self.mean_)**2).sum(axis=0) / w.sum(axis=0)
        self.scale_ = np.sqrt(var)
        return self

    def transform(self, X: np.ndarray):
        """Transforms the data X using the fitted scaler."""
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, weights: np.ndarray):
        """Fits and transforms the data X using sample weights."""
        return self.fit(X, weights).transform(X)

    def inverse_transform(self, X: np.ndarray):
        """Undo the scaling of X: X_original = X_scaled * scale_ + mean_."""
        return X * self.scale_ + self.mean_

# --- DataHandler Class ---

class DataHandler:
    """Handles loading data for CV/training/testing in the correct format for each model type."""

    def __init__(self, test_year: int = 2020, features_to_drop = ['P(C)']):
        """
        Initializes the DataHandler and determines input/output dimensions.

        Attr:
            raw_data: Raw DataFrame loaded from CSV.
            idx: Index columns = ['year', 'gisjoin', 'state', 'county']
            targets: Targets = ['P(democrat|C)', 'P(republican|C)', 'P(other|C)', 'P(non-voter|C)']
            features: List of features used for model input.
            input_dim: Number of features used for model input.
            years: List of years = [2008, 2012, 2016, 2020].
            test_year: Year used for test data.
            train_years: List of years used for training data (i.e. all but test_year).
            folds: List of tuples (train_years,val_year) defining the cross-validation splits.
            cv_data: List of tuples (train_data, val_data) for each CV fold, where each data is itself a triple (X, y, wts).
            final_data: Final training dataset, of the form (X_train, y_train, wts_train), (X_test, y_test, wts_test).
            cv_dataloaders: List of tuples (train_loader, val_loader) for each CV fold.
            final_dataloader: train_loader for final training.
            test_set: TensorDataset for the final test set, of the form (X_test, y_test, wts_test).
        """
        self.raw_data = pd.read_csv(os.path.join(DATA_DIR, 'final_dataset.csv'))

        # --- Feature and Target Definitions ---
        with open(os.path.join(DATA_DIR, 'variables.json'), 'r') as f:
            vars = json.load(f)
        self.idx = vars['idx']
        self.targets = vars['targets']
        feature_keys = set(vars.keys()) - set(['targets', 'years', 'idx'])
        self.features = sorted([item for key in feature_keys for item in vars[key] if item not in features_to_drop])
        self.input_dim = len(self.features)

        # --- Years for CV, Training, Testing ---
        self.years = vars['years']
        self.test_year = test_year
        self.train_years = sorted(set(self.years) - {test_year})
        self.cv_years = [
                ([self.train_years[0], self.train_years[1]], self.train_years[2]),
                ([self.train_years[0], self.train_years[2]], self.train_years[1]),
                ([self.train_years[1], self.train_years[2]], self.train_years[0])
                    ]
        
        self.cv_scalers = [self._fit_scaler(fit_years=train_years) for (train_years,_) in self.cv_years]
        self.final_scaler = self._fit_scaler(fit_years=self.train_years)

        print(f"DataHandler initialized - Using {self.input_dim} features - Test year: {self.test_year}")

    def _load_data(self,
                   train_years: List[int], 
                   test_year: int 
                   ):
        """Loads data for a specific train/validation split."""
        data = self.raw_data

        # Make datasets with fit years and transform years
        df_train = data[data['year'].isin(train_years)].reset_index(drop=True)
        df_test = data[data['year'] == test_year].reset_index(drop=True)

        # Make wts 
        wts_train = df_train['P(C)'].values.reshape(-1, 1)  # Ensure shape [n_samples, 1]
        wts_test = df_test['P(C)'].values.reshape(-1, 1)

        # Make y's (Shape [n_samples, 4])
        y_train = df_train[self.targets].values
        y_test = df_test[self.targets].values

        # Make X's (Shape [n_samples, n_features])
        X_train = df_train[self.features].values
        X_test = df_test[self.features].values

        return (X_train, y_train, wts_train), (X_test, y_test, wts_test)

    def _fit_scaler(self, fit_years: List[int]):
        """Fits a WeightedStandardScaler to the given years."""
        df = self.raw_data[self.raw_data['year'].isin(fit_years)].reset_index(drop=True)
        wts = df['P(C)'].values
        X = df[self.features].values
        scaler = WeightedStandardScaler()
        scaler.fit(X, wts)
        return scaler

    def _create_tensors(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Converts NumPy arrays (X, y, wts) to Pytorch tensors."""
        X_np, y_np, wts_np = data
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        wts_tensor = torch.tensor(wts_np, dtype=torch.float32).unsqueeze(1) # Ensure shape [n_samples, 1]
        return (X_tensor, y_tensor, wts_tensor)

    def _create_dataloader(self,
                            data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                            batch_size: int,
                            shuffle: bool = True
                            ) -> Tuple[DataLoader, DataLoader]:
        """Creates DataLoaders for training and validation sets."""

        # Create tensor datasets
        X, y, wts = self._create_tensors(data)
        dataset = TensorDataset(X, y, wts)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0) # Set num_workers if needed
        return loader

    def _create_dmatrix(self,
                         data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         only_X: bool = False
                        ) -> xgb.DMatrix:
        """
        (Internal Helper) Creates an xgb.DMatrix from input data tuple.
        """
        X, y, wts = data

        if only_X:
            dmatrix = xgb.DMatrix(X)
        else:
            # Flatten labels assumes y has shape (n_samples, n_classes)
            # y_flattened = y.flatten()
            dmatrix = xgb.DMatrix(X, label=y, weight=wts)

        return dmatrix

    def get_ridge_data(self, task: str):
        """
        Returns the following depending on `task`:
        - 'cv': List of tuples (train_data, val_data) for each CV fold.
        - 'train': Tuple (X_train, y_train, wts_train).
        - 'test': Tuple (X_test, y_test, wts_test).
        """
        if task == 'cv':
            train_val_data_list = []
            for j, (train_years, val_year) in enumerate(self.cv_years):
                scaler = self.cv_scalers[j]
                (X_train, y_train, wts_train), (X_test, y_test, wts_test) = self._load_data(train_years, val_year)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                train_val_data_list.append(((X_train, y_train, wts_train), (X_test, y_test, wts_test)))
            return train_val_data_list
        elif task == 'train':
            (X_train, y_train, wts_train), _ = self._load_data(self.train_years, self.test_year)
            X_train = self.final_scaler.transform(X_train)
            return X_train, y_train, wts_train
        elif task == 'test':
            _, (X_test, y_test, wts_test) = self._load_data(self.train_years, self.test_year)
            X_test = self.final_scaler.transform(X_test)
            return X_test, y_test, wts_test
        else:
            raise ValueError("Invalid task specified. Use 'cv', 'train', or 'test'.")

    def get_nn_data(self, task: str, batch_size: int):
        """
        Returns the data needed for Neural Networks depending on task:
        - 'cv': List of tuples (train_loader, val_loader) for each CV fold.
        - 'train': Train_loader for 3 training years.
        - 'test': Tuple of tensors (X_test, y_test).
        """
        if task == 'cv':
            loaders = []
            for train_data, val_data in self.get_ridge_data('cv'):
                train_loader = self._create_dataloader(train_data, batch_size)
                val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
                loaders.append((train_loader, val_loader))
            return loaders
        elif task == 'train':
            train_data = self.get_ridge_data('train')
            return self._create_dataloader(train_data, batch_size)
        elif task == 'test':
            test_data = self.get_ridge_data('test')
            X_tensor, y_tensor, _ = self._create_tensors(test_data)
            return X_tensor, y_tensor

    def get_xgb_data(self, task: str):
        """Returns the data needed for XGBoost based on the task, depending on task:
        - 'cv': DMatrix with labels and weights with all train years combined; folds are determined internally by xgb.cv
        - 'train': Same as 'cv'.
        - 'test': DMatrix for test year (X only, no labels or weights).
        """
        if task == 'cv' or task == 'train':
            # Use all training years for XGBoost CV or final training
            train_data = self.get_ridge_data('train')
            return self._create_dmatrix(train_data, only_X=False)
        elif task == 'test':
            test_data = self.get_ridge_data('test')
            return self._create_dmatrix(test_data, only_X=True)
        else:
            raise ValueError("Invalid task specified. Use 'cv', 'train', or 'test'.")
        
