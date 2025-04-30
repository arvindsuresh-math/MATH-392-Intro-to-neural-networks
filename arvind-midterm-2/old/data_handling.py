# =============================================================================
# Data Handling - Election Prediction Project
# =============================================================================
"""
Provides classes and functions for loading, preprocessing, and structuring
the county-level election and demographic data. Includes weighted scaling
and data splitting for cross-validation and final training/testing.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple, Dict, Any

# Import global definitions if needed (e.g., targets, all_features)
# Or load variables.json here if preferred over global utils.py definition
from utils import targets, years, idx, all_features


# =============================================================================
# WeightedStandardScaler Class
# =============================================================================
class WeightedStandardScaler:
    """Standard scaler that uses sample‐weights to compute mean & var. Assumes that the sample weights form a probability distribution over all samples (in particular, weights are non-negative and sum to 1)."""
    def __init__(self):
        self.mean_: Union[np.ndarray, None] = None
        self.scale_: Union[np.ndarray, None] = None

    def fit(self, X: np.ndarray, weights: np.ndarray):
        w = weights.reshape(-1, 1)
        # weighted mean
        self.mean_ = (X * w).sum(axis=0)
        # weighted variance
        var = (w * (X - self.mean_)**2).sum(axis=0)
        # Add epsilon to variance before sqrt for numerical stability if needed
        epsilon = 1e-9
        self.scale_ = np.sqrt(var + epsilon)
        return self

    def transform(self, X: np.ndarray):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fit yet.")
        # Handle potential division by zero if scale_ is zero for some features
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, weights: np.ndarray):
        return self.fit(X, weights).transform(X)

# =============================================================================
# DataHandler Class
# =============================================================================
class DataHandler:
    """Handles loading data, preprocessing, splitting, and creating DataLoaders."""

    def __init__(self,
                 data_csv_path: str,
                 test_year: int = 2020,
                 batch_size: int = 64,
                 features_to_drop: List[str] = []):
        """
        Initializes the DataHandler.

        Args:
            data_csv_path (str): Path to the final_dataset.csv file.
            test_year (int): Year to use for test data (default: 2020).
            batch_size (int): Batch size for DataLoader creation (default: 64).
            features_to_drop (List[str]): List of features to drop (default: []).
        """
        self.data_csv_path = data_csv_path
        self.test_year = test_year
        self.batch_size = batch_size
        # Note: Uses global 'targets', 'years', 'all_features' from utils.py
        self.features = sorted(set(all_features) - set(features_to_drop))
        self.input_dim = len(self.features)
        self.train_years = sorted(set(years) - {test_year})

        if not targets or not years or not all_features:
             raise ValueError("Could not load 'targets', 'years', or 'all_features'. Check variables.json and utils.py.")

        # Prepare arguments to create cross-validation dataset
        self.folds = [
                ([self.train_years[0], self.train_years[1]], self.train_years[2]),
                ([self.train_years[0], self.train_years[2]], self.train_years[1]),
                ([self.train_years[1], self.train_years[2]], self.train_years[0])
                    ]
        # Create cross-validation and final training datasets (as NumPy arrays)
        self.cv_data: List[Tuple[Tuple, Tuple]] = [
            self._load_data(train_yrs, val_yr) for train_yrs, val_yr in self.folds
        ]
        self.final_data: Tuple[Tuple, Tuple] = self._load_data(self.train_years, self.test_year)

        # Create DataLoaders for cross-validation and final training
        self.cv_dataloaders: List[Tuple[DataLoader, DataLoader]] = [
            self._create_dataloaders(train_data, val_data, self.batch_size) for train_data, val_data in self.cv_data
        ]
        self.final_dataloaders: Tuple[DataLoader, DataLoader] = self._create_dataloaders(
            self.final_data[0], self.final_data[1], self.batch_size
        )

        print(f"DataHandler initialized:")
        print(f"  Data path: {self.data_csv_path}")
        print(f"  Using {self.input_dim} features.")
        print(f"  Test year: {self.test_year}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Datasets and DataLoaders created for CV and final training.")

    def _load_data(self,
                   fit_years: List[int], # list of years to fit StandardScaler
                   transform_year: int # year to transform with StandardScaler
                   ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Loads and preprocesses data for a specific train/validation split."""
        try:
            data = pd.read_csv(self.data_csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_csv_path}")

        # Make datasets with fit years and transform years
        df_fit = data[data['year'].isin(fit_years)].reset_index(drop=True)
        df_transform = data[data['year'] == transform_year].reset_index(drop=True)

        # Make wts arrays (ensure they are treated as numerical)
        wts_fit = df_fit['P(C)'].values.astype(np.float32)
        wts_transform = df_transform['P(C)'].values.astype(np.float32)

        # Make y's (arrays of shape [n_samples, n_targets])
        y_fit = df_fit[targets].values.astype(np.float32)
        y_transform = df_transform[targets].values.astype(np.float32)

        # Make X's (arrays of shape [n_samples, n_features])
        X_fit = df_fit[self.features].values.astype(np.float32)
        X_transform = df_transform[self.features].values.astype(np.float32)

        # Apply weighted StandardScaler to X's
        scaler = WeightedStandardScaler()
        X_fit_scaled = scaler.fit_transform(X_fit, wts_fit) # Use raw weights
        X_transform_scaled = scaler.transform(X_transform)

        # Return tuples of NumPy arrays (X, y, wts)
        train_output = (X_fit_scaled, y_fit, wts_fit)
        val_output = (X_transform_scaled, y_transform, wts_transform)
        return train_output, val_output

    def _create_dataset(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> TensorDataset:
        """Converts NumPy arrays (X, y, wts) to a PyTorch TensorDataset."""
        X_np, y_np, wts_np = data
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        wts_tensor = torch.tensor(wts_np, dtype=torch.float32).unsqueeze(1) # Ensure shape [n_samples, 1]
        return TensorDataset(X_tensor, y_tensor, wts_tensor)

    def _create_dataloaders(self,
                            train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                            val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                            batch_size: int
                            ) -> Tuple[DataLoader, DataLoader]:
        """Creates DataLoaders for training and validation sets."""

        # Create tensor datasets
        train_dataset = self._create_dataset(train_data)
        val_dataset = self._create_dataset(val_data)

        # Note: Global seed setting should happen in the main notebook/script ONCE.
        # torch.manual_seed(42) # Removed from here

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0) # Set num_workers if needed
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)

        return train_loader, val_loader