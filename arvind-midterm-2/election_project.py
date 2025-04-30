# =============================================================================
# Presidential Election Outcome Prediction - Midterm Project Code
#
# High-Level Overview:
# This script provides a framework for predicting US presidential election
# outcomes (specifically, the probability of votes for Democrat, Republican,
# Other candidates, and Non-voters) at the county level. It leverages
# demographic features from multiple election years (2008, 2012, 2016, 2020).
# The primary goal is to compare the predictive performance of four different
# modeling approaches: Ridge Regression, Softmax Regression (implemented as a
# simple neural network), a 1-hidden-layer Multi-Layer Perceptron (MLP), and
# a 2-hidden-layer MLP.
#
# Code Structure:
# 0. Feature/Target Definitions: Lists defining input features and target variables.
# 1. Global Constants: Configuration for paths, device (CPU/GPU), and default hyperparameters.
# 2. DataHandler Class: Manages data loading, preprocessing (scaling), and splitting
#    for cross-validation and final training/testing. Ensures proper scaler fitting
#    to prevent data leakage.
# 3. RidgeModel Class: Implements Ridge Regression using scikit-learn, including
#    cross-validation for alpha selection, final model training, and evaluation
#    based on aggregate cross-entropy.
# 4. BaseNNModel Class: An abstract base class defining the common structure and
#    methods for PyTorch neural network models (Softmax, MLP1, MLP2). Handles
#    CV logic, training loops (with weighted cross-entropy loss), early stopping,
#    model saving/loading, and evaluation.
# 5. SoftmaxModel, MLP1Model, MLP2Model Classes: Concrete implementations inheriting
#    from BaseNNModel, each defining its specific network architecture (_build_network).
#
# Workflow (Example Usage in a Jupyter Notebook):
# 1. Instantiate `DataHandler` specifying the `test_year`.
# 2. Instantiate model handlers (e.g., `RidgeModel()`, `MLP1Model()`).
# 3. Create necessary directories (./data, ./models, ./results).
# 4. Run `cross_validate()` for each model handler, passing the `DataHandler` instance.
#    This performs 3-fold CV on the training years to find optimal hyperparameters,
#    saving results to CSV files in './results'.
# 5. Run `train_final_model()` for each model handler. This loads the best
#    hyperparameters found during CV, trains the model on all training years,
#    and saves the final model ('.joblib' for Ridge, '.pth' state_dict for NNs)
#    to './models'.
# 6. Run `evaluate()` for each model handler. This loads the trained model,
#    predicts on the `test_year` data, calculates the aggregate cross-entropy loss,
#    and saves county-level predictions to CSV files in './results'.
# 7. Compare the aggregate cross-entropy scores across models.
#
# File Locations:
# - ./data/final_dataset.csv : Input dataset (expected location).
# - ./models/ : Directory where trained models (and scaler) are saved.
# - ./results/ : Directory where CV results, final loss history (for NNs),
#                and county-level predictions are saved.
#
# =============================================================================

import os
import pandas as pd
import numpy as np
import math
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter          
from datetime import datetime  
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import joblib
from typing import List, Dict, Tuple, Any, Type, Union
from abc import ABC, abstractmethod
import copy

# =============================================================================
# 0. All features and targets
# =============================================================================

# --- Feature and Target Definitions ---
with open('variables.json', 'r') as f:
        vars = json.load(f)

# get lists of targets, years, and idx 
targets = vars['targets']
years = vars['years']
idx = vars['idx']

# all other keys in the dict are features
feature_keys = set(vars.keys()) - set(['targets', 'years', 'idx'])
all_features = [item for key in feature_keys for item in vars[key]]

# =============================================================================
# 1. Global Constants and Configuration
# =============================================================================

# --- File Paths ---
DATA_DIR = "./data"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
LOGS_DIR = "./logs"
PREDS_DIR = "./preds"

# --- Device Selection ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device")

# --- Default Training Hyperparameters ---
BATCH_SIZE: int = 64
MAX_CV_EPOCHS: int = 30 # Max epochs for CV
PATIENCE: int = 10      # Patience for early stopping during CV
FINAL_TRAIN_EPOCHS: int = 150 # Fixed epochs for final training
OPTIMIZER_CHOICE: Type[optim.Optimizer] = optim.AdamW # Default optimizer

# --- Default Hyperparameter Grids for CV ---
RIDGE_PARAM_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
SOFTMAX_PARAM_GRID = {
    'learning_rate': [1e-2, 1e-3, 1e-4],
    'weight_decay': [0, 1e-5, 1e-3]
}
MLP1_PARAM_GRID = {
    'n_hidden': [16, 64, 128],
    'dropout_rate': [0.1, 0.3, 0.5],
    'learning_rate': [1e-2, 1e-3, 1e-4]
    # Note: weight_decay could be added here too if desired
}
MLP2_PARAM_GRID = {
    'shared_hidden_size': [16, 32, 64],
    'dropout_rate': [0.1, 0.3, 0.5],
    'learning_rate': [1e-2, 1e-3, 1e-4]
    # Note: weight_decay could be added here too if desired
}

# --- XGBoost Hyperparameter Grid and Constants ---
XGB_PARAM_GRID = {
    'learning_rate': [0.05, 0.1, 0.2],     # Step size shrinkage (eta)
    'max_depth': [5, 7],                # Max depth of a tree
    'subsample': [0.8, 1.0],         # Fraction of samples used per tree
    'colsample_bytree': [0.8, 1.0],  # Fraction of features used per tree
    'gamma': [0.1, 0.2],                # Min loss reduction for split (min_split_loss)
    'reg_alpha': [0, 0.1, 1.0],            # L1 regularization
    'reg_lambda': [0, 0.1, 1.0],           # L2 regularization
    # Fixed parameters for consistency
    'objective': ['reg:squarederror'], # Regression objective for each target
    'n_estimators': [200],             # High initial value, CV uses early stopping
    'random_state': [42]               # For reproducibility
}

XGB_EARLY_STOPPING_ROUNDS = 20 # Early stopping rounds for CV fits

RUNG_EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200] # Rung epochs for MLP models
RUNG_PATIENCE = [15, 20, 25, 30, 35, 40, 45, 50] # Rung patience for MLP models

# =============================================================================
# 2. WeightedStandardScaler Class
# =============================================================================

class WeightedStandardScaler:
    """Standard scaler that uses sample‐weights to compute mean & var. Assumes that the sample weights form a probability distribution over all samples (in particular, weights are non-negative and sum to 1)."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray, weights: np.ndarray):
        w = weights.reshape(-1, 1)
        # weighted mean
        self.mean_ = (X * w).sum(axis=0)
        # weighted variance
        var = (w * (X - self.mean_)**2).sum(axis=0)
        self.scale_ = np.sqrt(var)
        return self

    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, weights: np.ndarray):
        return self.fit(X, weights).transform(X)

# =============================================================================
# 3. DataHandler Class
# =============================================================================

class DataHandler:
    """Handles loading data and creating PyTorch DataLoaders."""

    def __init__(self, test_year: int = 2020, features_to_drop = ['P(C)']):
        """
        Initializes the DataHandler and determines input/output dimensions.

        Args:
            test_year: Year to use for test data (default: 2020).
            features_to_drop: List of features to drop from the dataset (default: []).
        
        ---

        Attr:
            features: List of features used for model input.
            input_dim: Number of features used for model input.
            test_year: Year used for test data.
            train_years: List of years used for training data (i.e. all but test_year).
            folds: List of tuples (train_years,val_year) defining the cross-validation splits.
            cv_data: List of tuples (train_data, val_data) for each CV fold, where each data is itself a triple (X, y, wts).
            final_data: Final training dataset, of the form (X_train, y_train, wts_train), (X_test, y_test, wts_test).
            cv_dataloaders: List of tuples (train_loader, val_loader) for each CV fold.
            final_dataloaders: Tuple (train_loader, val_loader) for final training.
        """
        self.features = sorted(set(all_features) - set(features_to_drop))
        self.input_dim = len(self.features)
        self.test_year = test_year
        self.train_years = sorted(set(years) - {test_year})
        # prepare arguments to create cross-validation dataset
        self.folds = [
                ([self.train_years[0], self.train_years[1]], self.train_years[2]),
                ([self.train_years[0], self.train_years[2]], self.train_years[1]),
                ([self.train_years[1], self.train_years[2]], self.train_years[0])
                    ]
        # create cross-validation and final training datasets
        self.cv_data = [self._load_data(train_years, val_year) for train_years, val_year in self.folds]
        self.final_data = self._load_data(self.train_years, self.test_year)
        # create DataLoaders for cross-validation and final training
        self.cv_dataloaders = [self._create_dataloaders(train_data, val_data) for train_data, val_data in self.cv_data]
        self.final_dataloaders = self._create_dataloaders(self.final_data[0], self.final_data[1])

        print(f"DataHandler initialized:")
        print(f"  Using {self.input_dim} features")
        print(f"  Test year: {self.test_year}")
        print(f"  Dataloaders created for cross-validation and final training.")

    def _load_data(self,
                   fit_years: List[int], # list of years to fit StandardScaler
                   transform_year: int # year to transform with StandardScaler
                   ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Loads and preprocesses data for a specific train/validation split."""

        data = pd.read_csv(self.data_csv_path)

        # Make datasets with fit years and transform years
        df_fit = data[data['year'].isin(fit_years)].reset_index(drop=True)
        df_transform = data[data['year'] == transform_year].reset_index(drop=True)

        # Make wts arrays
        wts_fit = df_fit['P(C)'].values
        wts_transform = df_transform['P(C)'].values

        # Make y's (arrays of shape [n_samples, n_targets])
        y_fit = df_fit[targets].values
        y_transform = df_transform[targets].values

        # Make X's (arrays of shape [n_samples, n_features])
        X_fit = df_fit[self.features].values
        X_transform = df_transform[self.features].values

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
                            val_data: Tuple[np.ndarray, np.ndarray, np.ndarray]
                            ) -> Tuple[DataLoader, DataLoader]:
        """Creates DataLoaders for training and validation sets."""

        # Create tensor datasets
        train_dataset = self._create_dataset(train_data)
        val_dataset = self._create_dataset(val_data)

        # Note: Global seed setting should happen in the main notebook/script ONCE.
        # torch.manual_seed(42) # Removed from here

        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=0) # Set num_workers if needed
        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=0)

        return train_loader, val_loader

# =============================================================================
# 4. RidgeModel Class (Uses scikit-learn)
# =============================================================================

class RidgeModel:
    """Handles Ridge Regression CV, training, loading, and evaluation."""
    MODEL_NAME = "ridge"
    # Note: Ridge uses Mean Squared Error for CV, not weighted cross-entropy.

    def __init__(self):
        """Initializes the RidgeModel handler."""
        self.model: Union[Ridge, None] = None
        self.best_alpha: Union[float, None] = None
        # Store results path for convenience
        self.results_save_path = os.path.join(RESULTS_DIR, f"{self.MODEL_NAME}_cv_results.csv")
        self.model_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_final_model.joblib")

    def cross_validate(self,
                       dh: DataHandler,
                       param_grid: List[float] = RIDGE_PARAM_GRID
                       ) -> None:
        """Performs CV for Ridge, saves results, and stores the best alpha."""
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        results_list = []
        best_score = float('inf')
        current_best_alpha = None

        for alpha in param_grid:
            print("--------------------------------------")
            print(f"  Testing alpha = {alpha}:")
            fold_val_losses = []
            for j, ((X_train, y_train, wts_train), (X_val, y_val, wts_val)) in enumerate(dh.cv_data,1):
                # Fit Ridge model on training data
                model_fold = Ridge(alpha=alpha)
                model_fold.fit(X_train.values, y_train.values, sample_weight=wts_train.values.ravel())
                # Predict on validation data
                y_pred_val = model_fold.predict(X_val.values)

                # Use weighted MSE (weights = P(C) = wts_val) to compute validation loss
                weights_val = wts_val.values.squeeze() # Ensure weights is 1D array
                val_loss = mean_squared_error(y_val.values, y_pred_val, sample_weight=weights_val)

                fold_str = f" Fold {j} - Train Years: {dh.folds[j-1][0]} - Val Year: {dh.folds[j-1][1]} - "
                print(fold_str + f"Val Loss (Weighted MSE): {val_loss:.6f}")

                fold_val_losses.append(val_loss)

            # Calculate mean validation loss for this alpha
            mean_val_loss = np.mean(fold_val_losses)
            print(f"    Avg Val Score (Weighted MSE): {mean_val_loss:.6f}")

            results_list.append({'alpha': alpha, 'mean_cv_score': mean_val_loss})

            # Update best score and alpha if this fold is better
            if mean_val_loss < best_score:
                best_score = mean_val_loss
                current_best_alpha = alpha

        # Store best alpha found
        self.best_alpha = current_best_alpha 

        # Save results to CSV
        results_df = pd.DataFrame(results_list).sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(self.results_save_path, index=False)
        print(f"CV results saved to: {self.results_save_path}")

        # Print best alpha and score
        print(f"\nBest {self.MODEL_NAME.upper()} CV alpha: {self.best_alpha} (Score: {best_score:.6f})")
        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")

    def train_final_model(self,
                          dh: DataHandler
                          ) -> Ridge:
        """Trains the final Ridge model using the best alpha from CV results."""
        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")

        # 1. Read best alpha from CV results (or use stored one if CV was just run)
        if self.best_alpha is None:
            results_df = pd.read_csv(self.results_save_path)
            self.best_alpha = results_df.iloc[0]['alpha']
        print(f"Using best alpha: {self.best_alpha}")

        # 2. Build model instance
        self.model = Ridge(alpha=self.best_alpha)

        # 3. Load combined training data
        (X_train, y_train, wts_train), _ = dh.final_data
        print(f"Loaded final training data: {X_train.shape[0]} samples.")

        # 4. Fit the model
        self.model.fit(X_train.values, y_train.values, sample_weight=wts_train.values.ravel())
        # self.model.fit(X_train.values, y_train.values)
        print("Ridge model fitting complete.")

        # 5. Save the trained model
        joblib.dump(self.model, self.model_save_path)
        print(f"Saved final trained Ridge model to: {self.model_save_path}")
        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        return self.model

    def load_model(self, model_save_path = None) -> Ridge:
        """Loads a trained Ridge model into self.model."""
        if model_save_path is None: 
            model_save_path = self.model_save_path
        self.model = joblib.load(model_save_path)
        print(f"Ridge model loaded successfully from {model_save_path}.")

    def predict(self, dh: 'DataHandler', save: bool = False) -> pd.DataFrame:
        """Generates raw predictions for the test set using the trained Ridge model."""
        if self.model is None: self.load_model()

        _, (X_test, _, _) = dh.final_data
        y_pred = self.model.predict(X_test)

        if save:
            pred_df = pd.DataFrame(y_pred, columns=targets) # Use global 'targets'
            pred_save_path = os.path.join(PREDS_DIR, f"{dh.test_year}_{self.MODEL_NAME}_predictions.csv")
            pred_df.to_csv(pred_save_path, index=False)
            print(f"County-level raw predictions saved to: {pred_save_path}")
        
        return y_pred

# =============================================================================
# Consolidated Neural Network Model Class
# =============================================================================

class NNModel:
    """
    Handles Neural Network (Softmax, MLP) CV, training, loading, and prediction.
    Can build networks of varying depths dynamically based on hyperparameters.
    Each instance corresponds to a specific model type (e.g., NN0, NN1)
    determined by its model_name and associated param_grid.
    """

    def __init__(self, model_name: str, param_grid: Dict):
        """
        Initializes the NNModel handler for a specific model configuration.

        Args:
            model_name (str): A unique name for this model configuration
                              (e.g., "nn0", "nn1", "nn2"). Used for saving files.
            param_grid (Dict): The hyperparameter grid specific to this model
                               configuration to be used during CV. Must include
                               'hidden_layers' (list of lists) or similar keys
                               that _build_network can interpret.
        """
        self.model_name: str = model_name
        self.param_grid: Dict = param_grid
        self.model: Union[nn.Module, None] = None
        self.best_params: Dict = {}
        self.final_loss_history: List[Dict[str, Any]] = []

        # Define paths based on the specific model_name
        self.results_save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_cv_results.csv")
        self.state_dict_save_path = os.path.join(MODELS_DIR, f"{self.model_name}_final_state_dict.pth")
        self.loss_save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_final_training_loss.csv")
        print(f"NNModel initialized for '{self.model_name}'. CV results: {self.results_save_path}")

    @staticmethod
    def weighted_cross_entropy_loss(outputs: torch.Tensor,
                                    targets: torch.Tensor,
                                    weights: torch.Tensor) -> torch.Tensor:
        """
        Calculates a custom weighted cross-entropy loss for a batch.
        Loss(C) = - sum_k ( target_k * log(output_k) )
        Batch Loss = Expected sample loss = sum_C ( P(C) * Loss(C) ) / sum_C ( P(C) )

        Args:
            outputs (torch.Tensor): Model predictions (probabilities), shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth probabilities, shape [batch_size, num_classes].
            weights (torch.Tensor): Sample weights ('P(C)'), shape [batch_size, 1].

        Returns:
            torch.Tensor: Scalar tensor representing the weighted average loss.
        """
        epsilon = 1e-9
        outputs_clamped = torch.clamp(outputs, epsilon, 1. - epsilon)
        sample_ce_loss = -torch.sum(targets * torch.log(outputs_clamped), dim=1, keepdim=True)
        weights_reshaped = weights.view_as(sample_ce_loss)
        weighted_sample_losses = sample_ce_loss * weights_reshaped
        batch_loss = weighted_sample_losses.sum() / weights_reshaped.sum()
        return batch_loss

    def _build_network(self, params: Dict, input_dim: int) -> nn.Module:
        """
        Dynamically builds an nn.Sequential network based on hyperparameters.

        Args:
            params (Dict): Dictionary containing hyperparameters. Must include
                           'hidden_layers' (a list of integers specifying sizes
                           of hidden layers, e.g., [] for softmax, [64] for MLP1,
                           [32, 16] for MLP2) and 'dropout_rate'.
            input_dim (int): Input dimension for the network.

        Returns:
            nn.Module: The instantiated PyTorch sequential model.
        """
        hidden_layers_config = params.get('hidden_layers', []) # Default to empty list (Softmax)
        dropout_rate = params.get('dropout_rate', 0.0) # Default dropout if not specified
        output_dim = 4 # Hardcoded based on project definition (Dem, Rep, Other, Non-voter)

        layers = []
        current_dim = input_dim

        # Build hidden layers
        for hidden_size in hidden_layers_config:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_size # Update dimension for the next layer

        # Add final output layer and Softmax
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def _train_one_fold(self,
                          model: nn.Module,
                          optimizer: optim.Optimizer,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          rung_max_epochs: int,
                          rung_patience: int,
                          # --- Cumulative state passed in, tracked, and returned ---
                          # --- Assumes caller ensures these are valid initial values ---
                          best_model_state: Dict[str, torch.Tensor], # Previous best overall state
                          best_val_loss: float,
                          best_epoch: int,
                          train_loss_at_best: float,
                          last_epoch: int,
                          patience_used: int
                         ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int, int, float, float]:
        """
        Continues training a model for a single fold up to a target epoch count,
        resuming from its exact previous state and tracking cumulative patience.

        Receives the full cumulative state (including previous best state) and
        updates it based on training progress within this run. Stops if cumulative
        patience reaches `rung_patience` or `last_epoch` reaches `rung_max_epochs`.
        Minimalist implementation assumes valid non-None inputs are provided by the caller.

        Args:
            model (nn.Module): The PyTorch model instance. Assumed to be on DEVICE and
                               loaded with the *last state* from the previous run by the caller.
            optimizer (optim.Optimizer): The optimizer for the model.
            train_loader (DataLoader): DataLoader for the training set of this fold.
            val_loader (DataLoader): DataLoader for the validation set of this fold.
            rung_max_epochs (int): The target *total* number of epochs for this rung.
            rung_patience (int): The maximum cumulative patience allowed for this rung.
            best_model_state (Dict): State dict for the overall best validation loss seen so far.
            best_val_loss (float): The best validation loss achieved *across all previous runs*.
            best_epoch (int): The *cumulative* epoch number where `best_val_loss` occurred.
            train_loss_at_best (float): The training loss corresponding to `best_epoch`.
            last_epoch (int): The *cumulative* epoch number the model reached in the last run.
            patience_used (int): The cumulative count of epochs without improvement ending the last run.

        Returns:
            Tuple containing updated cumulative state for this fold:
              - best_model_state (Dict): Updated state dict corresponding to the overall best validation loss.
              - last_model_state (Dict): State dict at the exact point training stopped in this run.
              - patience_used (int): Updated cumulative patience count.
              - best_epoch (int): Updated cumulative epoch number where the best validation loss occurred.
              - last_epoch (int): Updated cumulative epoch number reached at the end of this run.
              - best_val_loss (float): Updated overall best validation loss.
              - train_loss_at_best (float): Updated training loss corresponding to the best validation loss epoch.
        """
        # Note: No internal initialization - relies on passed-in cumulative state arguments.
        # 'best_model_state' argument directly holds the state to be potentially returned.

        # --- Training Loop: Continue from last_epoch + 1 up to rung_max_epochs ---
        for current_epoch in range(last_epoch + 1, rung_max_epochs + 1):

            # --- Training Phase ---
            model.train()
            epoch_train_loss_sum = 0.0
            for features, targets_batch, weights in train_loader:
                features, targets_batch, weights = features.to(DEVICE), targets_batch.to(DEVICE), weights.to(DEVICE)
                outputs = model(features)
                loss = self.weighted_cross_entropy_loss(outputs, targets_batch, weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss_sum += loss.item()
            avg_train_loss = epoch_train_loss_sum / len(train_loader)

            # --- Validation Phase ---
            model.eval()
            epoch_val_loss_sum = 0.0
            with torch.no_grad():
                for features, targets_batch, weights in val_loader:
                    features, targets_batch, weights = features.to(DEVICE), targets_batch.to(DEVICE), weights.to(DEVICE)
                    outputs = model(features)
                    loss = self.weighted_cross_entropy_loss(outputs, targets_batch, weights)
                    epoch_val_loss_sum += loss.item()
            avg_val_loss = epoch_val_loss_sum / len(val_loader)

            # --- Update Cumulative State ---
            last_epoch = current_epoch # Update last epoch reached

            if avg_val_loss < best_val_loss:
                # Found a new overall best performance
                best_val_loss = avg_val_loss
                best_epoch = last_epoch # Update cumulative best epoch
                train_loss_at_best = avg_train_loss # Record corresponding train loss
                best_model_state = copy.deepcopy(model.state_dict()) # Update the best state *directly*
                patience_used = 0 # Reset cumulative patience
            else:
                # No improvement
                patience_used += 1 # Increment cumulative patience

            # --- Check Termination Conditions ---
            if patience_used >= rung_patience:
                break # Stop training for this fold/rung

        # --- Prepare Return Values ---
        last_model_state = copy.deepcopy(model.state_dict())

        # Return the updated cumulative metrics and states
        return (best_model_state, last_model_state, patience_used,
                best_epoch, last_epoch, best_val_loss, train_loss_at_best)

    def _train_one_config(self,
                            config: Dict[str, Any],
                            config_idx: int,
                            config_history: Dict[str, Any], # Renamed from history_entry
                            dh: 'DataHandler',
                            rung_epochs: int,
                            rung_patience: int,
                            optimizer_choice: Type[optim.Optimizer] = OPTIMIZER_CHOICE
                           ) -> None: # Returns None, updates config_history directly
        """
        Evaluates a single hyperparameter configuration across all CV folds
        up to a target cumulative epoch count, resuming state, and updates
        the provided config_history entry directly.

        Calls `_train_one_fold` for each fold, providing the necessary state
        information retrieved from `config_history`. Updates the fold-specific
        history tuples and the 'mean_val_loss' within the `config_history`.

        Args:
            config (Dict[str, Any]): The dictionary of hyperparameters for this config.
            config_idx (int): Unique index/ID for this configuration (can be useful, though not strictly used internally now).
            config_history (Dict[str, Any]): The dictionary associated with this config_idx
                                            in the main config history structure. This dictionary
                                            will be updated by this method.
            dh (DataHandler): The data handler instance providing CV DataLoaders.
            rung_epochs (int): The target *cumulative* number of epochs for this rung.
            rung_patience (int): The cumulative patience limit for the current rung.
            optimizer_choice (Type[optim.Optimizer]): The optimizer class to use.

        Returns:
            None: Modifies the `config_history` dictionary directly.
        """
        fold_best_val_losses = []
        fold_best_epochs = []

        # Loop through the cross-validation folds
        for j, (train_loader, val_loader) in enumerate(dh.cv_dataloaders,1):
            # --- Build model and optimizer ---
            model = self._build_network(config, dh.input_dim).to(DEVICE)
            optimizer = optimizer_choice(model.parameters(),
                                         lr=config['learning_rate'],
                                         weight_decay=config.get('weight_decay', 0))

            # --- Initialize or Unpack Cumulative State for the Fold ---
            if f'fold_{j}_history' not in config_history:
                # First run: initialize required arguments for _train_one_fold
                initial_state = copy.deepcopy(model.state_dict())
                best_model_state = initial_state
                last_model_state = initial_state # State to load (same as initial best)
                best_val_loss = float('inf')
                best_epoch = 0
                train_loss_at_best = float('inf')
                last_epoch = 0
                patience_used = 0
            else:
                # Resuming: directly unpack the previous state tuple
                # Tuple order: (best_state, last_state, patience, best_ep, last_ep, best_loss, train_loss)
                (best_model_state, last_model_state, patience_used, best_epoch,
                 last_epoch, best_val_loss, train_loss_at_best) = config_history[f'fold_{j}_history']

                # Load the *last* state from the previous run into the model
                model.load_state_dict(last_model_state) # Use the unpacked state

            # --- Call the training function for this fold ---
            updated_fold_history = self._train_one_fold(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                rung_max_epochs=rung_epochs,
                rung_patience=rung_patience,
                # Pass the unpacked or initial cumulative state values
                best_model_state=best_model_state,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                train_loss_at_best=train_loss_at_best,
                last_epoch=last_epoch,
                patience_used=patience_used
            )

            # --- Update config_history directly for this fold ---
            config_history[f'fold_{j}_history'] = updated_fold_history

            # Extract the final best val loss and best epoch for averaging
            fold_best_val_losses.append(updated_fold_history[5]) # Index 5 is best_val_loss
            fold_best_epochs.append(updated_fold_history[3]) # Index 3 is best_epoch

            # --- Optional: Clean up fold-specific resources ---
            del model, optimizer
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            elif DEVICE.type == 'mps': torch.mps.empty_cache()

        # --- Calculate and update mean validation loss in config_history ---
        config_history['mean_val_loss'] = np.mean(fold_best_val_losses)
        config_history['mean_best_epoch'] = np.mean(fold_best_epochs)
        
        # return the updated config_history
        return config_history
    
    def cross_validate(self,
                         dh: 'DataHandler',
                         param_grid: Dict[str, Any],
                         # Define the rung schedule directly
                         rung_schedule: List[Tuple[int, int]] = zip(RUNG_EPOCHS, RUNG_PATIENCE),
                         reduction_factor: int = 3, # eta
                         optimizer_choice: Type[optim.Optimizer] = OPTIMIZER_CHOICE
                        ) -> None:
        """
        Performs cross-validation using a Successive Halving Algorithm (SHA) approach.

        Manages the evaluation of multiple hyperparameter configurations over
        increasing training durations (epochs) specified in `rung_schedule`.
        It prunes lower-performing configurations progressively based on their
        validation performance across CV folds, using the specified `reduction_factor`.
        Uses `_train_one_config` to evaluate configurations at each stage, leveraging
        saved model states to avoid retraining from scratch.

        Finally, aggregates results for all configurations evaluated and saves a report
        mimicking a full grid search, indicating the performance achieved and the
        total epochs trained for each configuration before potential pruning. Sets
        `self.best_params` based on the best configuration among the survivors.

        Args:
            dh (DataHandler): The data handler instance.
            param_grid (Dict[str, Any]): The grid of hyperparameters to explore.
            rung_schedule (List[Tuple[int, int]]): A list of tuples, where each
                tuple defines a rung: (cumulative_max_epochs, cumulative_patience).
                Example: [(25, 10), (50, 15), ... , (150, 35)].
            reduction_factor (int): Factor by which to reduce the number of configs
                                   at each stage (e.g., 3 means keep top 1/3). Defaults to 3.
            optimizer_choice (Type[optim.Optimizer]): The optimizer class to use.

        Returns:
            None: This method primarily updates `self.best_params` and saves results
                  to a CSV file defined by `self.results_save_path`.
        """
        print(f"\n--- Starting SHA Cross-Validation for {self.model_name.upper()} (eta={reduction_factor}) ---")
        start_time = time.time()

        # --- 1. Initialization ---
        all_configs_list = list(ParameterGrid(param_grid))
        num_initial_configs = len(all_configs_list)
        # Map index to config for easy lookup
        config_dict = {i: config for i, config in enumerate(all_configs_list)}
        # Main history dictionary: Stores state for active configs
        # Value format will be: {'fold_0_history': tuple, ..., 'mean_val_loss': float, 'mean_best_epoch': float}
        cv_history_dict = {i: {} for i in range(num_initial_configs)} # Start with empty dicts for all

        # --- 2. Rung Iteration ---
        num_rungs = len(rung_schedule)
        for rung_idx, (rung_epochs, rung_patience) in enumerate(rung_schedule):
            num_configs_in_rung = len(cv_history_dict)
            if num_configs_in_rung == 0: # Should not happen if eta >= 1
                 print("Warning: No configurations remaining. Stopping SHA early.")
                 break

            print(f"\n>>> SHA Rung {rung_idx+1}/{num_rungs} | Target Epochs: {rung_epochs} | Patience: {rung_patience} | Evaluating {num_configs_in_rung} configs <<<")

            rung_results = {} # Temporary storage for results of this rung: {config_idx: (mean_loss, mean_epoch, updated_fold_tuples)}

            # --- Evaluate each active configuration ---
            for idx in cv_history_dict.keys():
                #print(f"  Rung {rung_idx+1} - Config {i+1}/{num_configs_in_rung} (ID: {config_idx})")

                # Call helper to train/evaluate this config for the current rung
                # Assuming _train_one_config returns: (mean_best_val_loss, mean_best_epoch, updated_fold_data_list)
                # where updated_fold_data_list is List[7-item state tuple]
                updated_config_history = self._train_one_config(
                    config=config_dict[idx],
                    # Pass the history dict directly, _train_one_config handles unpacking
                    config_history=cv_history_dict[idx],
                    dh=dh,
                    rung_epochs=rung_epochs,
                    rung_patience=rung_patience,
                    optimizer_choice=optimizer_choice
                )

                # Store the results temporarily for ranking
                rung_results[config_idx] = (mean_loss, mean_epoch, updated_fold_tuples)
                # Note: previous_history dict was updated directly inside _train_one_config

            # --- 3. Pruning Step (if not the last rung) ---
            if rung_idx < num_rungs - 1:
                n_promote = max(1, int(np.floor(num_configs_in_rung / reduction_factor)))

                # Sort configs evaluated in this rung by performance (lower loss is better)
                # Items are (config_idx, (mean_loss, mean_epoch, fold_tuples))
                sorted_rung_results = sorted(rung_results.items(), key=lambda item: item[1][0])

                # Identify the indices of the configurations to keep
                surviving_indices = {idx for idx, _ in sorted_rung_results[:n_promote]}

                # Create the history dictionary for the *next* rung
                next_cv_history_dict = {}
                for idx in surviving_indices:
                    # Retrieve the full history dict which was updated by _train_one_config
                    next_cv_history_dict[idx] = cv_history_dict[idx] # Carry over the updated history

                cv_history_dict = next_cv_history_dict # Replace current history with survivors' history
                print(f"  Rung {rung_idx+1} completed. Promoted {len(cv_history_dict)} configurations.")

            else:
                # Last rung completed
                print(f"  Final rung {rung_idx+1} completed. {len(cv_history_dict)} configurations survived.")


        # --- 4. Post-SHA Aggregation for Final Report ---
        print("\n--- Aggregating Final Results ---")
        final_results_list = []
        # Iterate through the *original* set of configs to report on all
        for config_idx, config_params in idx_to_config.items():
            if config_idx in cv_history_dict:
                # Config survived or was evaluated in the last rung
                final_history = cv_history_dict[config_idx]
                # Determine the max epochs reached for this config based on fold history
                # (Could also be inferred from the last rung it participated in)
                last_epochs_per_fold = [final_history.get(f'fold_{j}_history', (None,) * 7)[4] for j in range(num_folds)]
                # Use max epoch reached across folds, default 0 if no history
                max_epoch_reached = max(last_epochs_per_fold) if any(e is not None for e in last_epochs_per_fold) else 0

                result_entry = {
                    **config_params,
                    'mean_cv_score': final_history.get('mean_val_loss', float('inf')),
                    'mean_best_epoch': final_history.get('mean_best_epoch', 0), # Assumes _train_one_config added this
                    'epochs_trained': max_epoch_reached # Reflects actual max training epoch
                }
            else:
                # Config was pruned earlier - report based on its last available data (requires modification to store this)
                # --- Simplification: Only report survivors ---
                # If we only report survivors, we skip pruned configs.
                # To report all requires storing performance just before pruning.
                # Let's stick to reporting only the final survivors for simplicity here.
                continue # Skip configs that didn't make it to the final cv_history_dict

            # Stringify list-like hyperparameters for CSV compatibility
            if 'hidden_layers' in result_entry:
                result_entry['hidden_layers'] = str(result_entry['hidden_layers'])
            final_results_list.append(result_entry)

        # --- 5. Save Final Results ---
        results_df = pd.DataFrame(final_results_list)
        if not results_df.empty:
            results_df = results_df.sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
            results_df.to_csv(self.results_save_path, index=False)
            print(f"Final results for {len(results_df)} surviving configurations saved to: {self.results_save_path}")

            # --- 6. Store Best Params ---
            # Best params are simply the config corresponding to the top row of the saved results
            # Use the existing parser which expects specific column names
            self.best_params = self._parse_best_params_from_csv() # Assumes parser works with the output format

            print(f"\nBest {self.model_name.upper()} SHA CV params found: {self.best_params}")
            print(f"(Based on performance after {results_df.iloc[0]['epochs_trained']} max epochs)")
            print(f"Best Mean CV Score: {results_df.iloc[0]['mean_cv_score']:.6f}")
            print(f"Best Mean Epoch: {results_df.iloc[0]['mean_best_epoch']:.2f}")
        else:
            print("Warning: No configurations survived the SHA process.")
            self.best_params = {} # No best params found

        end_time = time.time()
        print(f"--- Finished SHA Cross-Validation for {self.model_name.upper()} ({end_time - start_time:.2f} seconds) ---")


    def cross_validate(self,
                       dh: DataHandler,
                       optimizer_choice: Type[optim.Optimizer] = OPTIMIZER_CHOICE,
                       max_epochs: int = MAX_CV_EPOCHS,
                       patience: int = PATIENCE
                       ) -> None:
        """Performs CV for the NN model, saves results, and stores best params."""
        print(f"\n--- Starting Cross-Validation for {self.model_name.upper()} ---")

        results_list = []
        best_mean_loss = float('inf')
        current_best_params = {}

        # Ensure the param_grid has 'hidden_layers' defined correctly for each config
        # ParameterGrid iterates through all combinations defined in self.param_grid
        for i, config in enumerate(ParameterGrid(self.param_grid), 1):
            print("--------------------------------------")
            print(f"  Testing config {i}: {config}")
            fold_val_losses = []

            # Loop over each fold's DataLoaders
            for j, (train_loader, val_loader) in enumerate(dh.cv_dataloaders, 1):
                # Build a new model instance for this fold using the current config
                model_fold = self._build_network(config, dh.input_dim).to(DEVICE)

                # Create optimizer for this fold based on the current config
                current_optimizer = optimizer_choice(
                    model_fold.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config.get('weight_decay', 0) # Handle optional weight decay
                )

                # Train one fold and get the best validation loss achieved
                val_loss, train_loss, best_epoch = self._train_one_fold(
                    model_fold,
                    train_loader,
                    val_loader,
                    current_optimizer,
                    max_epochs,
                    patience
                )
                fold_val_losses.append(val_loss)

                # Display fold results
                fold_str = f" Fold {j} - Train Years: {dh.folds[j-1][0]} - Val Year: {dh.folds[j-1][1]} - "
                epochs_str = f"Best epoch: {best_epoch} - "
                results_str = f"Val Loss: {val_loss:.6f} - Train Loss: {train_loss:.6f}"
                print(fold_str + epochs_str + results_str)

                # Clean up memory
                del model_fold, current_optimizer
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                elif DEVICE.type == 'mps': torch.mps.empty_cache()

            # Compute mean validation loss across all folds for this config
            mean_val_loss = np.mean(fold_val_losses)
            print(f"    Avg Val Score (Weighted CE): {mean_val_loss:.6f}")
            results_list.append({**config, 'mean_cv_score': mean_val_loss})

            # Update best score and params if this config is better
            if mean_val_loss < best_mean_loss:
                best_mean_loss = mean_val_loss
                current_best_params = config

        self.best_params = current_best_params # Store best params found
        results_df = pd.DataFrame(results_list)
        # Ensure 'hidden_layers' column is handled correctly if it exists
        if 'hidden_layers' in results_df.columns:
           results_df['hidden_layers'] = results_df['hidden_layers'].astype(str) # Save lists as strings
        results_df = results_df.sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(self.results_save_path, index=False)
        print("--------------------------------------")
        print(f"CV results saved to: {self.results_save_path}")

        print(f"\nBest {self.model_name.upper()} CV params: {self.best_params} (Score: {best_mean_loss:.6f})")
        print(f"--- Finished Cross-Validation for {self.model_name.upper()} ---")

    def _parse_best_params_from_csv(self) -> Dict:
        """Reads the best parameters from the CV results CSV file."""
        results_df = pd.read_csv(self.results_save_path)
        best_params_series = results_df.iloc[0]
        parsed_params = {}
        for key, value in best_params_series.items():
            if key == 'hidden_layers':
                # Safely evaluate the string representation of the list
                try:
                    parsed_params[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                     # Handle cases where it might be empty or not a list string
                     # Or if the key doesn't exist / wasn't saved correctly
                     # Defaulting to empty list if parsing fails
                     print(f"Warning: Could not parse 'hidden_layers' value '{value}'. Defaulting to [].")
                     parsed_params[key] = []
            elif key == 'mean_cv_score':
                 parsed_params[key] = pd.to_numeric(value) # Keep score as float
            else:
                # Attempt to convert other params to numeric, fallback to original if error
                try:
                    num_val = pd.to_numeric(value)
                    # Check if it's an integer that was saved as float (e.g., n_hidden)
                    if np.issubdtype(type(num_val), np.integer) or num_val == int(num_val):
                        parsed_params[key] = int(num_val)
                    else:
                        parsed_params[key] = num_val
                except ValueError:
                    parsed_params[key] = value # Keep as string if not numeric
        return parsed_params

    def train_final_model(self,
                          dh: DataHandler,
                          final_train_epochs: int = FINAL_TRAIN_EPOCHS,
                          optimizer_choice: Type[optim.Optimizer] = OPTIMIZER_CHOICE,
                          final_patience: int = 50 # Patience for final training early stopping
                          ) -> nn.Module:
        """Trains the final NN model using the best hyperparams from CV."""
        print(f"\n--- Starting Final Model Training for {self.model_name.upper()} ---")

        # 1. Read best hyperparameters from CV results if not already stored
        if not self.best_params:
            self.best_params = self._parse_best_params_from_csv()
        print(f"Using best hyperparameters from CV: {self.best_params}")

        # 2. Build model instance using the best parameters
        self.model = self._build_network(self.best_params, dh.input_dim).to(DEVICE)

        # 3. Create final DataLoader
        final_train_loader, _ = dh.final_dataloaders # We only need train loader here

        # 4. Setup Optimizer using best parameters
        lr = self.best_params['learning_rate']
        wd = self.best_params.get('weight_decay', 0) # Use 0 if not found
        optimizer = optimizer_choice(self.model.parameters(), lr=lr, weight_decay=wd)

        # 5. Final Training Loop with Early Stopping
        best_loss = float('inf')
        epochs_no_improve = 0
        best_state_dict = None # To store the best model state

        self.model.train() # Set model to training mode
        self.final_loss_history = []
        print(f"Starting final training for up to {final_train_epochs} epochs (Patience: {final_patience})...")

        for epoch in range(final_train_epochs):
            epoch_loss_sum = 0.0
            for features, targets_batch, weights in final_train_loader:
                features, targets_batch, weights = features.to(DEVICE), targets_batch.to(DEVICE), weights.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.weighted_cross_entropy_loss(outputs, targets_batch, weights)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item()

            avg_epoch_loss = epoch_loss_sum / len(final_train_loader) # Average loss over total weight
            self.final_loss_history.append({'epoch': epoch + 1, 'loss': avg_epoch_loss})

            # Check for improvement based on average epoch loss
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_no_improve = 0
                best_state_dict = copy.deepcopy(self.model.state_dict()) # Save best state
            else:
                epochs_no_improve += 1

            # Log progress periodically
            if (epoch + 1) % 10 == 0 or epoch == final_train_epochs - 1 or epochs_no_improve >= final_patience:
                print(f"  Epoch {epoch+1}/{final_train_epochs} - Loss: {avg_epoch_loss:.6f} "
                      f"(Best Loss: {best_loss:.6f}, Epochs No Improve: {epochs_no_improve})")

            # Early stopping check
            if epochs_no_improve >= final_patience:
                print(f"  Early stopping triggered at epoch {epoch+1}.")
                break

        # 6. Load the best performing model state before saving
        self.model.load_state_dict(best_state_dict)
        print(f"Loaded model state from epoch with best loss: {best_loss:.6f}")

        # 7. Save the best model state_dict and the loss history
        torch.save(self.model.state_dict(), self.state_dict_save_path)
        print(f"Saved best model state_dict to: {self.state_dict_save_path}")

        loss_df = pd.DataFrame(self.final_loss_history)
        loss_df.to_csv(self.loss_save_path, index=False)
        print(f"Saved training loss history to: {self.loss_save_path}")
        print(f"--- Finished Final Model Training for {self.model_name.upper()} ---")

        self.model.eval() # Set model to evaluation mode after training
        return self.model

    def load_model(self, input_dim: int) -> nn.Module:
        """Loads a trained NN model state_dict, using hyperparameters from CV results."""
        print(f"\n--- Loading Final Model for {self.model_name.upper()} ---")
        print(f"Loading state_dict from: {self.state_dict_save_path}")
        print(f"Reading hyperparameters from: {self.results_save_path}")

        # 1. Read best hyperparameters from CV results CSV to know model structure
        self.best_params = self._parse_best_params_from_csv()
        print(f"Building model architecture with best params: {self.best_params}")

        # 2. Build model architecture using subclass method and best params
        self.model = self._build_network(self.best_params, input_dim).to(DEVICE) # Build on specified device

        # 3. Load the saved state dictionary onto the same device
        self.model.load_state_dict(torch.load(self.state_dict_save_path, map_location=DEVICE))
        self.model.eval() # Set to evaluation mode
        print(f"{self.model_name.upper()} model loaded successfully onto {DEVICE}.")
        print(f"--- Finished Loading Model for {self.model_name.upper()} ---")
        return self.model

    def predict(self, dh: DataHandler, save: bool = False) -> np.ndarray:
        """
        Generates raw predictions for the test set using the trained NN model.

        Args:
            dh (DataHandler): The data handler instance containing test data.
            save (bool): If True, save predictions to a CSV file. Defaults to False.

        Returns:
            np.ndarray: The model's predictions as a NumPy array.
        """
        print(f"\n--- Generating Predictions for {self.model_name.upper()} on Year {dh.test_year} ---")

        # 2. Get test features from DataHandler
        # We only need X for prediction; load directly as tensor
        _, (X_test_pd, y_test, _) = dh.final_data
        X_test_tensor = torch.tensor(X_test_pd.values, dtype=torch.float32)
        y_tots_tensor = torch.tensor(y_test, dtype=torch.float32).sum(dim=1, keepdim=True).to(DEVICE)

        # 3. Ensure model is in eval mode and on the correct device
        self.model.eval()
        self.model.to(DEVICE) # Ensure model is on the global DEVICE

        # 4. Perform prediction
        with torch.no_grad():
            outputs = self.model(X_test_tensor) # Raw probabilities [n_samples, 4] on DEVICE
            y_pred = outputs * y_tots_tensor # Scaled predictions [n_samples, 4] on DEVICE

        # 5. Move predictions to CPU and convert to NumPy array
        y_pred = y_pred.cpu().numpy()

        # 6. Save predictions if requested
        if save:
            pred_df = pd.DataFrame(y_pred, columns=targets) # Use global 'targets'
            pred_save_path = os.path.join(PREDS_DIR, f"{dh.test_year}_{self.model_name}_predictions.csv")
            pred_df.to_csv(pred_save_path, index=False)
            print(f"County-level raw predictions saved to: {pred_save_path}")
        return y_pred

# =============================================================================
# 5. XGBoostModel Class
# =============================================================================

class XGBoostModel:
    """
    Handles XGBoost Regression CV, training, loading, and prediction.
    Treats each target variable as an independent regression task.
    Uses weighted RMSE for CV evaluation and determines optimal boosting rounds
    via early stopping during CV.
    """
    MODEL_NAME = "xgboost"

    def __init__(self, model_name: str = "xgboost", param_grid: Dict = XGB_PARAM_GRID):
        """
        Initializes the XGBoostModel handler.

        Args:
            model_name (str): Name for this model instance (default: "xgboost").
            param_grid (Dict): Hyperparameter grid for CV.
        """
        self.model_name: str = model_name
        self.param_grid: Dict = param_grid
        self.model: Union[xgb.XGBRegressor, None] = None
        self.best_params: Dict = {} # Stores best params found by CV
        self.optimal_cv_estimators: Union[int, None] = None # Stores avg best iteration from CV

        # Define file paths based on model_name
        self.results_save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_cv_results.csv")
        self.model_save_path = os.path.join(MODELS_DIR, f"{self.model_name}_final_model.json") # Save as JSON
        print(f"XGBoostModel initialized for '{self.model_name}'. CV results: {self.results_save_path}")

    def _parse_best_params_from_csv(self) -> Tuple[Dict, int]:
        """Reads the best parameters and optimal estimators from the CV results CSV."""
        results_df = pd.read_csv(self.results_save_path)
        best_params_series = results_df.iloc[0]
        parsed_params = {}
        optimal_estimators = None

        for key, value in best_params_series.items():
            if key == 'avg_best_iteration':
                optimal_estimators = int(round(pd.to_numeric(value))) # Convert to int
            elif key == 'mean_cv_score':
                # This key holds the score, not a model parameter
                continue
            else:
                # Attempt to convert params to numeric, handle lists/strings if needed
                try:
                    # Handle cases where grid values might be single-item lists
                    evaluated = ast.literal_eval(str(value))
                    if isinstance(evaluated, list) and len(evaluated) == 1:
                       parsed_params[key] = evaluated[0]
                    # Check for numeric types after potential list unpacking
                    elif isinstance(evaluated, (int, float)):
                        if evaluated == int(evaluated):
                           parsed_params[key] = int(evaluated)
                        else:
                           parsed_params[key] = float(evaluated)
                    else:
                         parsed_params[key] = evaluated # Keep as evaluated type (e.g., string)
                except (ValueError, SyntaxError):
                     # Fallback for simple values or strings that aren't list representations
                    try:
                       num_val = pd.to_numeric(value)
                       if num_val == int(num_val):
                           parsed_params[key] = int(num_val)
                       else:
                           parsed_params[key] = num_val
                    except ValueError:
                         parsed_params[key] = value # Keep as original string

        return parsed_params, optimal_estimators

    def cross_validate(self,
                       dh: DataHandler,
                       early_stopping_rounds: int = XGB_EARLY_STOPPING_ROUNDS
                       ) -> None:
        """
        Performs Cross-Validation for XGBoost using early stopping.

        Finds the best hyperparameters based on minimizing the average weighted
        RMSE across folds and determines the average optimal number of boosting
        rounds for those parameters. Saves results to CSV.

        Args:
            dh (DataHandler): The data handler instance.
            early_stopping_rounds (int): Number of rounds for early stopping in CV.
        """
        print(f"\n--- Starting Cross-Validation for {self.model_name.upper()} ---")
        results_list = []
        best_mean_score = float('inf')
        current_best_params = {}
        current_best_mean_iteration = None

        # Iterate through hyperparameter combinations
        for i, config in enumerate(ParameterGrid(self.param_grid), 1):
            print("--------------------------------------")
            # Ensure 'n_estimators' is removed if present, as early stopping controls it
            actual_config = {k: v for k, v in config.items() if k != 'n_estimators'}
            # Keep track of n_estimators from grid if needed, otherwise set high
            n_estimators_cv = config.get('n_estimators', 500) # Use value from grid or default high

            print(f"  Testing config {i}: {actual_config} (n_est={n_estimators_cv}, early_stop={early_stopping_rounds})")
            fold_val_scores = []
            fold_best_iterations = []

            # Loop over CV folds provided by DataHandler
            # dh.cv_data provides tuples: ((X_train, y_train, wts_train), (X_val, y_val, wts_val))
            # Data is expected as NumPy arrays here.
            for j, ((X_train, y_train, wts_train), (X_val, y_val, wts_val)) in enumerate(dh.cv_data, 1):

                # Instantiate XGBoost Regressor for this fold
                model_fold = xgb.XGBRegressor(**actual_config,
                                              n_estimators=n_estimators_cv, # Start with potentially high value
                                              early_stopping_rounds=early_stopping_rounds)

                # Prepare evaluation set for early stopping
                eval_set = [(X_val, y_val)]
                eval_weights = [wts_val.ravel()] # Must be list of weights

                # Fit model with early stopping based on validation set performance
                model_fold.fit(X_train, y_train,
                               sample_weight=wts_train.ravel(), # Weights for training
                               eval_set=eval_set,
                               sample_weight_eval_set=eval_weights, # Weights for validation loss
                               verbose=False) # Suppress verbose output during CV

                # Store the best score (weighted RMSE) and iteration achieved
                fold_val_scores.append(model_fold.best_score)
                fold_best_iterations.append(model_fold.best_iteration)

                fold_str = f" Fold {j} - Train Years: {dh.folds[j-1][0]} - Val Year: {dh.folds[j-1][1]} - "
                results_str = f"Best Score (W-RMSE): {model_fold.best_score:.6f} at iteration {model_fold.best_iteration}"
                print(fold_str + results_str)

                del model_fold # Clean up fold model

            # Calculate mean validation score and mean best iteration for this config
            mean_val_score = np.mean(fold_val_scores)
            mean_best_iteration = np.mean(fold_best_iterations)
            print(f"    Avg Val Score (Weighted RMSE): {mean_val_score:.6f}")
            print(f"    Avg Best Iteration: {mean_best_iteration:.2f}")

            # Store results for this configuration
            result_entry = {**config, # Log the original config from grid
                            'mean_cv_score': mean_val_score,
                            'avg_best_iteration': mean_best_iteration}
            results_list.append(result_entry)

            # Update overall best score, params, and iteration if this config is better
            if mean_val_score < best_mean_score:
                best_mean_score = mean_val_score
                current_best_params = config # Store the config from the grid
                current_best_mean_iteration = int(round(mean_best_iteration))

        # Store the best parameters and optimal estimator count found
        self.best_params = current_best_params
        self.optimal_cv_estimators = current_best_mean_iteration

        # Save results DataFrame to CSV
        results_df = pd.DataFrame(results_list)
        # Convert list-like columns (potentially from param grid) to strings for CSV compatibility
        for col in results_df.columns:
           if results_df[col].apply(lambda x: isinstance(x, list)).any():
               results_df[col] = results_df[col].astype(str)
        results_df = results_df.sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(self.results_save_path, index=False)
        print("--------------------------------------")
        print(f"CV results saved to: {self.results_save_path}")

        print(f"\nBest {self.model_name.upper()} CV params: {self.best_params}")
        print(f"Best Average Iteration (used for final n_estimators): {self.optimal_cv_estimators}")
        print(f"Best CV Score (Weighted RMSE): {best_mean_score:.6f}")
        print(f"--- Finished Cross-Validation for {self.model_name.upper()} ---")

    def train_final_model(self,
                          dh: DataHandler
                          ) -> xgb.XGBRegressor:
        """
        Trains the final XGBoost model using the best hyperparameters and
        the optimal number of estimators determined by CV.

        Args:
            dh (DataHandler): The data handler instance.

        Returns:
            xgboost.XGBRegressor: The trained final model.
        """
        print(f"\n--- Starting Final Model Training for {self.model_name.upper()} ---")

        # 1. Load best hyperparameters and optimal estimator count from CV results
        if not self.best_params or self.optimal_cv_estimators is None:
             loaded_params, loaded_estimators = self._parse_best_params_from_csv()
             self.best_params = loaded_params
             self.optimal_cv_estimators = loaded_estimators
        print(f"Using best hyperparameters from CV: {self.best_params}")
        print(f"Using optimal n_estimators from CV: {self.optimal_cv_estimators}")

        # Remove keys not directly usable by XGBRegressor init (like grid-search helpers)
        final_params = {k: v for k, v in self.best_params.items() if k not in ['n_estimators']}


        # 2. Instantiate the final model with best params and optimal n_estimators
        self.model = xgb.XGBRegressor(**final_params,
                                      n_estimators=self.optimal_cv_estimators) # Use specific estimator count

        # 3. Load the full final training dataset
        # dh.final_data provides ((X_train, y_train, wts_train), (X_test, y_test, wts_test))
        (X_train, y_train, wts_train), _ = dh.final_data
        print(f"Loaded final training data: {X_train.shape[0]} samples.")

        # 4. Fit the final model on the entire training set
        # No early stopping here - we use the n_estimators determined from CV.
        print(f"Fitting final model with {self.optimal_cv_estimators} estimators...")
        self.model.fit(X_train, y_train,
                       sample_weight=wts_train.ravel(),
                       verbose=False) # Keep fitting non-verbose
        print("Final model fitting complete.")

        # 5. Save the trained model to JSON format
        self.model.save_model(self.model_save_path)
        print(f"Saved final trained XGBoost model to: {self.model_save_path}")
        print(f"--- Finished Final Model Training for {self.model_name.upper()} ---")
        return self.model

    def load_model(self) -> xgb.XGBRegressor:
        """Loads a trained XGBoost model from its JSON file."""
        print(f"\n--- Loading Final Model for {self.model_name.upper()} ---")
        print(f"Loading model state from: {self.model_save_path}")

        # Instantiate a placeholder model
        self.model = xgb.XGBRegressor()
        # Load the saved state into the model object
        self.model.load_model(self.model_save_path)

        print(f"{self.model_name.upper()} model loaded successfully from JSON.")
        print(f"--- Finished Loading Model for {self.model_name.upper()} ---")
        return self.model

    def predict(self, dh: DataHandler, save: bool = False) -> np.ndarray:
        """
        Generates raw predictions for the test set using the trained XGBoost model.

        Note: Predictions are raw regression outputs per target and may not sum to 1
              or be constrained between 0 and 1.

        Args:
            dh (DataHandler): The data handler instance containing test data.
            save (bool): If True, save predictions to a CSV file. Defaults to False.

        Returns:
            np.ndarray: The model's predictions as a NumPy array [n_samples, n_targets].
        """
        print(f"\n--- Generating Predictions for {self.model_name.upper()} on Year {dh.test_year} ---")
        # 1. Load model if it hasn't been loaded or trained in this session
        if self.model is None:
            self.load_model()

        # 2. Get test features from DataHandler
        # dh.final_data provides ((X_train, y_train, wts_train), (X_test, y_test, wts_test))
        _, (X_test, _, _) = dh.final_data # Only need X_test (NumPy array)
        print(f"Loaded test features: {X_test.shape[0]} samples.")

        # 3. Perform prediction
        print("Performing inference...")
        y_pred = self.model.predict(X_test)
        print("Inference complete.")

        # 4. Save predictions if requested
        if save:
            pred_df = pd.DataFrame(y_pred, columns=targets) # Use global 'targets'
            pred_save_path = os.path.join(PREDS_DIR, f"{dh.test_year}_{self.model_name}_predictions.csv")
            pred_df.to_csv(pred_save_path, index=False)
            print(f"County-level raw predictions saved to: {pred_save_path}")

        print(f"--- Finished Predictions for {self.model_name.upper()} ---")
        return y_pred
