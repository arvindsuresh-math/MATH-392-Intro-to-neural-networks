# =============================================================================
# Presidential Election Outcome Prediction - Refactored Midterm Project Code
#
# High-Level Overview:
# This script implements a framework for predicting US presidential election
# outcomes (Democrat, Republican, Other, Non-voter probabilities) at the
# county level using demographic data. It compares four model types:
# Ridge Regression, Softmax Regression, 1-Layer MLP, and 2-Layer MLP.
#
# Code Structure:
# 1. Imports & Global Constants: Sets up libraries, configuration parameters
#    (file paths, hyperparameters, training settings), and device.
# 2. Weighted Loss Function: Defines the custom loss metric used for NN models.
# 3. DataHandler Class: Encapsulates all data loading and DataLoader creation.
#    Assumes data CSVs are in a './data/' subdirectory.
# 4. BaseNNModel Class: An abstract base class for PyTorch neural network models.
#    It implements the common logic for cross-validation (`cross_validate`),
#    final model training (`train_final_model`), model loading (`load_model`),
#    and evaluation (`evaluate`). Subclasses must implement `_build_network`
#    and define `MODEL_NAME` and `PARAM_GRID`.
# 5. NN Model Classes (SoftmaxModel, MLP1Model, MLP2Model):
#    - Inherit from `BaseNNModel`.
#    - Define their specific network architecture internally (_Net classes).
#    - Implement `_build_network` to instantiate their specific architecture
#      based on hyperparameters.
#    - Define `MODEL_NAME` and `PARAM_GRID`.
# 6. RidgeModel Class: Manages Ridge Regression separately due to its
#    different API (scikit-learn). Provides similar `cross_validate`,
#    `train_final_model`, `load_model`, and `evaluate` methods.
#
# Workflow (Example Usage in a Jupyter Notebook):
# a. Import this script (e.g., `import election_project as ep`).
# b. Initialize `ep.DataHandler()`.
# c. Initialize model handler instances (e.g., `ridge = ep.RidgeModel()`,
#    `mlp1 = ep.MLP1Model()`).
# d. To find best hyperparameters and train:
#    - `model.cross_validate(data_handler, ...)` - Finds best hypers, saves CSV.
#    - `trained_model = model.train_final_model(data_handler, ...)` - Reads
#      best hypers from CSV, trains on combined data, saves model artifact,
#      returns trained model instance.
# e. To load a previously trained model:
#    - `loaded_model = model.load_model(...)` - Reads best hypers from CSV,
#      builds architecture, loads saved artifact, returns model instance.
# f. Evaluate the final model (obtained from d or e):
#    - Create a test DataLoader using `DataHandler`.
#    - `metrics = model.evaluate(test_loader)` - Computes metrics on test set.
#
# File Locations:
# - Input Data: Assumed to be in './data/' (e.g., './data/X_2008.csv').
# - Outputs (CV results, models, loss histories): Saved to './models/'.
#   This directory will be created if it doesn't exist.
# =============================================================================

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import joblib
from typing import List, Dict, Tuple, Any, Type, Union
from abc import ABC, abstractmethod

# =============================================================================
# 1. Global Constants and Configuration
# =============================================================================

# --- File Paths ---
DATA_DIR = "./data"
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True) # Create models dir if it doesn't exist

# --- Cross-Validation Fold Definitions ---
FOLD_DEFINITIONS: List[Dict[str, str]] = [
    {'train_suffix': '2008_2012', 'val_suffix': '2016'},
    {'train_suffix': '2008_2016', 'val_suffix': '2012'},
    {'train_suffix': '2012_2016', 'val_suffix': '2008'},
]
COMBINED_DATA_SUFFIX = "2008_2012_2016" # Suffix for final training data

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
BATCH_SIZE: int = 50
MAX_CV_EPOCHS: int = 300 # Max epochs for early stopping during CV
PATIENCE: int = 15      # Patience for early stopping during CV
FINAL_TRAIN_EPOCHS: int = 100 # Fixed epochs for final training
OPTIMIZER_CHOICE: Type[optim.Optimizer] = optim.AdamW # Default optimizer

# --- Default Hyperparameter Grids for CV ---
RIDGE_PARAM_GRID = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
SOFTMAX_PARAM_GRID = {
    'learning_rate': [1e-2, 1e-3, 1e-4],
    'weight_decay': [0, 1e-5, 1e-4]
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

# =============================================================================
# 3. DataHandler Class
# =============================================================================

class DataHandler:
    """Handles loading data and creating PyTorch DataLoaders."""

    def __init__(self, sample_suffix: str = "2008"):
        """
        Initializes the DataHandler and determines input/output dimensions.

        Args:
            sample_suffix (str): Suffix of a sample file (e.g., '2008') used
                                 to determine data dimensions. Assumed to be in DATA_DIR.
        """
        self.data_dir = DATA_DIR
        # Determine dimensions from a sample file
        X_sample, y_sample, _ = self.load_raw_data(sample_suffix)
        self.input_dim = X_sample.shape[1]
        self.output_dim = y_sample.shape[1]
        print(f"DataHandler initialized: Input Dim={self.input_dim}, Output Dim={self.output_dim}")

    def get_dimensions(self) -> Tuple[int, int]:
        """Returns the input and output dimensions."""
        return self.input_dim, self.output_dim

    def load_raw_data(self, suffix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads features (X), targets (y), and weights (wts) CSV files from DATA_DIR."""
        x_path = os.path.join(self.data_dir, f'X_{suffix}.csv')
        y_path = os.path.join(self.data_dir, f'y_{suffix}.csv')
        wts_path = os.path.join(self.data_dir, f'wts_{suffix}.csv')

        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        wts = pd.read_csv(wts_path)
        # Ensure wts only contains the 'P(C)' column needed
        wts = wts[['P(C)']]
        return X, y, wts

    def _create_dataset(self, X_df: pd.DataFrame, y_df: pd.DataFrame, wts_df: pd.DataFrame) -> TensorDataset:
        """Converts pandas DataFrames to a PyTorch TensorDataset."""
        X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_df.values, dtype=torch.float32)
        wts_tensor = torch.tensor(wts_df.values, dtype=torch.float32) # Already [:, 1] shape
        return TensorDataset(X_tensor, y_tensor, wts_tensor)

    def create_dataloaders(self, train_suffix: str, val_suffix: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Creates DataLoaders for training and validation sets."""
        X_train, y_train, wts_train = self.load_raw_data(train_suffix)
        X_val, y_val, wts_val = self.load_raw_data(val_suffix)

        train_dataset = self._create_dataset(X_train, y_train, wts_train)
        val_dataset = self._create_dataset(X_val, y_val, wts_val)

        # Use persistent_workers and pin_memory for potentially faster loading if CUDA available
        # num_workers > 0 can speed things up but sometimes causes issues on Windows/macOS
        num_workers = 2 if DEVICE.type == 'cuda' else 0 # Adjust as needed
        pin_memory = True if DEVICE.type == 'cuda' else False

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)

        print(f"Created DataLoaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        return train_loader, val_loader

    def create_dataloader(self, data_suffix: str, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Creates a DataLoader for a specified dataset suffix."""
        X_pd, y_pd, wts_pd = self.load_raw_data(data_suffix)
        dataset = self._create_dataset(X_pd, y_pd, wts_pd)

        num_workers = 2 if DEVICE.type == 'cuda' else 0
        pin_memory = True if DEVICE.type == 'cuda' else False

        if shuffle:
            torch.manual_seed(42) # For reproducible shuffling if needed

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=num_workers, pin_memory=pin_memory)
        print(f"Created DataLoader for '{data_suffix}': {len(data_loader)} batches")
        return data_loader

# =============================================================================
# 4. RidgeModel Class (Uses scikit-learn)
# =============================================================================

class RidgeModel:
    """Handles Ridge Regression CV, training, loading, and evaluation."""
    MODEL_NAME = "ridge"
    # Note: Ridge uses Mean Squared Error for CV, not weighted cross-entropy.
    # The evaluate method will also report MSE, but can calculate weighted CE post-hoc.

    def __init__(self):
        """Initializes the RidgeModel handler."""
        self.model: Union[Ridge, None] = None
        self.best_alpha: Union[float, None] = None
        # Store results path for convenience
        self.results_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_cv_results.csv")
        self.model_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_final_model.joblib")

    def cross_validate(self,
                       data_handler: DataHandler,
                       fold_definitions: List[Dict[str, str]],
                       param_grid: List[float]
                       ) -> None:
        """Performs CV for Ridge, saves results, and stores the best alpha."""
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        results_list = []
        best_score = float('inf')
        current_best_alpha = None

        for alpha in param_grid:
            print(f"  Testing alpha = {alpha}:")
            fold_validation_scores = []
            for fold in fold_definitions:
                X_train_pd, y_train_pd, _ = data_handler.load_raw_data(fold['train_suffix'])
                X_val_pd, y_val_pd, wts_val_pd = data_handler.load_raw_data(fold['val_suffix'])

                model_fold = Ridge(alpha=alpha)
                model_fold.fit(X_train_pd.values, y_train_pd.values)
                y_pred_val = model_fold.predict(X_val_pd.values)

                # Use weighted MSE for validation score to be somewhat comparable
                # Weights are P(C), which is the 3rd element from load_raw_data
                weights_val = wts_val_pd.values.squeeze() # Ensure weights is 1D array
                validation_score = mean_squared_error(y_val_pd.values, y_pred_val, sample_weight=weights_val)
                # validation_score = mean_squared_error(y_val_pd.values, y_pred_val) # Original unweighted MSE

                fold_validation_scores.append(validation_score)

            avg_validation_score = np.mean(fold_validation_scores)
            print(f"    Avg Val Score (Weighted MSE): {avg_validation_score:.6f}")
            results_list.append({'alpha': alpha, 'mean_cv_score': avg_validation_score})
            if avg_validation_score < best_score:
                best_score = avg_validation_score
                current_best_alpha = alpha

        self.best_alpha = current_best_alpha # Store best alpha found
        results_df = pd.DataFrame(results_list).sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(self.results_save_path, index=False)

        if self.best_alpha is not None:
             print(f"\nBest {self.MODEL_NAME.upper()} CV alpha: {self.best_alpha} (Score: {best_score:.6f})")
        else:
             print(f"\nWarning: Could not determine best alpha for {self.MODEL_NAME.upper()}.")
        print(f"CV results saved to: {self.results_save_path}")
        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")

    def train_final_model(self,
                          data_handler: DataHandler,
                          train_data_suffix: str = COMBINED_DATA_SUFFIX
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
        X_train_pd, y_train_pd, _ = data_handler.load_raw_data(train_data_suffix)
        print(f"Loaded final training data: {X_train_pd.shape[0]} samples.")

        # 4. Fit the model
        print("Fitting final Ridge model...")
        self.model.fit(X_train_pd.values, y_train_pd.values)
        print("Ridge model fitting complete.")

        # 5. Save the trained model
        joblib.dump(self.model, self.model_save_path)
        print(f"Saved final trained Ridge model to: {self.model_save_path}")
        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        return self.model

    def load_model(self) -> Ridge:
        """Loads a trained Ridge model."""
        print(f"Loading trained Ridge model from: {self.model_save_path}")
        self.model = joblib.load(self.model_save_path)
        # Optionally load best_alpha too, though not strictly needed after loading model object
        # results_df = pd.read_csv(self.results_save_path)
        # self.best_alpha = results_df.iloc[0]['alpha']
        print("Ridge model loaded successfully.")
        return self.model

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluates the trained Ridge model on a given DataLoader."""
        print(f"Evaluating {self.MODEL_NAME.upper()}...")
        if self.model is None:
            print("Error: Model not trained or loaded. Cannot evaluate.")
            return {}

        all_features = []
        all_targets = []
        all_weights = []
        # We need to reconstruct the full dataset from the loader for sklearn
        for features, targets, weights in data_loader:
            all_features.append(features.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_weights.append(weights.cpu().numpy())

        X_test = np.concatenate(all_features, axis=0)
        y_test = np.concatenate(all_targets, axis=0)
        wts_test = np.concatenate(all_weights, axis=0).squeeze() # Ensure 1D

        y_pred = self.model.predict(X_test)

        # Calculate Weighted MSE (primary metric for Ridge in this setup)
        weighted_mse = mean_squared_error(y_test, y_pred, sample_weight=wts_test)
        unweighted_mse = mean_squared_error(y_test, y_pred)

        # Calculate weighted cross-entropy post-hoc for comparison
        # Note: Ridge doesn't guarantee outputs sum to 1 or are positive.
        # Apply softmax to normalize outputs before calculating CE.
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        wts_test_tensor = torch.tensor(wts_test, dtype=torch.float32).unsqueeze(1) # Add dim for loss func

        # Apply softmax to make predictions resemble probabilities
        y_pred_probs = torch.softmax(y_pred_tensor, dim=1)

        weighted_ce = weighted_cross_entropy_loss(y_pred_probs, y_test_tensor, wts_test_tensor).item()

        metrics = {
            'weighted_mse': weighted_mse,
            'unweighted_mse': unweighted_mse,
            'weighted_cross_entropy_approx': weighted_ce # Approximate CE after softmax
        }
        print(f"Evaluation complete. Metrics: {metrics}")
        return metrics


# =============================================================================
# 5. BaseNNModel Class (Abstract Base Class for PyTorch Models)
# =============================================================================

class BaseNNModel(ABC):
    """Abstract Base Class for PyTorch NN model training and evaluation."""
    MODEL_NAME: str = "base_nn"
    PARAM_GRID: Dict = {} # Must be defined by subclasses

    def __init__(self):
        """Initializes the BaseNNModel handler."""
        self.model: Union[nn.Module, None] = None
        self.best_params: Dict = {}
        self.final_loss_history: List[Dict[str, Any]] = []
        # Define paths based on MODEL_NAME (which subclasses will set)
        self.results_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_cv_results.csv")
        self.state_dict_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_final_state_dict.pth")
        self.loss_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_final_training_loss.csv")

    @staticmethod
    def weighted_cross_entropy_loss(outputs: torch.Tensor,
                                    targets: torch.Tensor,
                                    weights: torch.Tensor) -> torch.Tensor:
        """
        Calculates a custom weighted cross-entropy loss for a batch.
        Loss(C) = - sum_k ( target_k * log(output_k) )
        Expected Loss = sum_C ( P(C) * Loss(C) ) / sum_C ( P(C) )

        Args:
            outputs (torch.Tensor): Model predictions (probabilities), shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth probabilities, shape [batch_size, num_classes].
            weights (torch.Tensor): Sample weights ('P(C)'), shape [batch_size, 1].

        Returns:
            torch.Tensor: Scalar tensor representing the weighted average loss.
        """
        epsilon = 1e-9
        # Clamp outputs to avoid log(0)
        outputs_clamped = torch.clamp(outputs, epsilon, 1. - epsilon)
        # Calculate Cross-Entropy per sample: - sum(target * log(pred)) over classes
        sample_ce_loss = -torch.sum(targets * torch.log(outputs_clamped), dim=1, keepdim=True)
        # Ensure weights tensor has the same shape as sample_ce_loss for broadcasting
        weights_reshaped = weights.view_as(sample_ce_loss)
        # Apply weights: weight * sample_loss
        weighted_sample_losses = sample_ce_loss * weights_reshaped
        # Calculate the sum of weights for normalization
        total_weight = weights_reshaped.sum()
        # Calculate the mean weighted loss. Handle division by zero if total weight is zero.
        mean_weighted_loss = weighted_sample_losses.sum() / total_weight if total_weight > 0 else torch.tensor(0.0, device=outputs.device, requires_grad=True)
        return mean_weighted_loss

    @abstractmethod
    def _build_network(self, params: Dict, input_dim: int, output_dim: int) -> nn.Module:
        """
        Subclasses must implement this method to return an initialized nn.Module
        based on the provided hyperparameters.

        Args:
            params (Dict): Dictionary containing hyperparameters (e.g., 'learning_rate', 'n_hidden').
            input_dim (int): Input dimension for the network.
            output_dim (int): Output dimension for the network.

        Returns:
            nn.Module: The instantiated PyTorch model.
        """
        pass

    def _train_one_fold(self,
                        model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        optimizer: optim.Optimizer,
                        max_epochs: int,
                        patience: int,
                        verbose: bool = False
                        ) -> float:
        """Internal helper to train model for one CV fold with early stopping."""
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
        last_avg_val_loss = float('inf') # Keep track of last validation loss

        model.to(DEVICE)

        for epoch in range(max_epochs):
            model.train()
            train_loss_epoch = 0.0
            for features, targets, weights in train_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                outputs = model(features)
                loss = BaseNNModel.weighted_cross_entropy_loss(outputs, targets, weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
            avg_train_loss = train_loss_epoch / len(train_loader)

            model.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for features, targets, weights in val_loader:
                    features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                    outputs = model(features)
                    loss = BaseNNModel.weighted_cross_entropy_loss(outputs, targets, weights)
                    val_loss_epoch += loss.item()
            avg_val_loss = val_loss_epoch / len(val_loader)
            last_avg_val_loss = avg_val_loss # Store current validation loss

            if verbose and (epoch + 1) % 50 == 0: # Print less frequently during CV
                 print(f"      Epoch {epoch+1}/{max_epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose: print(f"    Early stopping at epoch {epoch+1}. Best val loss {best_val_loss:.6f} at epoch {best_epoch}.")
                break
        else: # No break occurred (ran for max_epochs)
             if verbose: print(f"    Reached max_epochs ({max_epochs}). Best val loss {best_val_loss:.6f} at epoch {best_epoch}.")


        # Handle case where validation loss was NaN or Inf, or never improved
        if not np.isfinite(best_val_loss):
             print(f"    Warning: Best validation loss was {best_val_loss}. Returning last finite loss: {last_avg_val_loss if np.isfinite(last_avg_val_loss) else float('inf')}")
             return last_avg_val_loss if np.isfinite(last_avg_val_loss) else float('inf')

        return best_val_loss

    def cross_validate(self,
                       data_handler: DataHandler,
                       fold_definitions: List[Dict[str, str]],
                       max_epochs: int,
                       patience: int,
                       batch_size: int,
                       optimizer_choice: Type[optim.Optimizer]
                       ) -> None:
        """Performs CV for the NN model, saves results, and stores best params."""
        input_dim, output_dim = data_handler.get_dimensions()
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        results_list = []
        best_avg_score = float('inf')
        current_best_params = {}

        # Use the PARAM_GRID defined in the subclass
        for config in ParameterGrid(self.PARAM_GRID):
            print(f"  Testing config: {config}")
            fold_validation_scores = []
            for i, fold in enumerate(fold_definitions):
                print(f"    Fold {i+1}/{len(fold_definitions)}...")
                train_loader, val_loader = data_handler.create_dataloaders(fold['train_suffix'], fold['val_suffix'], batch_size)

                # Build model using subclass implementation
                model_fold = self._build_network(config, input_dim, output_dim).to(DEVICE)

                # Setup optimizer using parameters from config
                lr = config['learning_rate']
                # Handle optional weight decay
                wd = config.get('weight_decay', 0) # Default to 0 if not in config
                optimizer = optimizer_choice(model_fold.parameters(), lr=lr, weight_decay=wd)

                validation_score = self._train_one_fold(
                    model_fold, train_loader, val_loader, optimizer, max_epochs, patience, device=DEVICE
                )
                fold_validation_scores.append(validation_score)

                # Clean up memory
                del train_loader, val_loader, model_fold, optimizer
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                elif DEVICE.type == 'mps': torch.mps.empty_cache()

            # Filter out non-finite scores before calculating mean
            finite_scores = [s for s in fold_validation_scores if np.isfinite(s)]
            avg_validation_score = np.mean(finite_scores) if finite_scores else float('inf')

            print(f"    Avg Val Score (Weighted CE): {avg_validation_score:.6f}")
            results_list.append({**config, 'mean_cv_score': avg_validation_score})

            if avg_validation_score < best_avg_score:
                best_avg_score = avg_validation_score
                current_best_params = config

        self.best_params = current_best_params # Store best params found
        results_df = pd.DataFrame(results_list).sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(self.results_save_path, index=False)

        if self.best_params:
             print(f"\nBest {self.MODEL_NAME.upper()} CV params: {self.best_params} (Score: {best_avg_score:.6f})")
        else:
             print(f"\nWarning: Could not determine best params for {self.MODEL_NAME.upper()}.")
        print(f"CV results saved to: {self.results_save_path}")
        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")

    def train_final_model(self,
                          data_handler: DataHandler,
                          train_data_suffix: str,
                          final_train_epochs: int,
                          batch_size: int,
                          optimizer_choice: Type[optim.Optimizer]
                          ) -> nn.Module:
        """Trains the final NN model using the best hyperparams from CV."""
        input_dim, output_dim = data_handler.get_dimensions()
        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")

        # 1. Read best hyperparameters from CV results if not already stored
        if not self.best_params:
            results_df = pd.read_csv(self.results_save_path)
            # Need to convert potentially numeric types back from string/object
            self.best_params = results_df.iloc[0].to_dict()
            # Example conversions (adjust based on actual param types)
            for key in ['learning_rate', 'dropout_rate', 'weight_decay', 'mean_cv_score']:
                if key in self.best_params:
                    self.best_params[key] = pd.to_numeric(self.best_params[key])
            for key in ['n_hidden', 'shared_hidden_size']:
                 if key in self.best_params:
                     self.best_params[key] = int(self.best_params[key])

        print(f"Using best hyperparameters from CV: {self.best_params}")

        # 2. Build model instance using subclass implementation
        self.model = self._build_network(self.best_params, input_dim, output_dim).to(DEVICE)

        # 3. Create final DataLoader
        final_train_loader = data_handler.create_dataloader(train_data_suffix, batch_size, shuffle=True)

        # 4. Setup Optimizer
        lr = self.best_params['learning_rate']
        wd = self.best_params.get('weight_decay', 0)
        optimizer = optimizer_choice(self.model.parameters(), lr=lr, weight_decay=wd)

        # 5. Training Loop
        self.model.train()
        self.final_loss_history = []
        print(f"Training for {final_train_epochs} epochs...")

        for epoch in range(final_train_epochs):
            epoch_loss = 0.0
            for features, targets, weights in final_train_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = BaseNNModel.weighted_cross_entropy_loss(outputs, targets, weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(final_train_loader)
            self.final_loss_history.append({'epoch': epoch + 1, 'loss': avg_epoch_loss})

            if (epoch + 1) % 10 == 0 or epoch == final_train_epochs - 1:
                print(f"  Epoch {epoch+1}/{final_train_epochs} - Training Loss: {avg_epoch_loss:.6f}")


        # 6. Save state dict and loss history
        torch.save(self.model.state_dict(), self.state_dict_save_path)
        print(f"Saved final model state_dict to: {self.state_dict_save_path}")
        loss_df = pd.DataFrame(self.final_loss_history)
        loss_df.to_csv(self.loss_save_path, index=False)
        print(f"Saved final training loss history to: {self.loss_save_path}")
        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")

        self.model.eval() # Set to evaluation mode after training
        return self.model

    def load_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Loads a trained NN model state_dict, using hyperparameters from CV results."""
        print(f"Loading trained {self.MODEL_NAME} state_dict from: {self.state_dict_save_path}")

        # 1. Read best hyperparameters from CV results CSV to know model structure
        results_df = pd.read_csv(self.results_save_path)
        self.best_params = results_df.iloc[0].to_dict()
        # Convert types as needed (example shown in train_final_model)
        for key in ['learning_rate', 'dropout_rate', 'weight_decay', 'mean_cv_score']:
             if key in self.best_params: self.best_params[key] = pd.to_numeric(self.best_params[key])
        for key in ['n_hidden', 'shared_hidden_size']:
             if key in self.best_params: self.best_params[key] = int(self.best_params[key])
        print(f"Building model architecture with params: {self.best_params}")

        # 2. Build model architecture using subclass method and best params
        self.model = self._build_network(self.best_params, input_dim, output_dim).to(DEVICE)

        # 3. Load the saved state dictionary
        self.model.load_state_dict(torch.load(self.state_dict_save_path, map_location=DEVICE))
        self.model.eval() # Set to evaluation mode
        print(f"{self.MODEL_NAME} model loaded successfully.")
        return self.model

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluates the trained NN model on the test dataset."""
        print(f"Evaluating {self.MODEL_NAME.upper()}...")
        if self.model is None:
            print("Error: Model not trained or loaded. Cannot evaluate.")
            return {}

        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        with torch.no_grad():
            for features, targets, weights in test_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                outputs = self.model(features) # Should be probabilities from Softmax

                # Calculate Weighted Cross-Entropy Loss
                loss = BaseNNModel.weighted_cross_entropy_loss(outputs, targets, weights)
                total_loss += loss.item()

                # Calculate Weighted Mean Squared Error
                # Ensure weights are broadcastable for MSE calculation if needed
                # PyTorch MSE doesn't directly support sample weights, do it manually
                batch_mse = ((outputs - targets)**2).sum(dim=1, keepdim=True) # Sum over classes
                weighted_batch_mse = (batch_mse * weights).sum()
                total_batch_weight = weights.sum()

                # Accumulate weighted sum of squared errors and total weight
                total_mse += weighted_batch_mse.item()
                num_batches += 1 # Count batches for CE averaging

        avg_loss = total_loss / num_batches
        avg_mse = total_mse / len(test_loader.dataset) # Average MSE over all samples

        metrics = {
            'weighted_cross_entropy': avg_loss,
            'weighted_mse': avg_mse
        }
        print(f"Evaluation complete. Metrics: {metrics}")
        return metrics


# =============================================================================
# 6. SoftmaxModel Class
# =============================================================================

class SoftmaxModel(BaseNNModel):
    """Handles Softmax Regression (as NN) CV, training, loading, evaluation."""
    MODEL_NAME = "softmax"
    PARAM_GRID = SOFTMAX_PARAM_GRID # Use global constant

    class _SoftmaxNet(nn.Module):
        """Internal nn.Module definition for Softmax Regression."""
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            logits = self.linear(x)
            probabilities = self.softmax(logits)
            return probabilities

    def _build_network(self, params: Dict, input_dim: int, output_dim: int) -> nn.Module:
        """Builds the Softmax Regression network."""
        # Softmax doesn't have structural hyperparameters in this setup
        return self._SoftmaxNet(input_dim, output_dim)


# =============================================================================
# 7. MLP1Model Class
# =============================================================================

class MLP1Model(BaseNNModel):
    """Handles 1-Layer MLP CV, training, loading, and evaluation."""
    MODEL_NAME = "mlp1"
    PARAM_GRID = MLP1_PARAM_GRID # Use global constant

    class _MLP1Net(nn.Module):
        """Internal nn.Module definition for 1-Hidden-Layer MLP."""
        def __init__(self, input_dim, output_dim, n_hidden, dropout_rate):
            super().__init__()
            self.layer_1 = nn.Linear(input_dim, n_hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.layer_2 = nn.Linear(n_hidden, output_dim)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.dropout(self.relu(self.layer_1(x)))
            logits = self.layer_2(x)
            probabilities = self.softmax(logits)
            return probabilities

    def _build_network(self, params: Dict, input_dim: int, output_dim: int) -> nn.Module:
        """Builds the 1-Layer MLP network using hyperparameters from params."""
        n_hidden = int(params['n_hidden']) # Ensure integer type
        dropout_rate = params['dropout_rate']
        return self._MLP1Net(input_dim, output_dim, n_hidden, dropout_rate)


# =============================================================================
# 8. MLP2Model Class
# =============================================================================

class MLP2Model(BaseNNModel):
    """Handles 2-Layer MLP CV, training, loading, and evaluation."""
    MODEL_NAME = "mlp2"
    PARAM_GRID = MLP2_PARAM_GRID # Use global constant

    class _MLP2Net(nn.Module):
        """Internal nn.Module definition for 2-Hidden-Layer MLP."""
        def __init__(self, input_dim, output_dim, shared_hidden_size, dropout_rate):
            super().__init__()
            self.layer_1 = nn.Linear(input_dim, shared_hidden_size)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.layer_2 = nn.Linear(shared_hidden_size, shared_hidden_size)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)
            self.layer_3 = nn.Linear(shared_hidden_size, output_dim)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.dropout1(self.relu1(self.layer_1(x)))
            x = self.dropout2(self.relu2(self.layer_2(x)))
            logits = self.layer_3(x)
            probabilities = self.softmax(logits)
            return probabilities

    def _build_network(self, params: Dict, input_dim: int, output_dim: int) -> nn.Module:
        """Builds the 2-Layer MLP network using hyperparameters from params."""
        shared_hidden_size = int(params['shared_hidden_size']) # Ensure integer type
        dropout_rate = params['dropout_rate']
        return self._MLP2Net(input_dim, output_dim, shared_hidden_size, dropout_rate)


# =============================================================================
# End of Script (Ready for import and use in a notebook)
# =============================================================================