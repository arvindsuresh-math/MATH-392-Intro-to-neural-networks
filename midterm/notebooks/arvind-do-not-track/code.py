# =============================================================================
# Presidential Election Outcome Prediction - Midterm Project Code
#
# High-Level Overview:
# This script implements a framework for predicting US presidential election
# outcomes (Democrat, Republican, Other, Non-voter probabilities) at the
# county level using demographic data. It compares four model types:
# Ridge Regression, Softmax Regression, 1-Layer MLP, and 2-Layer MLP.
#
# Code Structure:
# 1. Imports & Global Constants: Sets up libraries and configuration parameters
#    (file paths, hyperparameters, training settings).
# 2. Weighted Loss Function: Defines the custom loss metric.
# 3. DataHandler Class: Encapsulates all data loading and DataLoader creation.
# 4. Model Classes (RidgeModel, SoftmaxModel, MLP1Model, MLP2Model):
#    - Each class manages a specific model type.
#    - PyTorch model classes define their network architecture internally
#      (as private inner classes).
#    - Each class provides methods for:
#        - `cross_validate`: Performs hyperparameter tuning via 3-fold CV,
#          saving results to a CSV. Includes internal training loops.
#        - `train_final_model`: Trains the model on the combined dataset
#          using the best hyperparameters found during CV. Saves the
#          trained model/state_dict and loss history (for NNs).
#        - `load_model`: Loads a previously saved trained model/state_dict.
#    - Configuration (epochs, paths, etc.) is passed explicitly to methods.
#
# Workflow (Implemented in a separate execution block, not shown here):
# a. Initialize DataHandler.
# b. Initialize each Model class.
# c. Run `cross_validate` for each model.
# d. Determine the best overall model type by comparing CV results CSVs.
# e. Run `train_final_model` for the best model type.
# f. (Optional) Load the final model and evaluate on a test set (e.g., 2020).
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
from typing import List, Dict, Tuple, Any, Callable, Optional, Union, Type

# =============================================================================
# 1. Global Constants and Configuration
# =============================================================================

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
}
MLP2_PARAM_GRID = {
    'shared_hidden_size': [16, 32, 64],
    'dropout_rate': [0.1, 0.3, 0.5],
    'learning_rate': [1e-2, 1e-3, 1e-4]
}

# --- Default File Paths ---
# (These could also be generated dynamically or passed to methods)
DEFAULT_RESULTS_DIR = "./cv_results"
DEFAULT_MODELS_DIR = "./trained_models"
DEFAULT_LOSS_DIR = "./loss_histories"

os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
os.makedirs(DEFAULT_LOSS_DIR, exist_ok=True)

# =============================================================================
# 2. Weighted Loss Function
# =============================================================================

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
    # Compute cross entropy loss for each sample
    # Note: targets are probabilities, not class indices
    sample_ce_loss = -torch.sum(targets * torch.log(outputs_clamped), dim=1, keepdim=True)
    # Ensure weights are broadcastable [batch_size, 1]
    weights_reshaped = weights.view_as(sample_ce_loss)
    # Weight each sample loss by P(C)
    weighted_sample_losses = sample_ce_loss * weights_reshaped
    # Return expected value (weighted average loss for the batch)
    total_weight = weights_reshaped.sum()
    if total_weight > 0:
        return weighted_sample_losses.sum() / total_weight
    else:
        # Handle cases with zero total weight (e.g., empty batch)
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)


# =============================================================================
# 3. DataHandler Class
# =============================================================================

class DataHandler:
    """Handles loading data and creating PyTorch DataLoaders."""

    def __init__(self, data_dir: str = ".", sample_suffix: str = "2008"):
        """
        Initializes the DataHandler and determines input/output dimensions.

        Args:
            data_dir (str): Directory containing the CSV data files.
            sample_suffix (str): Suffix of a sample file (e.g., '2008') used
                                 to determine data dimensions.
        """
        self.data_dir = data_dir
        # Determine dimensions from a sample file
        try:
            X_sample, y_sample, _ = self.load_raw_data(sample_suffix)
            self.input_dim = X_sample.shape[1]
            self.output_dim = y_sample.shape[1]
            print(f"DataHandler initialized: Input Dim={self.input_dim}, Output Dim={self.output_dim}")
        except FileNotFoundError:
            print(f"Error: Sample data files for suffix '{sample_suffix}' not found in '{data_dir}'. Cannot determine dimensions.")
            raise # Re-raise the error

    def get_dimensions(self) -> Tuple[int, int]:
        """Returns the input and output dimensions."""
        return self.input_dim, self.output_dim

    def load_raw_data(self, suffix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads features (X), targets (y), and weights (wts) CSV files."""
        x_path = os.path.join(self.data_dir, f'X_{suffix}.csv')
        y_path = os.path.join(self.data_dir, f'y_{suffix}.csv')
        wts_path = os.path.join(self.data_dir, f'wts_{suffix}.csv')
        try:
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            wts = pd.read_csv(wts_path)
            # Ensure wts only contains the 'P(C)' column needed
            if 'P(C)' not in wts.columns:
                 raise ValueError(f"'P(C)' column not found in {wts_path}")
            wts = wts[['P(C)']]
            return X, y, wts
        except FileNotFoundError as e:
            print(f"Error loading data file: {e}")
            raise

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

    def create_final_dataloader(self, combined_suffix: str, batch_size: int) -> DataLoader:
        """Creates a DataLoader for the final combined training dataset."""
        X_final, y_final, wts_final = self.load_raw_data(combined_suffix)
        final_dataset = self._create_dataset(X_final, y_final, wts_final)

        num_workers = 2 if DEVICE.type == 'cuda' else 0
        pin_memory = True if DEVICE.type == 'cuda' else False
        torch.manual_seed(42) # For reproducible shuffling in final training

        final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        print(f"Created final DataLoader: {len(final_loader)} batches")
        return final_loader


# =============================================================================
# 4. RidgeModel Class
# =============================================================================

class RidgeModel:
    """Handles Ridge Regression CV, training, and saving."""
    MODEL_NAME = "ridge"

    def __init__(self):
        self.model: Optional[Ridge] = None

    def cross_validate(self,
                       data_handler: DataHandler,
                       fold_definitions: List[Dict[str, str]],
                       param_grid: List[float],
                       results_dir: str = DEFAULT_RESULTS_DIR
                       ) -> None:
        """Performs CV for Ridge and saves results."""
        results_save_path = os.path.join(results_dir, f"{self.MODEL_NAME}_cv_results.csv")
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        results_list = []
        best_score = float('inf')

        for alpha in param_grid:
            print(f"  Testing alpha = {alpha}:")
            fold_validation_scores = []
            for fold in fold_definitions:
                X_train_pd, y_train_pd, _ = data_handler.load_raw_data(fold['train_suffix'])
                X_val_pd, y_val_pd, _ = data_handler.load_raw_data(fold['val_suffix'])

                model_fold = Ridge(alpha=alpha)
                model_fold.fit(X_train_pd.values, y_train_pd.values)
                y_pred_val = model_fold.predict(X_val_pd.values)
                validation_score = mean_squared_error(y_val_pd.values, y_pred_val)
                fold_validation_scores.append(validation_score)

            avg_validation_score = np.mean(fold_validation_scores)
            print(f"    Avg Val Score (MSE): {avg_validation_score:.6f}")
            results_list.append({'alpha': alpha, 'mean_cv_score': avg_validation_score})
            if avg_validation_score < best_score:
                best_score = avg_validation_score

        results_df = pd.DataFrame(results_list).sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(results_save_path, index=False)
        print(f"\nBest {self.MODEL_NAME.upper()} CV score: {results_df['mean_cv_score'].iloc[0]:.6f}")
        print(f"CV results saved to: {results_save_path}")
        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")

    def train_final_model(self,
                          data_handler: DataHandler,
                          train_data_suffix: str,
                          results_dir: str = DEFAULT_RESULTS_DIR,
                          models_dir: str = DEFAULT_MODELS_DIR
                          ) -> Ridge:
        """Trains the final Ridge model using the best alpha from CV results."""
        cv_results_path = os.path.join(results_dir, f"{self.MODEL_NAME}_cv_results.csv")
        model_save_path = os.path.join(models_dir, f"{self.MODEL_NAME}_final_model.joblib")

        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")
        # 1. Read best alpha from CV results
        try:
            results_df = pd.read_csv(cv_results_path)
            if results_df.empty: raise ValueError("CV results file is empty.")
            best_alpha = results_df.iloc[0]['alpha']
            print(f"Using best alpha from CV: {best_alpha}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error reading best alpha from {cv_results_path}: {e}. Cannot train final model.")
            raise

        # 2. Build model instance
        self.model = Ridge(alpha=best_alpha)

        # 3. Load combined training data
        X_train_pd, y_train_pd, _ = data_handler.load_raw_data(train_data_suffix)
        print(f"Loaded final training data: {X_train_pd.shape[0]} samples.")

        # 4. Fit the model
        print("Fitting final Ridge model...")
        self.model.fit(X_train_pd.values, y_train_pd.values)
        print("Ridge model fitting complete.")

        # 5. Save the trained model
        joblib.dump(self.model, model_save_path)
        print(f"Saved final trained Ridge model to: {model_save_path}")
        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        return self.model

    def load_model(self, models_dir: str = DEFAULT_MODELS_DIR) -> Ridge:
        """Loads a trained Ridge model."""
        load_path = os.path.join(models_dir, f"{self.MODEL_NAME}_final_model.joblib")
        print(f"Loading trained Ridge model from: {load_path}")
        try:
            self.model = joblib.load(load_path)
            print("Ridge model loaded successfully.")
            return self.model
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}")
            raise


# =============================================================================
# 5. SoftmaxModel Class
# =============================================================================

class SoftmaxModel:
    """Handles Softmax Regression CV, training, and saving."""
    MODEL_NAME = "softmax"

    class _SoftmaxNet(nn.Module):
        """Internal nn.Module definition for Softmax Regression."""
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            # Softmax is implicitly handled by CrossEntropyLoss if using logits,
            # but our custom loss expects probabilities.
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            logits = self.linear(x)
            probabilities = self.softmax(logits)
            return probabilities

    def __init__(self):
        self.model: Optional[SoftmaxModel._SoftmaxNet] = None
        self.final_loss_history: List[Dict[str, Any]] = []

    def _train_one_fold(self, model, train_loader, val_loader, optimizer, loss_fn, max_epochs, patience, device, verbose=False):
        """Internal helper to train model for one CV fold with early stopping."""
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0

        model.to(device)

        for epoch in range(max_epochs):
            model.train()
            train_loss_epoch = 0.0
            for features, targets, weights in train_loader:
                features, targets, weights = features.to(device), targets.to(device), weights.to(device)
                outputs = model(features)
                loss = loss_fn(outputs, targets, weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
            avg_train_loss = train_loss_epoch / len(train_loader)

            model.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for features, targets, weights in val_loader:
                    features, targets, weights = features.to(device), targets.to(device), weights.to(device)
                    outputs = model(features)
                    loss = loss_fn(outputs, targets, weights)
                    val_loss_epoch += loss.item()
            avg_val_loss = val_loss_epoch / len(val_loader)

            if verbose and (epoch + 1) % 20 == 0: # Print less frequently during CV
                 print(f"    Epoch {epoch+1}/{max_epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose: print(f"    Early stopping at epoch {epoch+1}. Best val loss {best_val_loss:.6f} at epoch {best_epoch}.")
                break

        # Handle case where validation loss never improved
        if best_val_loss == float('inf'):
             print("    Warning: Validation loss did not improve.")
             # Decide how to handle this, e.g., return inf or avg_val_loss from last epoch
             return avg_val_loss if avg_val_loss != float('inf') else float('inf') # Return last epoch's loss if valid

        return best_val_loss


    def cross_validate(self,
                       data_handler: DataHandler,
                       fold_definitions: List[Dict[str, str]],
                       param_grid: Dict,
                       max_epochs: int,
                       patience: int,
                       batch_size: int,
                       optimizer_choice: Type[optim.Optimizer],
                       loss_fn: Callable,
                       device: torch.device,
                       results_dir: str = DEFAULT_RESULTS_DIR
                       ) -> None:
        """Performs CV for Softmax and saves results."""
        results_save_path = os.path.join(results_dir, f"{self.MODEL_NAME}_cv_results.csv")
        input_dim, output_dim = data_handler.get_dimensions()
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        results_list = []
        best_score = float('inf')

        for config in ParameterGrid(param_grid):
            print(f"  Testing config: {config}")
            fold_validation_scores = []
            for i, fold in enumerate(fold_definitions):
                print(f"    Fold {i+1}/{len(fold_definitions)}...")
                train_loader, val_loader = data_handler.create_dataloaders(fold['train_suffix'], fold['val_suffix'], batch_size)

                model_fold = self._SoftmaxNet(input_dim, output_dim).to(device)
                optimizer = optimizer_choice(model_fold.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

                validation_score = self._train_one_fold(model_fold, train_loader, val_loader, optimizer, loss_fn, max_epochs, patience, device)
                fold_validation_scores.append(validation_score)
                # Explicitly delete loaders and model to free memory if needed, especially GPU memory
                del train_loader, val_loader, model_fold, optimizer
                if device.type == 'cuda': torch.cuda.empty_cache()
                elif device.type == 'mps': torch.mps.empty_cache()


            avg_validation_score = np.mean(fold_validation_scores) if fold_validation_scores else float('inf')
            print(f"    Avg Val Score: {avg_validation_score:.6f}")
            results_list.append({**config, 'mean_cv_score': avg_validation_score})
            if avg_validation_score < best_score:
                best_score = avg_validation_score

        results_df = pd.DataFrame(results_list).sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(results_save_path, index=False)
        # Check if results_df is not empty before accessing iloc[0]
        if not results_df.empty:
            print(f"\nBest {self.MODEL_NAME.upper()} CV score: {results_df['mean_cv_score'].iloc[0]:.6f}")
        else:
             print(f"\nNo valid results found for {self.MODEL_NAME.upper()} CV.")
        print(f"CV results saved to: {results_save_path}")
        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")


    def train_final_model(self,
                          data_handler: DataHandler,
                          train_data_suffix: str,
                          final_train_epochs: int,
                          batch_size: int,
                          optimizer_choice: Type[optim.Optimizer],
                          loss_fn: Callable,
                          device: torch.device,
                          results_dir: str = DEFAULT_RESULTS_DIR,
                          models_dir: str = DEFAULT_MODELS_DIR,
                          loss_dir: str = DEFAULT_LOSS_DIR
                          ) -> nn.Module:
        """Trains the final Softmax model using the best hyperparams from CV."""
        cv_results_path = os.path.join(results_dir, f"{self.MODEL_NAME}_cv_results.csv")
        state_dict_save_path = os.path.join(models_dir, f"{self.MODEL_NAME}_final_state_dict.pth")
        loss_save_path = os.path.join(loss_dir, f"{self.MODEL_NAME}_final_training_loss.csv")
        input_dim, output_dim = data_handler.get_dimensions()

        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")

        # 1. Read best hyperparameters from CV results
        try:
            results_df = pd.read_csv(cv_results_path)
            if results_df.empty: raise ValueError("CV results file is empty.")
            best_params = results_df.iloc[0].to_dict()
            learning_rate = best_params['learning_rate']
            weight_decay = best_params.get('weight_decay', 0) # Use 0 if not found
            print(f"Using best hyperparameters from CV: LR={learning_rate}, WD={weight_decay}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error reading best hyperparameters from {cv_results_path}: {e}. Cannot train final model.")
            raise

        # 2. Build model instance
        self.model = self._SoftmaxNet(input_dim, output_dim).to(device)

        # 3. Create final DataLoader
        final_train_loader = data_handler.create_final_dataloader(train_data_suffix, batch_size)

        # 4. Setup Optimizer
        optimizer = optimizer_choice(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 5. Training Loop
        self.model.train()
        self.final_loss_history = []
        print(f"Training for {final_train_epochs} epochs...")

        for epoch in range(final_train_epochs):
            epoch_loss = 0.0
            for features, targets, weights in final_train_loader:
                features, targets, weights = features.to(device), targets.to(device), weights.to(device)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = loss_fn(outputs, targets, weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(final_train_loader)

            if (epoch + 1) % 10 == 0 or epoch == final_train_epochs - 1:
                print(f"  Epoch {epoch+1}/{final_train_epochs} - Training Loss: {avg_epoch_loss:.6f}")
                self.final_loss_history.append({'epoch': epoch + 1, 'loss': avg_epoch_loss})

        # 6. Save state dict and loss history
        torch.save(self.model.state_dict(), state_dict_save_path)
        print(f"Saved final model state_dict to: {state_dict_save_path}")
        loss_df = pd.DataFrame(self.final_loss_history)
        loss_df.to_csv(loss_save_path, index=False)
        print(f"Saved final training loss history to: {loss_save_path}")
        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        self.model.eval() # Set to evaluation mode after training
        return self.model

    def load_model(self, input_dim: int, output_dim: int, device: torch.device, models_dir: str = DEFAULT_MODELS_DIR) -> nn.Module:
        """Loads a trained Softmax model state_dict."""
        load_path = os.path.join(models_dir, f"{self.MODEL_NAME}_final_state_dict.pth")
        print(f"Loading trained {self.MODEL_NAME} state_dict from: {load_path}")
        try:
            # Need to instantiate model architecture first
            self.model = self._SoftmaxNet(input_dim, output_dim).to(device)
            self.model.load_state_dict(torch.load(load_path, map_location=device))
            self.model.eval() # Set to evaluation mode
            print(f"{self.MODEL_NAME} model loaded successfully.")
            return self.model
        except FileNotFoundError:
            print(f"Error: State dict file not found at {load_path}")
            raise
        except RuntimeError as e:
             print(f"Error loading state dict (likely architecture mismatch): {e}")
             raise


# =============================================================================
# 6. MLP1Model Class
# =============================================================================

class MLP1Model:
    """Handles 1-Layer MLP CV, training, and saving."""
    MODEL_NAME = "mlp1"

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

    def __init__(self):
        self.model: Optional[MLP1Model._MLP1Net] = None
        self.final_loss_history: List[Dict[str, Any]] = []

    # --- _train_one_fold (Identical to SoftmaxModel's, could be inherited/shared) ---
    def _train_one_fold(self, model, train_loader, val_loader, optimizer, loss_fn, max_epochs, patience, device, verbose=False):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
        model.to(device)
        for epoch in range(max_epochs):
            model.train()
            train_loss_epoch = 0.0
            for features, targets, weights in train_loader:
                features, targets, weights = features.to(device), targets.to(device), weights.to(device)
                outputs = model(features)
                loss = loss_fn(outputs, targets, weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
            avg_train_loss = train_loss_epoch / len(train_loader)

            model.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for features, targets, weights in val_loader:
                    features, targets, weights = features.to(device), targets.to(device), weights.to(device)
                    outputs = model(features)
                    loss = loss_fn(outputs, targets, weights)
                    val_loss_epoch += loss.item()
            avg_val_loss = val_loss_epoch / len(val_loader)

            if verbose and (epoch + 1) % 20 == 0:
                 print(f"    Epoch {epoch+1}/{max_epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose: print(f"    Early stopping at epoch {epoch+1}. Best val loss {best_val_loss:.6f} at epoch {best_epoch}.")
                break
        if best_val_loss == float('inf'):
             print("    Warning: Validation loss did not improve.")
             return avg_val_loss if avg_val_loss != float('inf') else float('inf')
        return best_val_loss

    def cross_validate(self,
                       data_handler: DataHandler,
                       fold_definitions: List[Dict[str, str]],
                       param_grid: Dict,
                       max_epochs: int,
                       patience: int,
                       batch_size: int,
                       optimizer_choice: Type[optim.Optimizer],
                       loss_fn: Callable,
                       device: torch.device,
                       results_dir: str = DEFAULT_RESULTS_DIR
                       ) -> None:
        """Performs CV for MLP1 and saves results."""
        results_save_path = os.path.join(results_dir, f"{self.MODEL_NAME}_cv_results.csv")
        input_dim, output_dim = data_handler.get_dimensions()
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        results_list = []
        best_score = float('inf')

        for config in ParameterGrid(param_grid):
            print(f"  Testing config: {config}")
            fold_validation_scores = []
            for i, fold in enumerate(fold_definitions):
                print(f"    Fold {i+1}/{len(fold_definitions)}...")
                train_loader, val_loader = data_handler.create_dataloaders(fold['train_suffix'], fold['val_suffix'], batch_size)

                # Build model with specific structural params from config
                model_fold = self._MLP1Net(input_dim, output_dim, int(config['n_hidden']), config['dropout_rate']).to(device)
                optimizer = optimizer_choice(model_fold.parameters(), lr=config['learning_rate']) # WD not tuned here

                validation_score = self._train_one_fold(model_fold, train_loader, val_loader, optimizer, loss_fn, max_epochs, patience, device)
                fold_validation_scores.append(validation_score)
                del train_loader, val_loader, model_fold, optimizer
                if device.type == 'cuda': torch.cuda.empty_cache()
                elif device.type == 'mps': torch.mps.empty_cache()

            avg_validation_score = np.mean(fold_validation_scores) if fold_validation_scores else float('inf')
            print(f"    Avg Val Score: {avg_validation_score:.6f}")
            results_list.append({**config, 'mean_cv_score': avg_validation_score})
            if avg_validation_score < best_score:
                best_score = avg_validation_score

        results_df = pd.DataFrame(results_list).sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(results_save_path, index=False)
        if not results_df.empty:
            print(f"\nBest {self.MODEL_NAME.upper()} CV score: {results_df['mean_cv_score'].iloc[0]:.6f}")
        else:
             print(f"\nNo valid results found for {self.MODEL_NAME.upper()} CV.")
        print(f"CV results saved to: {results_save_path}")
        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")

    def train_final_model(self,
                          data_handler: DataHandler,
                          train_data_suffix: str,
                          final_train_epochs: int,
                          batch_size: int,
                          optimizer_choice: Type[optim.Optimizer],
                          loss_fn: Callable,
                          device: torch.device,
                          results_dir: str = DEFAULT_RESULTS_DIR,
                          models_dir: str = DEFAULT_MODELS_DIR,
                          loss_dir: str = DEFAULT_LOSS_DIR
                          ) -> nn.Module:
        """Trains the final MLP1 model using the best hyperparams from CV."""
        cv_results_path = os.path.join(results_dir, f"{self.MODEL_NAME}_cv_results.csv")
        state_dict_save_path = os.path.join(models_dir, f"{self.MODEL_NAME}_final_state_dict.pth")
        loss_save_path = os.path.join(loss_dir, f"{self.MODEL_NAME}_final_training_loss.csv")
        input_dim, output_dim = data_handler.get_dimensions()

        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")

        # 1. Read best hyperparameters from CV results
        try:
            results_df = pd.read_csv(cv_results_path)
            if results_df.empty: raise ValueError("CV results file is empty.")
            best_params = results_df.iloc[0].to_dict()
            learning_rate = best_params['learning_rate']
            n_hidden = int(best_params['n_hidden'])
            dropout_rate = best_params['dropout_rate']
            print(f"Using best hyperparameters from CV: LR={learning_rate}, Hidden={n_hidden}, Dropout={dropout_rate}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error reading best hyperparameters from {cv_results_path}: {e}. Cannot train final model.")
            raise

        # 2. Build model instance
        self.model = self._MLP1Net(input_dim, output_dim, n_hidden, dropout_rate).to(device)

        # 3. Create final DataLoader
        final_train_loader = data_handler.create_final_dataloader(train_data_suffix, batch_size)

        # 4. Setup Optimizer
        optimizer = optimizer_choice(self.model.parameters(), lr=learning_rate) # WD not tuned here

        # 5. Training Loop
        self.model.train()
        self.final_loss_history = []
        print(f"Training for {final_train_epochs} epochs...")
        for epoch in range(final_train_epochs):
            epoch_loss = 0.0
            for features, targets, weights in final_train_loader:
                features, targets, weights = features.to(device), targets.to(device), weights.to(device)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = loss_fn(outputs, targets, weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(final_train_loader)
            if (epoch + 1) % 10 == 0 or epoch == final_train_epochs - 1:
                print(f"  Epoch {epoch+1}/{final_train_epochs} - Training Loss: {avg_epoch_loss:.6f}")
                self.final_loss_history.append({'epoch': epoch + 1, 'loss': avg_epoch_loss})

        # 6. Save state dict and loss history
        torch.save(self.model.state_dict(), state_dict_save_path)
        print(f"Saved final model state_dict to: {state_dict_save_path}")
        loss_df = pd.DataFrame(self.final_loss_history)
        loss_df.to_csv(loss_save_path, index=False)
        print(f"Saved final training loss history to: {loss_save_path}")
        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        self.model.eval()
        return self.model

    def load_model(self, n_hidden: int, dropout_rate: float, input_dim: int, output_dim: int, device: torch.device, models_dir: str = DEFAULT_MODELS_DIR) -> nn.Module:
        """Loads a trained MLP1 model state_dict."""
        load_path = os.path.join(models_dir, f"{self.MODEL_NAME}_final_state_dict.pth")
        print(f"Loading trained {self.MODEL_NAME} state_dict from: {load_path}")
        try:
            self.model = self._MLP1Net(input_dim, output_dim, n_hidden, dropout_rate).to(device)
            self.model.load_state_dict(torch.load(load_path, map_location=device))
            self.model.eval()
            print(f"{self.MODEL_NAME} model loaded successfully.")
            return self.model
        except FileNotFoundError:
            print(f"Error: State dict file not found at {load_path}")
            raise
        except RuntimeError as e:
             print(f"Error loading state dict (likely architecture mismatch): {e}")
             raise


# =============================================================================
# 7. MLP2Model Class
# =============================================================================

class MLP2Model:
    """Handles 2-Layer MLP CV, training, and saving."""
    MODEL_NAME = "mlp2"

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

    def __init__(self):
        self.model: Optional[MLP2Model._MLP2Net] = None
        self.final_loss_history: List[Dict[str, Any]] = []

    # --- _train_one_fold (Identical to SoftmaxModel's, could be inherited/shared) ---
    def _train_one_fold(self, model, train_loader, val_loader, optimizer, loss_fn, max_epochs, patience, device, verbose=False):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
        model.to(device)
        for epoch in range(max_epochs):
            model.train()
            train_loss_epoch = 0.0
            for features, targets, weights in train_loader:
                features, targets, weights = features.to(device), targets.to(device), weights.to(device)
                outputs = model(features)
                loss = loss_fn(outputs, targets, weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
            avg_train_loss = train_loss_epoch / len(train_loader)

            model.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for features, targets, weights in val_loader:
                    features, targets, weights = features.to(device), targets.to(device), weights.to(device)
                    outputs = model(features)
                    loss = loss_fn(outputs, targets, weights)
                    val_loss_epoch += loss.item()
            avg_val_loss = val_loss_epoch / len(val_loader)

            if verbose and (epoch + 1) % 20 == 0:
                 print(f"    Epoch {epoch+1}/{max_epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose: print(f"    Early stopping at epoch {epoch+1}. Best val loss {best_val_loss:.6f} at epoch {best_epoch}.")
                break
        if best_val_loss == float('inf'):
             print("    Warning: Validation loss did not improve.")
             return avg_val_loss if avg_val_loss != float('inf') else float('inf')
        return best_val_loss


    def cross_validate(self,
                       data_handler: DataHandler,
                       fold_definitions: List[Dict[str, str]],
                       param_grid: Dict,
                       max_epochs: int,
                       patience: int,
                       batch_size: int,
                       optimizer_choice: Type[optim.Optimizer],
                       loss_fn: Callable,
                       device: torch.device,
                       results_dir: str = DEFAULT_RESULTS_DIR
                       ) -> None:
        """Performs CV for MLP2 and saves results."""
        results_save_path = os.path.join(results_dir, f"{self.MODEL_NAME}_cv_results.csv")
        input_dim, output_dim = data_handler.get_dimensions()
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        results_list = []
        best_score = float('inf')

        for config in ParameterGrid(param_grid):
            print(f"  Testing config: {config}")
            fold_validation_scores = []
            for i, fold in enumerate(fold_definitions):
                print(f"    Fold {i+1}/{len(fold_definitions)}...")
                train_loader, val_loader = data_handler.create_dataloaders(fold['train_suffix'], fold['val_suffix'], batch_size)

                # Build model with specific structural params from config
                model_fold = self._MLP2Net(input_dim, output_dim, int(config['shared_hidden_size']), config['dropout_rate']).to(device)
                optimizer = optimizer_choice(model_fold.parameters(), lr=config['learning_rate'])

                validation_score = self._train_one_fold(model_fold, train_loader, val_loader, optimizer, loss_fn, max_epochs, patience, device)
                fold_validation_scores.append(validation_score)
                del train_loader, val_loader, model_fold, optimizer
                if device.type == 'cuda': torch.cuda.empty_cache()
                elif device.type == 'mps': torch.mps.empty_cache()

            avg_validation_score = np.mean(fold_validation_scores) if fold_validation_scores else float('inf')
            print(f"    Avg Val Score: {avg_validation_score:.6f}")
            results_list.append({**config, 'mean_cv_score': avg_validation_score})
            if avg_validation_score < best_score:
                best_score = avg_validation_score

        results_df = pd.DataFrame(results_list).sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(results_save_path, index=False)
        if not results_df.empty:
            print(f"\nBest {self.MODEL_NAME.upper()} CV score: {results_df['mean_cv_score'].iloc[0]:.6f}")
        else:
             print(f"\nNo valid results found for {self.MODEL_NAME.upper()} CV.")
        print(f"CV results saved to: {results_save_path}")
        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")


    def train_final_model(self,
                          data_handler: DataHandler,
                          train_data_suffix: str,
                          final_train_epochs: int,
                          batch_size: int,
                          optimizer_choice: Type[optim.Optimizer],
                          loss_fn: Callable,
                          device: torch.device,
                          results_dir: str = DEFAULT_RESULTS_DIR,
                          models_dir: str = DEFAULT_MODELS_DIR,
                          loss_dir: str = DEFAULT_LOSS_DIR
                          ) -> nn.Module:
        """Trains the final MLP2 model using the best hyperparams from CV."""
        cv_results_path = os.path.join(results_dir, f"{self.MODEL_NAME}_cv_results.csv")
        state_dict_save_path = os.path.join(models_dir, f"{self.MODEL_NAME}_final_state_dict.pth")
        loss_save_path = os.path.join(loss_dir, f"{self.MODEL_NAME}_final_training_loss.csv")
        input_dim, output_dim = data_handler.get_dimensions()

        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")

        # 1. Read best hyperparameters from CV results
        try:
            results_df = pd.read_csv(cv_results_path)
            if results_df.empty: raise ValueError("CV results file is empty.")
            best_params = results_df.iloc[0].to_dict()
            learning_rate = best_params['learning_rate']
            shared_hidden_size = int(best_params['shared_hidden_size'])
            dropout_rate = best_params['dropout_rate']
            print(f"Using best hyperparameters from CV: LR={learning_rate}, Hidden={shared_hidden_size}, Dropout={dropout_rate}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error reading best hyperparameters from {cv_results_path}: {e}. Cannot train final model.")
            raise

        # 2. Build model instance
        self.model = self._MLP2Net(input_dim, output_dim, shared_hidden_size, dropout_rate).to(device)

        # 3. Create final DataLoader
        final_train_loader = data_handler.create_final_dataloader(train_data_suffix, batch_size)

        # 4. Setup Optimizer
        optimizer = optimizer_choice(self.model.parameters(), lr=learning_rate)

        # 5. Training Loop
        self.model.train()
        self.final_loss_history = []
        print(f"Training for {final_train_epochs} epochs...")
        for epoch in range(final_train_epochs):
            epoch_loss = 0.0
            for features, targets, weights in final_train_loader:
                features, targets, weights = features.to(device), targets.to(device), weights.to(device)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = loss_fn(outputs, targets, weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(final_train_loader)
            if (epoch + 1) % 10 == 0 or epoch == final_train_epochs - 1:
                print(f"  Epoch {epoch+1}/{final_train_epochs} - Training Loss: {avg_epoch_loss:.6f}")
                self.final_loss_history.append({'epoch': epoch + 1, 'loss': avg_epoch_loss})

        # 6. Save state dict and loss history
        torch.save(self.model.state_dict(), state_dict_save_path)
        print(f"Saved final model state_dict to: {state_dict_save_path}")
        loss_df = pd.DataFrame(self.final_loss_history)
        loss_df.to_csv(loss_save_path, index=False)
        print(f"Saved final training loss history to: {loss_save_path}")
        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        self.model.eval()
        return self.model

    def load_model(self, shared_hidden_size: int, dropout_rate: float, input_dim: int, output_dim: int, device: torch.device, models_dir: str = DEFAULT_MODELS_DIR) -> nn.Module:
        """Loads a trained MLP2 model state_dict."""
        load_path = os.path.join(models_dir, f"{self.MODEL_NAME}_final_state_dict.pth")
        print(f"Loading trained {self.MODEL_NAME} state_dict from: {load_path}")
        try:
            self.model = self._MLP2Net(input_dim, output_dim, shared_hidden_size, dropout_rate).to(device)
            self.model.load_state_dict(torch.load(load_path, map_location=device))
            self.model.eval()
            print(f"{self.MODEL_NAME} model loaded successfully.")
            return self.model
        except FileNotFoundError:
            print(f"Error: State dict file not found at {load_path}")
            raise
        except RuntimeError as e:
             print(f"Error loading state dict (likely architecture mismatch): {e}")
             raise

# =============================================================================
# End of Script (Execution block would follow here)
# =============================================================================