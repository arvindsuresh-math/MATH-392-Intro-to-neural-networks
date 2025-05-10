"""
Presidential Election Outcome Prediction - Project code-base

High-Level Overview:
This script provides a framework for predicting US presidential election outcomes (specifically, the probability of votes for Democrat, Republican,
Other candidates, and Non-voters) at the county level. It uses demographic features from multiple election years (2008, 2012, 2016, 2020).
The goal is to compare the predictive performance of different modeling approaches. 

Code Structure:
- Global Constants: File paths and device selection.
- DataHandler Class: Manages data loading, preprocessing (scaling), and splitting for cross-validation and final training/testing.
- RidgeModel Class: Implements Ridge Regression using scikit-learn, including cross-validation for alpha selection, final model training, and evaluation.
- MLPModel Class: A base class for Multi-Layer Perceptron models, providing methods for cross-validation, training, and evaluation.
- XGBoostModel Class: Implements XGBoost, including cross-validation, final training, and evaluation.

Example Usage in a Jupyter Notebook:
1. Instantiate `DataHandler`.
2. Instantiate model handlers (e.g., `RidgeModel()`, `MLPModel()`).
3. Create necessary directories (./data, ./models, ./results).
4. Run `cross_validate()` for each model handler, passing the `DataHandler` instance. This performs 3-fold CV on the training years to find optimal hyperparameters and saves the results to CSV files in './results'. Each fold uses two of the years for training and the third for validation
5. Run `train_final_model()` for each model handler. This loads the best hyperparameters found during CV, trains the model on all three training years, and saves the final model to './models'. (The file format is '.joblib' for Ridge, '.pth' state_dict for MLPs, '.json' for XGBoost)
6. Run `predict()` for each model handler. This loads the trained model (or uses the model  stored in attribute if one exists), predicts on the `test_year` data, and (if asked for) saves county-level predictions to CSV files in './results'.
7. Compare the aggregate cross-entropy scores across models.

File Locations:
- ./data/final_dataset.csv : Input dataset (expected location).
- ./models/ : Directory where trained models (and scaler) are saved.
- ./results/ : Directory where CV results, final loss history (for NNs),
               and county-level predictions are saved.
- ./preds/ : Directory where final predictions are saved.
- ./logs/ : Directory for logging information (if needed).
"""

# --- Imports ---

import os
import pandas as pd
import numpy as np
import json
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import joblib
from typing import List, Dict, Tuple, Any, Type, Union
import copy
import ast

# ======================================

# --- File paths ---
DATA_DIR = "./data"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
LOGS_DIR = "./logs"
PREDS_DIR = "./preds"

# --- Device selection ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device")

# ======================================

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

# ======================================

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
        self.raw_data = pd.read_csv(DATA_DIR + '/final_dataset.csv')

        # --- Feature and Target Definitions ---
        with open('variables.json', 'r') as f:
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

# ======================================

class RidgeModel:
    """Handles Ridge Regression CV, training, loading, and evaluation."""

    def __init__(self, model_name: str = 'ridge'):
        """Initializes the RidgeModel handler."""
        self.model_name: str = model_name
        self.model: Union[Ridge, None] = None
        self.best_alpha: Union[float, None] = None
        # Store results path for convenience
        self.results_save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_cv_results.csv")
        self.model_save_path = os.path.join(MODELS_DIR, f"{self.model_name}_final_model.joblib")

    def cross_validate(self,
                       dh: DataHandler,
                       param_grid: List[float],
                       save: bool = True,
                       ):
        """Performs CV for Ridge, stores the best alpha, saves results if asked for. Uses weighted MSE as train/val loss function."""
        start_time = time.time()
        print(f"\n--- Starting Cross-Validation for {self.model_name.upper()} ---")
        results_list = []
        best_val_loss = float('inf')
        best_alpha = None

        cv_data = dh.get_ridge_data('cv')

        for alpha in param_grid:
            print("--------------------------------------")
            fold_val_losses = []
            for (X_train, y_train, wts_train), (X_val, y_val, wts_val) in cv_data:
                model_fold = Ridge(alpha=alpha)
                model_fold.fit(X_train, y_train, sample_weight=wts_train.ravel())
                y_pred = model_fold.predict(X_val)
                val_loss = mean_squared_error(y_val, y_pred, sample_weight=wts_val)
                fold_val_losses.append(val_loss)

            mean_val_loss = np.mean(fold_val_losses)
            print(f"Alpha: {alpha:rjust(5)} | Val Loss: {mean_val_loss:.6f}")
            results_list.append({'alpha': alpha, 'Val loss': mean_val_loss})

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_alpha = alpha 

        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values(by='Val loss', ascending=True).reset_index(drop=True)

        if save:
            results_df.to_csv(self.results_save_path, index=False)
            print(f"CV results saved to: {self.results_save_path}")

        time_taken = time.time() - start_time
        mins = int(time_taken) // 60
        secs = int(time_taken) % 60
        print(f"{self.model_name.upper()} cross-validation completed in {mins}m {secs}s.")

        self.best_alpha = best_alpha
        return results_df

    def train_final_model(self,
                          dh: DataHandler
                          ):
        """Trains the final Ridge model using the best alpha from attribute or CV results."""
        if self.best_alpha is None:
            results_df = pd.read_csv(self.results_save_path)
            self.best_alpha = results_df.iloc[0]['alpha']
        self.model = Ridge(alpha=self.best_alpha)
        print(f"Using best alpha: {self.best_alpha}")

        X_train, y_train, wts_train = dh.get_ridge_data('train')
        self.model.fit(X_train, y_train, sample_weight=wts_train.ravel())

        joblib.dump(self.model, self.model_save_path)
        print(f"{self.model_name.upper()} training complete. Saved to {self.model_save_path}.")

    def load_model(self):
        """Loads a trained Ridge model into self.model."""
        self.model = joblib.load(self.model_save_path)
        print(f"Ridge model loaded successfully from {self.model_save_path}.")

    def predict(self, dh: 'DataHandler', save: bool = False):
        """Generates raw predictions for the test set using the trained Ridge model."""
        if self.model is None: self.load_model()

        X_test = dh.get_ridge_data('test')[0]  # Get test data (X_test, y_test, wts_test)
        y_pred = self.model.predict(X_test)

        if save:
            pred_df = pd.DataFrame(y_pred, columns=dh.targets)
            pred_df.to_csv(self.pred_save_path, index=False)
            print(f"County-level raw predictions saved to: {self.pred_save_path}")
        
        return y_pred

# ======================================

class MLPModel:
    """Handles Multi-Layer Perceptron CV, training, loading, and prediction."""

    def __init__(self, model_name: str = 'mlp'):
        """
        Initializes the MLPModel handler for a specific model configuration.
        Args:
            model_name (str): A unique name for this model configuration
                              (e.g., "softmax", "mlp1", "mlp2"). Used for saving files.
        """
        self.model_name: str = model_name
        self.model: Union[nn.Module, None] = None
        self.best_params: Dict = {}
        self.final_loss_history: List[Dict[str, Any]] = []

        # Define paths based on the specific model_name
        self.results_save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_cv_results.csv")
        self.state_dict_save_path = os.path.join(MODELS_DIR, f"{self.model_name}_final_state_dict.pth")
        self.loss_save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_final_training_loss.csv")
        self.pred_save_path = os.path.join(PREDS_DIR, f"{self.model_name}_predictions.csv")

    @staticmethod
    def weighted_cross_entropy_loss(outputs: torch.Tensor,
                                    targets: torch.Tensor,
                                    weights: torch.Tensor):
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

    def _build_network(self, params: Dict, input_dim: int):
        """
        Dynamically builds an nn.Sequential network based on hyperparameters. `params` must include `hidden_layers` (a list of integers specifying sizes of hidden layers, e.g., [] for softmax, [64] for MLP1, [32, 16] for MLP2) and 'dropout_rate'. Returns a PyTorch sequential model.
        """
        hidden_layers_config = params.get('hidden_layers', []) # Default to Softmax
        dropout_rate = params.get('dropout_rate', 0.0) # Default to no dropout

        layers = []
        current_dim = input_dim

        for hidden_size in hidden_layers_config:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_size # Update dim for next layer

        layers.append(nn.Linear(current_dim, 4))
        layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def _parse_best_params_from_csv(self, results_df = None):
        """Reads the best parameters from the CV results CSV file."""
        if results_df is None:
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

    def _train_one_config(self,
                            config: Dict[str, Any], # Hyperparam config for this run
                            config_id: int, # Unique ID for this config
                            config_history: Dict[str, Any],
                            dh: 'DataHandler',
                            batch_size: int,
                            rung_epochs: int,
                            rung_patience: int,
                            optimizer_choice: Type[optim.Optimizer]
                           ): 
        """
        Evaluates a single hyperparameter configuration across all CV folds
        up to a target cumulative epoch count, resuming state, and returns an updated config_history dict.
        """
        fold_best_val_losses = []
        fold_train_losses_at_best = []
        fold_last_epochs = []
        fold_best_epochs = []
        cv_dataloaders = dh.get_nn_data('cv', batch_size)

        start_time = time.time()
        # Loop through CV folds
        for j, (train_loader, val_loader) in enumerate(cv_dataloaders,1):
            # --- Build model and optimizer ---
            model = self._build_network(config, dh.input_dim).to(DEVICE)
            optimizer = optimizer_choice(model.parameters(),
                                         lr=config['learning_rate'],
                                         weight_decay=config.get('weight_decay', 0))

            # --- Initialize or Unpack Cumulative State for the Fold ---
            if f'fold_{j}_history' not in config_history:
                # First run: initialize required arguments for _train_one_fold
                best_model_state = copy.deepcopy(model.state_dict())
                last_model_state = copy.deepcopy(model.state_dict())
                best_val_loss = float('inf')
                train_loss_at_best = float('inf')
                best_epoch, last_epoch, patience_used = 0, 0, 0
            else:
                # Resuming: directly unpack the previous state tuple
                (best_model_state, last_model_state, patience_used, best_epoch,
                 last_epoch, best_val_loss, train_loss_at_best) = config_history[f'fold_{j}_history']

                # Load the last state from the previous run into the model
                model.load_state_dict(last_model_state) # Use the unpacked state

            # --- Training Loop: Continue from last_epoch + 1 up to rung_max_epochs ---
            for current_epoch in range(last_epoch + 1, rung_epochs + 1):
                # --- Training Phase ---
                model.train()
                epoch_train_loss_sum = 0.0
                for features, targets, weights in train_loader:
                    features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                    outputs = model(features)
                    loss = self.weighted_cross_entropy_loss(outputs, targets, weights)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss_sum += loss.item()
                avg_train_loss = epoch_train_loss_sum / len(train_loader)

                # --- Validation Phase ---
                model.eval()
                epoch_val_loss_sum = 0.0
                with torch.no_grad():
                    for features, targets, weights in val_loader:
                        features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                        outputs = model(features)
                        loss = self.weighted_cross_entropy_loss(outputs, targets, weights)
                        epoch_val_loss_sum += loss.item()
                avg_val_loss = epoch_val_loss_sum / len(val_loader)

                # --- Update metrics ---
                last_epoch = current_epoch # Update last epoch reached

                if avg_val_loss < best_val_loss:
                    # Improvement found
                    best_val_loss = avg_val_loss
                    best_epoch = last_epoch 
                    train_loss_at_best = avg_train_loss 
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_used = 0 # Reset patience counter
                else:
                    # No improvement
                    patience_used += 1 

                # --- Check Termination Conditions ---
                if patience_used >= rung_patience:
                    break

            # --- Prepare last model state ---
            last_model_state = copy.deepcopy(model.state_dict())

            # --- Update config_history for this fold ---
            config_history[f'fold_{j}_history'] = (best_model_state, 
                                                   last_model_state, 
                                                   patience_used,
                                                   best_epoch, 
                                                   last_epoch, 
                                                   best_val_loss, 
                                                   train_loss_at_best)

            # Extract the final best val loss and best epoch for averaging
            fold_best_val_losses.append(best_val_loss) 
            fold_best_epochs.append(best_epoch)
            fold_train_losses_at_best.append(train_loss_at_best)
            fold_last_epochs.append(last_epoch)

            # --- Clean up fold-specific resources ---
            del model, optimizer
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            elif DEVICE.type == 'mps': torch.mps.empty_cache()

        # --- Calculate and update means in config_history ---
        config_history['best_val_loss'] = np.round(np.mean(fold_best_val_losses),6)
        config_history['best_epoch'] = int(np.mean(fold_best_epochs))
        config_history['train_loss_at_best'] = np.round(np.mean(fold_train_losses_at_best),6)
        config_history['last_epoch'] = int(np.mean(fold_last_epochs))

        # Get time taken for this config in seconds
        time_taken = time.time() - start_time

        # Pretty print the results for this config
        config_str = '|  ' + str(config_id).rjust(4) + '  |'
        last_epoch_str = '    ' + str(config_history['last_epoch']).rjust(3) + '     |'
        best_epoch_str = '    ' + str(config_history['best_epoch']).rjust(3) + '     |'
        train_loss_str = '  ' + str(config_history['train_loss_at_best']).rjust(8) + '  |'
        val_loss_str = '  ' + str(config_history['best_val_loss']).rjust(8) + '  |'
        time_str = f"{time_taken:.2f}s"
        time_str = '    ' + time_str.rjust(8) + '    |'
        print(config_str + last_epoch_str + best_epoch_str + train_loss_str + val_loss_str + time_str)
        
        # return the updated config_history
        return config_history
    
    def _prune_config_history(self, 
                              cv_history_dict: Dict[int, Dict[str, Any]],
                              n_promote: int):
        """Accepts a dictionary of config histories and prunes it to the best `n_promote` entries based on the lowest 'best_val_loss' items. Returns the pruned dictionary."""
        sorted_items = sorted(cv_history_dict.items(), key=lambda item: item[1]['best_val_loss'])
        pruned_items = sorted_items[:n_promote]
        return {idx: history for idx, history in pruned_items}

    def cross_validate(self,
                         dh: 'DataHandler',
                         param_grid: Dict[str, Any],rung_schedule: List[Tuple[int, int]],
                         batch_size: int = 64,
                         optimizer_choice: Type[optim.Optimizer] = optim.AdamW,
                         reduction_factor: int = 3,
                         min_finalists: int = 3,
                         max_finalists: int = 10, 
                         save: bool = True
                        ):
        """
        Performs cross-validation using a Successive Halving Algorithm (SHA) approach. Sets `self.best_params` based on the best configuration among the survivors.

        Args:
            dh (DataHandler): The data handler instance.
            param_grid (Dict[str, Any]): The grid of hyperparameters to explore.
            batch_size (int): Batch size for training.
            optimizer_choice (Type[optim.Optimizer]): The optimizer class to use.
            rung_schedule (List[Tuple[int, int]]): A list of tuples (rung_epochs, rung_patience).
            reduction_factor (int): Factor by which to reduce the number of configs
                                   at each stage (e.g., 3 means keep top 2/3).
            min_finalists (int): Minimum number of configurations to keep at the end. 
            max_finalists (int): Maximum number of configurations to keep at the end.
            save (bool): Whether to save the results to a CSV file.

        Returns:
            results_df: Dataframe with the finalist configs and their performance metrics.
        """
        print(f"\n--- Starting SHA Cross-Validation for {self.model_name.upper()} (eta={reduction_factor}) ---")
        start_time = time.time()

        # --- 1. Initialization ---
        all_configs_list = list(ParameterGrid(param_grid))
        num_initial_configs = len(all_configs_list)
        # Map index to config for easy lookup
        config_dict = {i: config for i, config in enumerate(all_configs_list,1)}
        # Main history dictionary: Stores state for active configs
        cv_history_dict = {i+1: {} for i in range(num_initial_configs)} 

        # --- 2. Rung Iteration ---
        num_rungs = len(rung_schedule)
        for rung, (rung_epochs, rung_patience) in enumerate(rung_schedule, 1):
            num_configs_in_rung = len(cv_history_dict)
            print("-------------------------------------------------------------------------------")
            print(f">>> SHA Rung {rung}/{num_rungs} | Target Epochs: {rung_epochs} | Patience: {rung_patience} | Evaluating {num_configs_in_rung} configs <<<")
            print("-------------------------------------------------------------------------------")
            print("| Config | Last Epoch | Best Epoch | Train Loss |  Val Loss  | Time (seconds) |")
            # --- Evaluate each active configuration ---
            for config_id, config_history in cv_history_dict.items():
                config=config_dict[config_id]
                # Call helper to train/evaluate this config for the current rung
                updated_config_history = self._train_one_config(
                                                config,
                                                config_id,
                                                config_history,
                                                dh,
                                                batch_size, 
                                                rung_epochs,
                                                rung_patience,
                                                optimizer_choice
                                            )

                # Update the main history dictionary with the results
                cv_history_dict[config_id] = updated_config_history

            # --- 3. Pruning Step ---
            if rung < num_rungs:
                # Not last round, so at least min_finalist configs should be promoted
                n_promote = max(min_finalists, int(num_configs_in_rung * (1-1/reduction_factor)))
            else:
                # Last rung, so promote up to max_finalists
                n_promote = min(max_finalists, num_configs_in_rung)
            cv_history_dict = self._prune_config_history(cv_history_dict, n_promote)

        # --- 4. Post-SHA Aggregation for Final Report ---
        final_results_list = []
        # Iterate through the final set of config histories
        for config_id, config_history in cv_history_dict.items():
            config = config_dict[config_id]
            result_entry = {
                **config, 
                'last_epoch': config_history['last_epoch'],
                'best_epoch': config_history['best_epoch'],
                'train_loss_at_best': config_history['train_loss_at_best'],
                'best_val_loss': config_history['best_val_loss']
                            }
            if 'hidden_layers' in result_entry:
                result_entry['hidden_layers'] = str(result_entry['hidden_layers'])
            final_results_list.append(result_entry)

        results_df = pd.DataFrame(final_results_list)
        results_df = results_df.sort_values(by='best_val_loss', ascending=True).reset_index(drop=True)

        # --- 5. Save/return Results ---
        if save:
            results_df.to_csv(self.results_save_path, index=False)
            print(f"Final results for {len(results_df)} surviving configurations saved to: {self.results_save_path}")

        time_taken = time.time() - start_time
        mins = int(time_taken) // 60
        secs = int(time_taken) % 60
        print(f"{self.model_name.upper()} cross-validation completed in {mins}m {secs}s.")

        self.best_params = self._parse_best_params_from_csv(results_df)
        return results_df
 
    def train_final_model(self,
                          dh: DataHandler,
                          final_train_epochs: int,
                          optimizer_choice: Type[optim.Optimizer] = optim.AdamW,
                          batch_size: int = 64,
                          final_patience: int = 50 # For early stopping
                          ):
        """Trains the final NN model using the best hyperparams from CV."""
        print(f"\n--- Starting Final Model Training for {self.model_name.upper()} ---")

        if not self.best_params:
            self.best_params = self._parse_best_params_from_csv()
        print(f"Using best hyperparameters from CV: {self.best_params}")

        self.model = self._build_network(self.best_params, dh.input_dim).to(DEVICE)
        final_train_loader = dh.get_nn_data('train', batch_size)

        lr = self.best_params['learning_rate']
        wd = self.best_params.get('weight_decay', 0) # Use 0 if not found
        optimizer = optimizer_choice(self.model.parameters(), lr=lr, weight_decay=wd)

        best_loss = float('inf')
        epochs_no_improve = 0
        best_state_dict = None 

        self.model.train() 
        self.final_loss_history = []
        print(f"Starting final training for up to {final_train_epochs} epochs (Patience: {final_patience})...")

        for epoch in range(final_train_epochs):
            epoch_loss_sum = 0.0
            for features, targets, weights in final_train_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.weighted_cross_entropy_loss(outputs, targets, weights)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item()

            # Calculate mean batch loss for the epoch
            avg_epoch_loss = epoch_loss_sum / len(final_train_loader) 
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

        # Load the best performing model state before saving
        self.model.load_state_dict(best_state_dict)
        print(f"Loaded model state from epoch with best loss: {best_loss:.6f}")

        torch.save(self.model.state_dict(), self.state_dict_save_path)
        print(f"Saved best model state_dict to: {self.state_dict_save_path}")

        loss_df = pd.DataFrame(self.final_loss_history)
        loss_df.to_csv(self.loss_save_path, index=False)
        print(f"Saved training loss history to: {self.loss_save_path}")
        print(f"{self.model_name.upper()} training complete.")

        self.model.eval()

    def load_model(self, input_dim: int):
        """Loads a trained NN model state_dict, using hyperparameters from CV results."""
        print(f"Loading state_dict from: {self.state_dict_save_path}")
        print(f"Reading hyperparameters from: {self.results_save_path}")

        # 1. Build model architecture using subclass method and best params
        self.best_params = self._parse_best_params_from_csv()
        self.model = self._build_network(self.best_params, input_dim).to(DEVICE)

        # 2. Load the saved state dictionary onto the same device
        self.model.load_state_dict(torch.load(self.state_dict_save_path, map_location=DEVICE))
        print(f"{self.model_name.upper()} model loaded successfully onto {DEVICE}.")
        
        return self.model

    def predict(self, dh: DataHandler, save: bool = False):
        """
        Generates raw predictions (numpy array of shape [n_samples, 4]) for the test set using the trained NN model. If `save` is True, then saves the predictions to a CSV file.
        """
        print(f"\n--- Generating Predictions for {self.model_name.upper()} on Year {dh.test_year} ---")

        X_test, y_test = dh.get_nn_data('test', batch_size=1) 
        y_tots = y_test.sum(axis = 1, keepdims=True) # Probability of being 18plus by county

        X_test, y_test, y_tots = X_test.to(DEVICE), y_test.to(DEVICE), y_tots.to(DEVICE)
        self.model.eval()
        self.model.to(DEVICE)

        with torch.no_grad():
            outputs = self.model(X_test) 
            y_pred = outputs * y_tots # Shape [n_samples, 4]

        y_pred = y_pred.cpu().numpy()

        if save:
            pred_df = pd.DataFrame(y_pred, columns=dh.targets) # Use global 'targets'
            pred_df.to_csv(self.pred_save_path, index=False)
            print(f"County-level raw predictions saved to: {self.pred_save_path}")

        return y_pred

# ======================================


class XGBoostModel:
    """Handles XGBoost CV, training, loading, and prediction."""

    def __init__(self, model_name: str = 'xgboost'):
        """
        Initializes the XGBoostModel handler.

        Args:
            model_name (str): A unique name for this model instance (e.g., "xgboost_custom").
        """
        self.model_name: str = model_name
        self.model: Union[xgb.Booster, None] = None
        self.best_params: Dict = {} # Stores best hyperparams found by CV
        self.optimal_boost_rounds: Union[int, None] = None # Stores optimal avg # rounds from CV

        # Define file paths based on model_name
        self.results_save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_cv_results.csv")
        self.model_save_path = os.path.join(MODELS_DIR, f"{self.model_name}_final_model.json")
        self.pred_save_path = os.path.join(PREDS_DIR, f"{self.model_name}_predictions.csv")

    @staticmethod
    def softmax(x: np.ndarray):
        """Numerically stable softmax function."""
        # Subtract max for numerical stability before exp
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def weighted_softprob_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Static custom XGBoost objective for weighted cross-entropy with soft labels.
        """
        # 1. Get true labels (soft probabilities) and weights
        n_samples = preds.shape[0]
        labels = dtrain.get_label().reshape((n_samples, 4))
        weights = dtrain.get_weight().reshape((n_samples, 1))
        weights = weights / weights.sum() # Normalize weights

        # 3. Calculate predicted probabas
        tots = labels.sum(axis=1, keepdims=True) # P(18plus) per sample
        probs = XGBoostModel.softmax(preds)
        probs = probs * tots 

        # 4. Calculate Gradient: (q - p)
        grad = probs - labels

        # 5. Calculate Hessian (diagonal approximation): w * q * (1 - q)
        # hess = weights * probs * (tots - probs)
        # hess = np.maximum(hess, 1e-12) # Ensure non-negative hessian
        hess = probs * (1 - probs)
        hess = np.maximum(hess, 1e-12)

        return (grad,hess)

    @staticmethod
    def weighted_cross_entropy_eval(preds: np.ndarray, dtrain: xgb.DMatrix):
        """
        Static custom evaluation metric for weighted cross-entropy with soft labels.
        """
        # 1. Get true labels (soft probabilities) and weights
        n_samples = preds.shape[0]
        labels = dtrain.get_label().reshape((n_samples, 4))
        weights = dtrain.get_weight().reshape((n_samples, 1))
        weights = weights / weights.sum() # Normalize weights

        # 3. Calculate predicted probabilities using softmax
        tots = labels.sum(axis=1, keepdims=True) # Total votes per sample
        probs = XGBoostModel.softmax(preds)
        probs = probs * tots

        # 4. Calculate weighted cross-entropy per sample
        epsilon = 1e-9
        probs = np.clip(probs, epsilon, 1. - epsilon)
        sample_surprisals = - labels * np.log(probs) #surprisal weighted by labels
        sample_loss = sample_surprisals.sum(axis=1) #sum across classes to get CE loss per sample

        # 5. Calculate average weighted cross-entropy
        weighted_avg_ce = np.average(sample_loss, weights=weights.flatten())

        return 'weighted-CE', weighted_avg_ce # 

    def _parse_best_params_from_csv(self):
        """
        (Internal Helper) Reads the best parameters and optimal rounds from CV results CSV.
        """
        results_df = pd.read_csv(self.results_save_path)
        best_params_series = results_df.iloc[0] # Assumes sorted ascending by score
        parsed_params = {}
        optimal_estimators = None

        for key, value in best_params_series.items():
            if key == 'optimal_boost_rounds': # Note: key name updated
                optimal_estimators = int(round(pd.to_numeric(value)))
            elif key == 'mean_cv_score':
                continue
            else:
                # Attempt parsing (handles numbers, lists stored as strings)
                try:
                    evaluated = ast.literal_eval(str(value))
                    if isinstance(evaluated, (int, float)):
                        if evaluated == int(evaluated): parsed_params[key] = int(evaluated)
                        else: parsed_params[key] = float(evaluated)
                    else:
                         parsed_params[key] = evaluated
                except (ValueError, SyntaxError):
                    try:
                       num_val = pd.to_numeric(value)
                       if num_val == int(num_val): parsed_params[key] = int(num_val)
                       else: parsed_params[key] = num_val
                    except ValueError:
                         parsed_params[key] = value # Keep as string if all else fails

        return parsed_params, optimal_estimators

    def cross_validate(self,
                       dh: 'DataHandler',
                       param_grid: Dict,
                       max_finalists: int = 10,
                       early_stopping_rounds: int = 30,
                       num_boost_rounds: int = 150, 
                       save: bool = True,
                       ):
        """
        Performs Cross-Validation for XGBoost using xgb.cv and early stopping
        with the custom weighted cross-entropy objective and evaluation metric.
        """       
        metric_name = 'weighted-CE' # Name used in eval
        results_list = []
        best_val_loss = float('inf')
        best_params = {}
        optimal_rounds = float('inf')
        dtrain = dh.get_xgb_data('train') 
        config_list = list(ParameterGrid(param_grid))
        print(f"\n--- Starting Cross-Validation for {self.model_name.upper()}. Testing   {len(config_list)} configs ---")
        print(f"Using device {DEVICE}.")
        print("--------------------------------------")
        start_time = time.time()
        for i, config in enumerate(config_list, 1):
            config_start_time = time.time() 
            params_for_cv = {**config,
                            #  'objective': 'weighted-CE',
                            #  'num_target': 4,
                             'n_jobs': -1, # Use all available cores
                             'disable_default_eval_metric': True, # Disable default eval metric
                             }
            # add gpu_hist param if cuda is available
            if DEVICE.type == 'cuda':
                params_for_cv['tree_method'] = 'hist'
                params_for_cv['device'] = DEVICE
                

            #NOTE: setting shuffle=False ensures that the folds are exactly same as what's used by other models
            cv_results_df = xgb.cv(
                params=params_for_cv,
                dtrain=dtrain,
                num_boost_round=num_boost_rounds,
                nfold=3, 
                obj = self.weighted_softprob_obj, # Custom objective
                custom_metric=self.weighted_cross_entropy_eval,
                maximize=False, # Minimize weighted-CE
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False, # No output
                shuffle=False,
            )
            # print(cv_results_df)
            time_taken = time.time() - config_start_time
            train_loss = cv_results_df[f'train-{metric_name}-mean'].min()
            val_loss = cv_results_df[f'test-{metric_name}-mean'].min()
            config_rounds = cv_results_df[f'test-{metric_name}-mean'].idxmin() + 1

            result_entry = {**config,
                            'Train loss': train_loss,
                            'Val Loss': val_loss,
                            'Optimal rounds': config_rounds,
                            'Time taken': f"{time_taken:.2f}s"
                            }
            results_list.append(result_entry)
            print(f"Config: {str(i).rjust(3)} | Train loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Optimal Rounds: {config_rounds} | Time taken: {time_taken:.2f}s")

            # Update overall best score, params, and rounds if this config is better
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = config
                optimal_rounds = config_rounds

        self.best_params = best_params
        self.optimal_boost_rounds = optimal_rounds

        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values(by='Val Loss', ascending=True).reset_index(drop=True)

        if save:
            results_df.to_csv(self.results_save_path, index=False)
            print(f"CV results saved to: {self.results_save_path}")

        time_taken = time.time() - start_time
        mins = int(time_taken) // 60
        secs = int(time_taken) % 60
        print(f"XGB cross-Validation completed in {mins}m {secs}s.")

        return results_df.head(max_finalists) #return only top max_finalists

    def train_final_model(self, dh: 'DataHandler'):
        """
        Trains the final XGBoost model using the best hyperparameters and
        the optimal number of boosting rounds determined by cross-validation.
        """
        print(f"\n--- Starting Final Model Training for {self.model_name.upper()} ---")

        # 1. Load best hyperparameters and optimal rounds if not already set
        if not self.best_params or self.optimal_boost_rounds is None:
             loaded_params, loaded_rounds = self._parse_best_params_from_csv()
             self.best_params = loaded_params
             self.optimal_boost_rounds = loaded_rounds

        print(f"Using best hyperparameters from CV: {self.best_params}")
        print(f"Using optimal boosting rounds from CV: {self.optimal_boost_rounds}")

        # Prepare parameters for xgb.train
        final_train_params = {
            **self.best_params,
            'n_jobs': -1, # Use all available cores
        }

        # 2. Prepare the final training DMatrix
        dtrain = dh.get_xgb_data('train') 

        # 3. Fit the final model using xgb.train
        print(f"Fitting final model with {self.optimal_boost_rounds} boosting rounds...")
        start_time_train = time.time()
        bst = xgb.train(
            params=final_train_params,
            dtrain=dtrain,
            num_boost_round=self.optimal_boost_rounds, # Use exact rounds from CV
            custom_metric=self.weighted_cross_entropy_eval, # Needed if using evals list
            maximize=False,
            verbose_eval=10 # Print progress every 10 rounds if evals is used
        )
        end_time_train = time.time()
        print(f"Final model fitting complete ({end_time_train - start_time_train:.2f} seconds).")

        # 4. Save the trained Booster model
        self.model = bst
        bst.save_model(self.model_save_path)
        print(f"Saved final trained XGBoost model to: {self.model_save_path}")

    def load_model(self):
        """Loads a trained XGBoost Booster model from its saved file."""
        self.model = xgb.Booster()
        self.model.load_model(self.model_save_path)
        print(f"{self.model_name.upper()} model loaded successfully from {self.model_save_path}.")

    def predict(self,
                dh: 'DataHandler', # Assumes DataHandler class definition exists
                save: bool = False
               ):
        """
        Generates predictions (numpy array of shape [n_samples, 4]) for the test set using the trained XGBoost model. If `save` is True, saves the predictions to a CSV file. 
        """
        if self.model is None:
            self.load_model()

        dtest = dh.get_xgb_data('test') 
        X_test, y_test, _ = dh.get_ridge_data('test') 
        n_samples = X_test.shape[0]
        y_tots = y_test.sum(axis=1, keepdims=True) # P(18plus|C), shape [n_samples, 1]

        # Create final, scaled predictions
        y_pred = self.model.predict(dtest)
        y_pred = y_pred.reshape((n_samples, 4))
        y_pred = self.softmax(y_pred) 
        y_pred = y_pred * y_tots

        if save:
            pred_df = pd.DataFrame(y_pred, columns=dh.targets)
            pred_df.to_csv(self.pred_save_path, index=False)
            print(f"County-level scaled predictions saved to: {self.pred_save_path}")

        return y_pred

# ======================================

def evaluate_predictions(pred_dict: Dict[str, np.ndarray],
                         dh: DataHandler,
                         save: bool = False):
    """
    Evaluates and compares model predictions. `pred_dict` should be a dictionary
    where keys are model names (str) and values are numpy arrays containing the predictions. If `save` is True, the resulting evaluation DataFrame is saved to a CSV file.

    Returns:
        pd.DataFrame: A DataFrame summarizing the evaluation. 
        Columns:
        - 'P(democrat)', 'P(other)', 'P(republican)', 'P(non_voter)': Aggregate shares.
        - 'P(underage)': Calculated as 1 minus the sum of the four target shares.
        - 'Cross-entropy': Cross-entropy against true distribution.
        - 'KL Div': Kullback-Leibler divergence against true distribution.
        - 'KL Div%': Percentage of KL divergence relative to true self-entropy.
    """
    print("\n--- Evaluating Model Predictions ---")

    _, y_true, wts = dh.get_ridge_data('test') 
    pred_dict['true'] = y_true

    for key, y in pred_dict.items():
        y = np.clip(y, 1e-9, 1.0) # For RidgeModel
        y = (wts * y).sum(axis=0) 
        y = np.append(y,1.0 - np.sum(y)) # Append P(underage)
        pred_dict[key] = list(y)

    y_true = pred_dict['true']
    ce_true = -np.sum(y_true * np.log(y))

    for key, y in pred_dict.items():
        ce = -np.sum(y_true * np.log(y))
        kl_div = ce - ce_true
        kl_div_percent = (kl_div / ce_true) * 100
        pred_dict[key] = list(y) + [ce, kl_div, kl_div_percent] 

    target_cols = [t.replace('|C)', ')') for t in dh.targets] + ['P(underage)', 'Cross-entropy', 'KL Div', 'KL Div%']
    eval_df = pd.DataFrame.from_dict(pred_dict, orient='index', columns=target_cols)
    eval_df = eval_df.sort_values(by='KL Div', ascending=True)

    if save:
        eval_save_path = os.path.join(RESULTS_DIR, f"Final_evaluation_{dh.test_year}.csv")
        eval_df.to_csv(eval_save_path, index=True, index_label='model')
        print(f"\nEvaluation summary saved to: {eval_save_path}")

    return eval_df