# =============================================================================
# Model Handlers - Election Prediction Project
# =============================================================================
"""
Provides high-level handler classes for different model types (Ridge, XGBoost, NN).
These classes encapsulate the workflow for each model, including triggering HPO,
training the final model, loading/saving models, and generating predictions.
They aim to provide a consistent interface for use in the main project script.
"""

import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib
from typing import List, Dict, Tuple, Any, Type, Union, Optional, Callable

# Imports from other modules
from utils import DEVICE, targets, ACTIVATION_MAP, OPTIMIZER_MAP # Removed SCHEDULER_MAP - not used directly here
from data_handling import DataHandler # Type hints only needed
from core_nn import DynamicMLP
# Import HPO classes for type hints or if needed directly
from hpo import HyperparameterTuner, ObjectiveNN, ObjectiveXGBoost
from metrics import weighted_cross_entropy_loss, weighted_mse_xgb


# =============================================================================
# Ridge Model Handler
# =============================================================================
class RidgeModel:
    """Handles Ridge Regression CV, training, loading, and prediction."""
    MODEL_NAME = "ridge"

    def __init__(self, model_dir: str, results_dir: str):
        """Initializes the RidgeModel handler."""
        self.model: Union[Ridge, None] = None
        self.best_alpha: Union[float, None] = None
        # Construct paths relative to provided directories
        self.model_save_path = os.path.join(model_dir, f"{self.MODEL_NAME}_final_model.joblib")
        self.cv_results_save_path = os.path.join(results_dir, f"{self.MODEL_NAME}_cv_results.csv")
        # Ensure directories exist upon instantiation or before saving
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)


    def cross_validate(self,
                       dh: 'DataHandler', # Use DataHandler type hint
                       param_grid: List[float] # Made required argument
                       ) -> None:
        """Performs grid search CV for Ridge, saves results, stores best alpha."""
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        if not param_grid:
            raise ValueError("Parameter grid (param_grid) cannot be empty.")

        results_list = []
        best_score = float('inf')
        current_best_alpha = None

        for alpha in param_grid:
            print(f"  Testing alpha = {alpha}:")
            fold_val_losses = []
            # Use NumPy data directly from DataHandler
            for j, ((X_train, y_train, wts_train), (X_val, y_val, wts_val)) in enumerate(dh.cv_data, 1):
                model_fold = Ridge(alpha=alpha)
                # Use weights directly (ravel if needed by sklearn version)
                model_fold.fit(X_train, y_train, sample_weight=wts_train.ravel())
                y_pred_val = model_fold.predict(X_val)
                # Use weights directly
                val_loss = mean_squared_error(y_val, y_pred_val, sample_weight=wts_val.ravel())

                # Get fold years info correctly from dh.folds
                train_yrs, val_yr = dh.folds[j-1]
                fold_str = f" Fold {j} - Train: {train_yrs} - Val: {val_yr} - "
                print(fold_str + f"Val Loss (W-MSE): {val_loss:.6f}")
                fold_val_losses.append(val_loss)

            mean_val_loss = np.mean(fold_val_losses) if fold_val_losses else float('inf')
            print(f"    Avg Val Loss (W-MSE): {mean_val_loss:.6f}")
            results_list.append({'alpha': alpha, 'mean_cv_score': mean_val_loss})

            if mean_val_loss < best_score:
                best_score = mean_val_loss
                current_best_alpha = alpha

        self.best_alpha = current_best_alpha
        results_df = pd.DataFrame(results_list).sort_values(by='mean_cv_score')
        results_df.to_csv(self.cv_results_save_path, index=False)
        print(f"CV results saved to: {self.cv_results_save_path}")
        if self.best_alpha is not None:
            print(f"\nBest {self.MODEL_NAME.upper()} CV alpha: {self.best_alpha} (Score: {best_score:.6f})")
        else:
             print(f"\n{self.MODEL_NAME.upper()} CV finished, but no best alpha found (check grid/data).")
        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")

    def train_final_model(self, dh: 'DataHandler') -> Ridge:
        """Trains the final Ridge model using the best alpha from CV results."""
        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")
        if self.best_alpha is None:
            try:
                results_df = pd.read_csv(self.cv_results_save_path)
                self.best_alpha = results_df.iloc[0]['alpha']
            except (FileNotFoundError, IndexError, KeyError):
                 raise RuntimeError(f"Best alpha not found. Run cross_validate or check {self.cv_results_save_path}.")
        print(f"Using best alpha: {self.best_alpha}")

        self.model = Ridge(alpha=self.best_alpha)
        (X_train, y_train, wts_train), _ = dh.final_data
        print(f"Loaded final training data: {X_train.shape[0]} samples.")

        self.model.fit(X_train, y_train, sample_weight=wts_train.ravel())
        print("Ridge model fitting complete.")

        joblib.dump(self.model, self.model_save_path)
        print(f"Saved final trained Ridge model to: {self.model_save_path}")
        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        return self.model

    def load_model(self) -> Ridge:
        """Loads a trained Ridge model."""
        print(f"Loading trained Ridge model from: {self.model_save_path}")
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(f"Model file not found at {self.model_save_path}")
        self.model = joblib.load(self.model_save_path)
        print("Ridge model loaded successfully.")
        return self.model

    def predict(self, dh: 'DataHandler', save: bool = False, results_dir: Optional[str] = None) -> pd.DataFrame:
        """Generates raw predictions for the test set using the trained Ridge model."""
        print(f"Generating predictions for {self.MODEL_NAME.upper()} on year {dh.test_year}...")
        if self.model is None: self.load_model()

        _, (X_test, _, _) = dh.final_data
        y_pred_raw = self.model.predict(X_test)
        pred_df = pd.DataFrame(y_pred_raw, columns=targets) # Use global 'targets'

        if save:
            if results_dir is None:
                print("Warning: 'save' is True but 'results_dir' not provided. Skipping save.")
            else:
                pred_save_path = os.path.join(results_dir, f"{self.MODEL_NAME}_{dh.test_year}_predictions.csv")
                os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                pred_df.to_csv(pred_save_path, index=False)
                print(f"  Raw predictions saved to: {pred_save_path}")

        print(f"Finished generating {self.MODEL_NAME.upper()} predictions.")
        return pred_df

# =============================================================================
# XGBoost Model Handler
# =============================================================================
class XGBoostModel:
    """Handles XGBoost HPO, training, loading, and prediction."""
    MODEL_NAME = "xgboost"

    def __init__(self, model_dir: str, results_dir: str):
        """Initializes the XGBoostModel handler."""
        self.model: Optional[xgb.XGBRegressor] = None
        self.best_params: Optional[Dict[str, Any]] = None
        # Use native format for model, json for best HPO params
        self.model_save_path = os.path.join(model_dir, f"{self.MODEL_NAME}_final_model.json")
        self.best_params_save_path = os.path.join(results_dir, f"{self.MODEL_NAME}_best_params.json")
        # Ensure directories exist upon instantiation or before saving
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    def cross_validate(self,
                       dh: 'DataHandler',
                       tuner: 'HyperparameterTuner',
                       objective_xgb: 'ObjectiveXGBoost', # Use ObjectiveXGBoost hint
                       n_trials: int) -> Dict[str, Any]:
        """Performs Optuna CV using HyperparameterTuner."""
        print(f"\n--- Starting Cross-Validation (Optuna) for {self.MODEL_NAME.upper()} ---")
        tuner.tune(objective=objective_xgb, n_trials=n_trials, direction='minimize') # Assume minimizing weighted MSE

        self.best_params = tuner.get_best_params()
        print(f"\nBest {self.MODEL_NAME.upper()} CV params found.")
        print(f"Best {self.MODEL_NAME.upper()} CV score (weighted_mse): {tuner.get_best_value():.6f}")

        # Save best HPO parameters
        with open(self.best_params_save_path, 'w') as f:
             json.dump(self.best_params, f, indent=2)
        print(f"Saved best HPO parameters to: {self.best_params_save_path}")

        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")
        return self.best_params

    def _load_best_params_if_needed(self):
        """Loads best HPO parameters from file if not already set."""
        if self.best_params is None:
            print(f"Attempting to load best HPO parameters from: {self.best_params_save_path}")
            if not os.path.exists(self.best_params_save_path):
                raise FileNotFoundError(f"Best HPO parameters file not found at {self.best_params_save_path}. Run cross_validate first.")
            try:
                with open(self.best_params_save_path, 'r') as f:
                    self.best_params = json.load(f)
                print("Loaded best HPO parameters from file.")
            except Exception as e:
                raise IOError(f"Error loading best HPO parameters from {self.best_params_save_path}: {e}")


    def train_final_model(self,
                          dh: 'DataHandler',
                          final_fit_params: Optional[Dict[str, Any]] = None
                         ) -> xgb.XGBRegressor:
        """Trains the final XGBoost model using best HPO parameters."""
        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")
        self._load_best_params_if_needed()
        if self.best_params is None: raise RuntimeError("Failed to load best HPO params.")
        print(f"Using best HPO parameters: {self.best_params}")

        fixed_model_params = {
             'objective': 'reg:squarederror',
             'eval_metric': weighted_mse_xgb, # Use static method
             # 'tree_method': 'gpu_hist', # Optional: if using GPU
        }
        # Don't use n_estimators from HPO; rely on early stopping
        current_best_params = self.best_params.copy()
        current_best_params.pop('n_estimators', None)
        fit_config = final_fit_params or {}
        n_estimators_final = fit_config.get('n_estimators', 1000) # Default high value
        early_stopping_rounds_final = fit_config.get('early_stopping_rounds', 50) # Default patience

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators_final,
            **fixed_model_params,
            **current_best_params # Pass HPO params
            )

        (X_train, y_train, wts_train), (X_test, y_test, wts_test) = dh.final_data
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=wts_train)
        dtest = xgb.DMatrix(X_test, label=y_test, weight=wts_test)

        fit_kwargs = {
            'evals': [(dtest, 'validation')],
            'verbose': False,
        }
        if early_stopping_rounds_final > 0:
             fit_kwargs['early_stopping_rounds'] = early_stopping_rounds_final

        print(f"Loaded final training data: {X_train.shape[0]} samples.")
        print(f"Fitting final model with up to {n_estimators_final} estimators (early stopping patience={early_stopping_rounds_final} on weighted_mse)...")

        self.model.fit(dtrain, **fit_kwargs)
        print("XGBoost model fitting complete.")

        eval_results = self.model.evals_result()
        if early_stopping_rounds_final > 0 and 'validation' in eval_results and 'weighted_mse' in eval_results['validation']:
            best_score_final = min(eval_results['validation']['weighted_mse'])
            print(f"  Best iteration: {self.model.best_iteration}, Best validation score (weighted_mse): {best_score_final:.6f}")
        else:
             print("  Early stopping possibly not triggered or results not available.")

        self.model.save_model(self.model_save_path)
        print(f"Saved final trained XGBoost model to: {self.model_save_path}")
        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        return self.model

    def load_model(self) -> xgb.XGBRegressor:
        """Loads a trained XGBoost model from native JSON format."""
        print(f"Loading trained XGBoost model from: {self.model_save_path}")
        if not os.path.exists(self.model_save_path):
             raise FileNotFoundError(f"Model file not found at {self.model_save_path}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(self.model_save_path)
        print("XGBoost model loaded successfully.")
        return self.model

    def predict(self, dh: 'DataHandler', save: bool = False, results_dir: Optional[str] = None) -> pd.DataFrame:
        """Generates raw predictions for the test set using the trained XGBoost model."""
        print(f"Generating predictions for {self.MODEL_NAME.upper()} on year {dh.test_year}...")
        if self.model is None: self.load_model()

        _, (X_test, _, _) = dh.final_data
        y_pred_raw = self.model.predict(X_test)
        pred_df = pd.DataFrame(y_pred_raw, columns=targets) # Use global 'targets'

        if save:
            if results_dir is None:
                print("Warning: 'save' is True but 'results_dir' not provided. Skipping save.")
            else:
                 pred_save_path = os.path.join(results_dir, f"{self.MODEL_NAME}_{dh.test_year}_predictions.csv")
                 os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                 pred_df.to_csv(pred_save_path, index=False)
                 print(f"  Raw predictions saved to: {pred_save_path}")

        print(f"Finished generating {self.MODEL_NAME.upper()} predictions.")
        return pred_df

# =============================================================================
# Neural Network (NN) Model Handler
# =============================================================================
class NNModel:
    """Handles NN (DynamicMLP) HPO, training, loading, and prediction."""
    MODEL_NAME = "NN"

    def __init__(self, input_dim: int, model_dir: str, results_dir: str):
        """Initializes the NNModel handler."""
        self.input_dim = input_dim
        self.model: Optional[DynamicMLP] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.fixed_params: Optional[Dict[str, Any]] = None # Store fixed HPO params
        self.final_loss_history: List[Dict[str, Any]] = []

        model_name_lower = self.MODEL_NAME.lower()
        self.best_params_save_path = os.path.join(results_dir, f"{model_name_lower}_best_params.json")
        self.model_save_path = os.path.join(model_dir, f"{model_name_lower}_final_model.pth")
        self.loss_save_path = os.path.join(results_dir, f"{model_name_lower}_final_training_loss.csv")
        # Ensure directories exist upon instantiation or before saving
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    def _load_best_params_if_needed(self):
        """Loads best HPO parameters and fixed params from file if not set."""
        if self.best_params is None:
            print(f"Attempting to load best parameters from: {self.best_params_save_path}")
            if not os.path.exists(self.best_params_save_path):
                raise FileNotFoundError(f"Best parameters file not found at {self.best_params_save_path}. Run cross_validate first.")
            try:
                with open(self.best_params_save_path, 'r') as f:
                    saved_data = json.load(f)
                    if isinstance(saved_data, dict) and 'best_params' in saved_data and 'fixed_params' in saved_data:
                         self.best_params = saved_data['best_params']
                         self.fixed_params = saved_data['fixed_params']
                         print("Loaded best_params and fixed_params from file.")
                    else: # Assume old format
                         self.best_params = saved_data
                         self.fixed_params = {} # Initialize as empty dict
                         print("Loaded best_params (old format). Fixed HPO params might be missing.")
            except Exception as e:
                 raise IOError(f"Error loading best parameters from {self.best_params_save_path}: {e}")

    def cross_validate(self,
                       dh: 'DataHandler',
                       tuner: 'HyperparameterTuner',
                       objective_nn: 'ObjectiveNN', # Use ObjectiveNN hint
                       n_trials: int) -> Dict[str, Any]:
        """Performs Optuna CV using HyperparameterTuner."""
        print(f"\n--- Starting Cross-Validation (Optuna) for {self.MODEL_NAME.upper()} ---")
        self.fixed_params = objective_nn.fixed_params # Store fixed params used

        tuner.tune(objective=objective_nn, n_trials=n_trials, direction='minimize') # NN Objective returns loss

        self.best_params = tuner.get_best_params()
        print(f"\nBest {self.MODEL_NAME.upper()} CV params found.")
        print(f"Best {self.MODEL_NAME.upper()} CV score (Loss): {tuner.get_best_value():.6f}")

        # Save best HPO params and the fixed params used during HPO
        save_data = {'best_params': self.best_params, 'fixed_params': self.fixed_params}
        with open(self.best_params_save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved best parameters and fixed HPO params to: {self.best_params_save_path}")

        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")
        return self.best_params

    def train_final_model(self,
                          dh: 'DataHandler',
                          final_epochs: int = 150,
                          final_patience: int = 30
                         ) -> DynamicMLP:
        """Trains the final NN model using best HPO parameters."""
        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")
        self._load_best_params_if_needed()
        if self.best_params is None: raise RuntimeError("Failed to load best_params.")
        print(f"Using best hyperparameters: {self.best_params}")
        if self.fixed_params is None: print("Warning: Fixed params from HPO not found. Using defaults.")

        train_loader, val_loader = dh.final_dataloaders

        nn_params = self.best_params
        # Default to 2 hidden layers if fixed_params not loaded/available
        num_hidden_layers = (self.fixed_params or {}).get('num_hidden_layers', 2)
        hidden_layers = [nn_params[f"n_units_l{i}"] for i in range(num_hidden_layers)]
        activation_fn = ACTIVATION_MAP[nn_params['activation']]
        dropout_rate = nn_params['dropout_rate']
        self.model = DynamicMLP(self.input_dim, hidden_layers, activation_fn, dropout_rate).to(DEVICE)

        optimizer_cls = OPTIMIZER_MAP[nn_params['optimizer']]
        optimizer = optimizer_cls(self.model.parameters(), lr=nn_params['learning_rate'], weight_decay=nn_params['weight_decay'])
        scheduler = None # Optional: configure scheduler for final training if needed

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state_dict = None
        self.final_loss_history = []

        print(f"Starting final training for max {final_epochs} epochs (patience={final_patience})...")
        for epoch in range(final_epochs):
            self.model.train()
            epoch_train_loss = 0.0
            for features, targets_batch, weights in train_loader: # Renamed targets to avoid clash
                features, targets_batch, weights = features.to(DEVICE), targets_batch.to(DEVICE), weights.to(DEVICE)
                outputs = self.model(features)
                loss = weighted_cross_entropy_loss(outputs, targets_batch, weights)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_train_loss += loss.item()
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')

            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for features, targets_batch, weights in val_loader:
                    features, targets_batch, weights = features.to(DEVICE), targets_batch.to(DEVICE), weights.to(DEVICE)
                    outputs = self.model(features)
                    loss = weighted_cross_entropy_loss(outputs, targets_batch, weights)
                    epoch_val_loss += loss.item()
            avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

            self.final_loss_history.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
            if (epoch + 1) % 10 == 0 or epoch == final_epochs - 1:
                 print(f"  Epoch {epoch+1}/{final_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= final_patience:
                print(f"  Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
                break

            if scheduler: scheduler.step(avg_val_loss) # Assumes ReduceLROnPlateau like logic if used

        if best_model_state_dict:
            self.model.load_state_dict(best_model_state_dict)
            print(f"Loaded best model state from epoch {epoch+1-epochs_no_improve} (Val Loss: {best_val_loss:.6f}).")
        else:
            print("Warning: No improvement observed or no epochs run. Using model from last state.")
        self.model.eval() # Ensure model is in eval mode after training

        self._save_model()
        self._save_loss_history()

        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        return self.model

    def _save_model(self):
        """Saves the trained final model state dict."""
        if self.model is None: print("Error: No model instance to save."); return
        print(f"Saving final {self.MODEL_NAME} model state_dict to: {self.model_save_path}")
        torch.save(self.model.state_dict(), self.model_save_path)
        print("Model state_dict saved successfully.")

    def _save_loss_history(self):
        """Saves the final training loss history to CSV."""
        if not self.final_loss_history: print("Warning: No loss history to save."); return
        print(f"Saving final training loss history to: {self.loss_save_path}")
        loss_df = pd.DataFrame(self.final_loss_history)
        loss_df.to_csv(self.loss_save_path, index=False)
        print("Loss history saved successfully.")

    def load_model(self) -> DynamicMLP:
        """Loads a pre-trained NN model state dict."""
        print(f"Loading trained {self.MODEL_NAME} model from: {self.model_save_path}")
        if not os.path.exists(self.model_save_path):
             raise FileNotFoundError(f"Model file not found at {self.model_save_path}")

        self._load_best_params_if_needed() # Need params to reconstruct architecture
        if self.best_params is None: raise RuntimeError("Failed to load best_params for model reconstruction.")
        if self.fixed_params is None: print("Warning: Fixed params from HPO not found. Using defaults for layer count.")

        nn_params = self.best_params
        num_hidden_layers = (self.fixed_params or {}).get('num_hidden_layers', 2)
        hidden_layers = [nn_params[f"n_units_l{i}"] for i in range(num_hidden_layers)]
        activation_fn = ACTIVATION_MAP[nn_params['activation']]
        dropout_rate = nn_params['dropout_rate']
        # Instantiate model first, then load state dict
        self.model = DynamicMLP(self.input_dim, hidden_layers, activation_fn, dropout_rate)
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=DEVICE))
        self.model.to(DEVICE) # Ensure model is on the correct device
        self.model.eval()
        print(f"{self.MODEL_NAME} model loaded successfully and set to eval mode on {DEVICE}.")
        return self.model

    def predict(self, dh: 'DataHandler', save: bool = False, results_dir: Optional[str] = None) -> pd.DataFrame:
        """Generates scaled predictions using the trained NN model."""
        print(f"Generating predictions for {self.MODEL_NAME.upper()} on year {dh.test_year}...")
        if self.model is None: self.load_model() # Loads model to DEVICE and sets eval mode

        _, (X_test, y_test, _) = dh.final_data

        # Prepare tensors and move to model's device
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_tots_tensor = torch.tensor(y_test, dtype=torch.float32).sum(dim=1, keepdim=True).to(DEVICE)

        self.model.eval() # Redundant if load_model worked, but safe
        with torch.no_grad():
            outputs = self.model(X_test_tensor) # Raw probabilities [n_samples, 4] on DEVICE
            y_pred_scaled = outputs * y_tots_tensor # Scaled predictions [n_samples, 4] on DEVICE

        # Convert scaled predictions to DataFrame (move to CPU first)
        y_pred_scaled_np = y_pred_scaled.cpu().numpy()
        pred_df = pd.DataFrame(y_pred_scaled_np, columns=targets) # Use global 'targets'

        if save:
            if results_dir is None:
                 print("Warning: 'save' is True but 'results_dir' not provided. Skipping save.")
            else:
                 model_name_lower = self.MODEL_NAME.lower()
                 pred_save_path = os.path.join(results_dir, f"{model_name_lower}_{dh.test_year}_predictions.csv")
                 os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                 pred_df.to_csv(pred_save_path, index=False)
                 print(f"  Scaled predictions saved to: {pred_save_path}")

        print(f"Finished generating {self.MODEL_NAME.upper()} predictions.")
        return pred_df