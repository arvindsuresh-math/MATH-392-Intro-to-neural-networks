# =============================================================================
# Hyperparameter Optimization - Election Prediction Project
# =============================================================================
"""
Contains classes and functions related to hyperparameter optimization (HPO)
using Optuna. Includes configuration management for search spaces, the main
tuner orchestration class, and objective functions for specific models (NN, XGBoost).
"""

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader # Only DataLoader needed here
import xgboost as xgb
import optuna
from typing import List, Dict, Tuple, Any, Type, Union, Optional, Callable

# Imports from other modules
from utils import DEVICE, ACTIVATION_MAP, OPTIMIZER_MAP, SCHEDULER_MAP
from data_handling import DataHandler # Type hint only needed
from core_nn import DynamicMLP

# loss functions
from metrics import weighted_cross_entropy_loss, weighted_mse_xgb


# =============================================================================
# Hyperparameter Configuration Class
# =============================================================================
class HyperparameterConfig:
    """
    Manages hyperparameter search space configurations for different model types.

    Provides default search spaces and allows users to retrieve, update,
    and display the configurations for model types like Neural Networks ('NN'),
    Ridge Regression ('Ridge'), and XGBoost ('XGBoost').
    """
    def __init__(self):
        """Initializes with default search spaces."""
        default_nn_space = {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.1, "log": False},
            "dropout_rate": {"type": "float", "low": 0.0, "high": 0.7, "log": False},
            "optimizer": {"type": "categorical", "choices": ["AdamW", "Adam", "SGD"]},
            "activation": {"type": "categorical", "choices": ["ReLU", "Tanh"]},
            "n_units_l0": {"type": "int", "low": 8, "high": 128, "log": True},
            "n_units_l1": {"type": "int", "low": 8, "high": 128, "log": True},
        }
        default_ridge_space = {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True}
        }
        default_xgboost_space = {
            "eta": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 10, "log": False},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
            "gamma": {"type": "float", "low": 0.0, "high": 5.0, "log": False},
            "lambda": {"type": "float", "low": 1e-2, "high": 10.0, "log": True},
            "alpha": {"type": "float", "low": 1e-2, "high": 10.0, "log": True},
        }
        self.active_spaces = {
            "NN": default_nn_space.copy(),
            "Ridge": default_ridge_space.copy(), # Still useful for reference
            "XGBoost": default_xgboost_space.copy()
        }

    def get_space(self, model_type: str) -> Dict:
        """Retrieves the active hyperparameter search space."""
        if model_type not in self.active_spaces:
            raise KeyError(f"Model type '{model_type}' not recognized.")
        return self.active_spaces[model_type]

    def update_param_space(self, model_type: str, param_name: str, param_config: Dict):
        """Updates or adds a hyperparameter configuration."""
        if model_type not in self.active_spaces:
            raise KeyError(f"Model type '{model_type}' not recognized.")
        self.active_spaces[model_type][param_name] = param_config
        print(f"Updated '{param_name}' config for model type '{model_type}'.")

    def display_space(self, model_type: str):
        """Prints the current search space configuration."""
        if model_type not in self.active_spaces:
            raise KeyError(f"Model type '{model_type}' not recognized.")
        print(f"\n--- Active Search Space for '{model_type}' ---")
        print(json.dumps(self.active_spaces[model_type], indent=2))
        print("-" * (30 + len(model_type)))


# =============================================================================
# Objective Classes for Hyperparameter Optimization
# =============================================================================

class ObjectiveNN:
    """
    Optuna Objective function for Neural Network (DynamicMLP) models.

    Evaluates NN hyperparameters using cross-validation, weighted cross-entropy loss,
    and Optuna's pruning features. Uses pre-computed DataLoaders from DataHandler.
    """
    def __init__(self,
                 data_handler: 'DataHandler', # Use DataHandler type hint
                 search_space: Dict[str, Dict],
                 fixed_params: Dict[str, Any]):
        """Initializes the NN Objective."""
        self.data_handler = data_handler
        self.search_space = search_space
        self.fixed_params = fixed_params
        self.num_hidden_layers = self.fixed_params.get('num_hidden_layers', 2)

    def _suggest_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Suggests parameters for a trial based on the search_space config."""
        params = {}
        for name, config in self.search_space.items():
            param_type = config['type']
            suggest_kwargs = {k: v for k, v in config.items() if k != 'type'}
            if param_type == 'float': params[name] = trial.suggest_float(name, **suggest_kwargs)
            elif param_type == 'int': params[name] = trial.suggest_int(name, **suggest_kwargs)
            elif param_type == 'categorical': params[name] = trial.suggest_categorical(name, **suggest_kwargs)
        return params

    def _train_one_fold(self, trial, model, train_loader, val_loader, optimizer, scheduler, pruning_checkpoints, patience) -> float:
        """Trains one fold with pruning and early stopping."""
        best_val_loss = float('inf')
        epochs_no_improve = 0
        max_epochs_this_run = max(pruning_checkpoints) if pruning_checkpoints else self.fixed_params.get('max_epochs', 100)
        model.to(DEVICE)

        for epoch in range(max_epochs_this_run):
            model.train()
            for features, targets, weights in train_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                outputs = model(features)
                # Use the static method from NNModel (imported)
                loss = weighted_cross_entropy_loss(outputs, targets, weights)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for features, targets, weights in val_loader:
                    features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                    outputs = model(features)
                    # Use the static method from NNModel (imported)
                    loss = weighted_cross_entropy_loss(outputs, targets, weights)
                    val_loss_epoch += loss.item()
            avg_val_loss = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else float('inf')


            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience: break

            if (epoch + 1) in pruning_checkpoints:
                trial.report(avg_val_loss, step=epoch + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
        return best_val_loss

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Runs cross-validation for one NN trial."""
        suggested_params = self._suggest_params(trial)

        lr = suggested_params['learning_rate']
        weight_decay = suggested_params['weight_decay']
        dropout_rate = suggested_params['dropout_rate']
        optimizer_name = suggested_params['optimizer']
        activation_name = suggested_params['activation']
        optimizer_cls = OPTIMIZER_MAP[optimizer_name]
        activation_fn = ACTIVATION_MAP[activation_name]
        hidden_layers = [suggested_params[f"n_units_l{i}"] for i in range(self.num_hidden_layers)]

        scheduler_name = suggested_params.get("scheduler", "None")
        scheduler_params = {}
        if scheduler_name == "ReduceLROnPlateau":
             scheduler_params['factor'] = suggested_params.get('sched_factor', 0.1) # Provide default if not in space
             scheduler_params['patience'] = suggested_params.get('sched_patience', 10) # Provide default
             scheduler_params['mode'] = 'min'

        input_dim = self.data_handler.input_dim
        pruning_checkpoints = self.fixed_params.get('pruning_epochs', [10, 20, 40])
        patience = self.fixed_params.get('patience', 10)

        fold_validation_losses = []
        # Iterate through PRE-COMPUTED DataLoaders
        for fold_idx, (train_loader, val_loader) in enumerate(self.data_handler.cv_dataloaders):
            model = DynamicMLP(input_dim, hidden_layers, activation_fn, dropout_rate)
            optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

            scheduler = None
            if scheduler_name != "None":
                 scheduler_cls = SCHEDULER_MAP[scheduler_name]
                 scheduler = scheduler_cls(optimizer, **scheduler_params)

            try:
                best_fold_loss = self._train_one_fold(
                    trial, model, train_loader, val_loader, optimizer, scheduler,
                    pruning_checkpoints, patience
                )
                fold_validation_losses.append(best_fold_loss)
            except optuna.TrialPruned:
                return float('inf')

            del model, optimizer, scheduler
            if DEVICE.type in ['cuda', 'mps']:
                torch.cuda.empty_cache() if DEVICE.type == 'cuda' else torch.mps.empty_cache()

        mean_cv_loss = np.mean(fold_validation_losses) if fold_validation_losses else float('inf')
        return mean_cv_loss if not np.isnan(mean_cv_loss) else float('inf')


class ObjectiveXGBoost:
    """
    Optuna Objective function for XGBoost models.

    Evaluates XGBoost hyperparameters using cross-validation, weighted MSE
    as the evaluation metric, and Optuna's XGBoostPruningCallback.
    """
    def __init__(self,
                 data_handler: 'DataHandler', # Use DataHandler type hint
                 search_space: Dict[str, Dict],
                 fixed_params: Dict[str, Any]):
        """Initializes the XGBoost Objective."""
        self.data_handler = data_handler
        self.search_space = search_space
        self.fixed_params = fixed_params

    def _suggest_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Suggests parameters for a trial based on the search_space config."""
        params = {}
        for name, config in self.search_space.items():
             param_type = config['type']
             suggest_kwargs = {k: v for k, v in config.items() if k != 'type'}
             if param_type == 'float': params[name] = trial.suggest_float(name, **suggest_kwargs)
             elif param_type == 'int': params[name] = trial.suggest_int(name, **suggest_kwargs)
             elif param_type == 'categorical': params[name] = trial.suggest_categorical(name, **suggest_kwargs)
        return params

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Runs cross-validation for one XGBoost trial."""
        suggested_params = self._suggest_params(trial)

        n_estimators = self.fixed_params.get('n_estimators_max_pruning', 200)
        early_stopping_rounds = self.fixed_params.get('early_stopping_rounds', 20)

        fold_validation_metrics = []
        # Iterate through NumPy data from DataHandler
        for fold_idx, ((X_train, y_train, wts_train), (X_val, y_val, wts_val)) in enumerate(self.data_handler.cv_data):

            dtrain = xgb.DMatrix(X_train, label=y_train, weight=wts_train)
            dval = xgb.DMatrix(X_val, label=y_val, weight=wts_val)
            eval_set = [(dval, 'validation_0')]

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                # Use the static method from XGBoostModel (imported)
                eval_metric=weighted_mse_xgb,
                **suggested_params
            )

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial,
                observation_key='validation_0-weighted_mse' # Use the string name returned
            )

            try:
                model.fit(
                    dtrain,
                    evals=eval_set,
                    callbacks=[pruning_callback],
                    verbose=False
                )
                eval_results = model.evals_result()
                # Use the string name returned by the metric function
                best_score = min(eval_results['validation_0']['weighted_mse'])
                fold_validation_metrics.append(best_score)
            except optuna.TrialPruned:
                return float('inf')
            except Exception as e:
                 print(f"Warning: Trial {trial.number}, Fold {fold_idx+1} failed: {e}")
                 return float('inf')

        mean_cv_metric = np.mean(fold_validation_metrics) if fold_validation_metrics else float('inf')
        return mean_cv_metric if not np.isnan(mean_cv_metric) else float('inf')


# =============================================================================
# Hyperparameter Tuner Class
# =============================================================================
class HyperparameterTuner:
    """Orchestrates hyperparameter optimization using Optuna."""

    def __init__(self,
                 study_name: str = "election_hpo_study",
                 storage_path: Optional[str] = None, # e.g., "sqlite:///hpo_study.db"
                 pruner: Optional[optuna.pruners.BasePruner] = None):
        """Initializes the HyperparameterTuner."""
        self.study_name = study_name
        self.storage_path = storage_path
        self.pruner = pruner if pruner is not None else optuna.pruners.SuccessiveHalvingPruner()
        self.study: Optional[optuna.study.Study] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_value_: Optional[float] = None

    def tune(self,
             objective: Callable[[optuna.trial.Trial], float],
             n_trials: int,
             direction: str = 'minimize',
             load_if_exists: bool = False):
        """Runs the hyperparameter optimization process."""
        print(f"\n--- Starting Optuna HPO ---")
        print(f"Study Name: {self.study_name}, N Trials: {n_trials}, Pruner: {self.pruner.__class__.__name__}")
        print(f"Direction: {direction}, Storage: {'In-memory' if self.storage_path is None else self.storage_path}")

        self.study = optuna.create_study(
            study_name=self.study_name, storage=self.storage_path,
            load_if_exists=load_if_exists, direction=direction, pruner=self.pruner
        )

        try:
            self.study.optimize(objective, n_trials=n_trials)
        except KeyboardInterrupt: print("\nOptimization interrupted by user.")
        except Exception as e: print(f"\nAn error occurred during optimization: {e}"); raise

        if self.study.best_trial:
            self.best_params_ = self.study.best_params
            self.best_value_ = self.study.best_value
            print("\n--- Optuna HPO Finished ---")
            print(f"Best Trial: {self.study.best_trial.number}, Best Value: {self.best_value_:.6f}")
            print("Best Parameters:", json.dumps(self.best_params_, indent=2))
            print("-" * 30)
        else:
             print("\n--- Optuna HPO Finished ---"); print("No successful trials completed.")
             self.best_params_ = {}
             self.best_value_ = float('inf') if direction == 'minimize' else float('-inf')

    def get_best_params(self) -> Dict[str, Any]:
        """Returns the best hyperparameters found."""
        if self.best_params_ is None: raise RuntimeError("Tuning not run or no trials succeeded.")
        return self.best_params_

    def get_best_value(self) -> float:
        """Returns the best objective value achieved."""
        if self.best_value_ is None: raise RuntimeError("Tuning not run or no trials succeeded.")
        return self.best_value_

    def save_study_results(self, filepath: str):
        """Saves essential study results (best params, value) to a JSON file."""
        if self.study is None or self.best_params_ is None:
            print("Warning: Cannot save results, tuning not run or no trials succeeded.")
            return

        print(f"Saving study summary results to: {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        results_summary = {
            "study_name": self.study_name,
            "direction": self.study.direction.name,
            "best_trial_number": self.study.best_trial.number if self.study.best_trial else None,
            "best_value": self.best_value_,
            "best_params": self.best_params_,
            "n_trials_completed": len(self.study.trials),
        }
        try:
            with open(filepath, 'w') as f: json.dump(results_summary, f, indent=2)
            print("Study summary saved successfully.")
        except Exception as e: print(f"Error saving study summary: {e}")

    def save_plots(self, results_dir: str):
        """Generates and saves Optuna visualization plots to the specified directory."""
        if not self.study: print("Warning: Study not available for plotting."); return
        if not os.path.exists(results_dir): os.makedirs(results_dir, exist_ok=True)
        try:
            # Import plotting libraries only when needed
            import matplotlib
            import plotly
            figures = [
                optuna.visualization.plot_optimization_history(self.study),
                optuna.visualization.plot_param_importances(self.study),
                optuna.visualization.plot_slice(self.study),
            ]
            plot_names = ['optimization_history', 'param_importances', 'slice_plot']
            print(f"Saving Optuna plots to: {results_dir}")
            for fig, name in zip(figures, plot_names):
                try:
                    fig.write_image(os.path.join(results_dir, f"{self.study_name}_{name}.png"))
                except Exception as e:
                     print(f"  Could not save plot '{name}': {e}") # More specific error
            print(f"Finished saving plots.")
        except ImportError:
            print("Warning: Cannot save plots. Install plotly and kaleido (pip install plotly kaleido).")
        except Exception as e:
            print(f"Error generating Optuna plots: {e}")