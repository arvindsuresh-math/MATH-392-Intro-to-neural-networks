# =============================================================================
# Presidential Election Outcome Prediction - Framework Code
# Author: [Your Name/Lab Name]
# Date: [Current Date]
#
# =============================================================================
#
# --- High-Level Overview ---
# This script provides a comprehensive framework for predicting US presidential
# election outcomes at the county level using demographic features. The goal is
# to predict the conditional probability of voting for Democrat, Republican,
# Other candidates, or being a Non-voter, given county demographics P(Outcome|C).
# The framework supports multiple modeling approaches and incorporates automated
# hyperparameter optimization (HPO).
#
# --- Modeling Approaches Included ---
# 1.  Ridge Regression: Implemented using scikit-learn. HPO uses simple grid search.
# 2.  Neural Networks (MLP): Implemented using PyTorch with a flexible `DynamicMLP`
#     class allowing variable hidden layers. HPO uses Optuna.
# 3.  XGBoost: Implemented using the XGBoost library. HPO uses Optuna.
#
# --- Code Structure ---
# The script is organized into the following main sections/classes:
#
#   0. Imports and Global Constants: Loads libraries and defines constants
#      (paths, device, mappings, feature lists).
#   1. HyperparameterConfig: Manages default and user-modifiable search spaces
#      for HPO for different model types.
#   2. WeightedStandardScaler: A utility for standard scaling with sample weights.
#   3. DataHandler: Loads, preprocesses (scaling per fold), and splits data
#      for cross-validation and final training/testing.
#   4. DynamicMLP: A flexible PyTorch `nn.Module` for creating MLPs with
#      varying architectures (used by NNModel).
#   5. Objective Classes (ObjectiveNN, ObjectiveXGBoost): Defines the functions
#      called by Optuna to evaluate a single hyperparameter trial using
#      cross-validation for NNs and XGBoost, respectively. Includes pruning logic.
#   6. HyperparameterTuner: Orchestrates the Optuna HPO process (study creation,
#      optimization execution, result retrieval).
#   7. Model Handler Classes (RidgeModel, XGBoostModel, NNModel): High-level
#      interfaces for each model type. They manage the workflow including:
#      - Triggering HPO via HyperparameterTuner (for NN, XGBoost) or handling
#        simple HPO internally (Ridge).
#      - Training the final model using the best hyperparameters.
#      - Saving and loading the trained model.
#      - Evaluating the model on test data using a consistent (aggregate CE) or
#        model-specific metric.
#
# --- Typical Workflow (in a Jupyter Notebook) ---
# 1.  Instantiate `HyperparameterConfig`, modify search spaces if desired.
# 2.  Instantiate `DataHandler` (specifying `test_year`).
# 3.  Instantiate `HyperparameterTuner` (configure study name, storage, pruner).
# 4.  Instantiate necessary `Objective` classes (e.g., `ObjectiveNN`, `ObjectiveXGBoost`),
#     passing the `DataHandler`, relevant search space from `HyperparameterConfig`,
#     and any fixed training parameters.
# 5.  Instantiate the desired model handlers (e.g., `ridge = RidgeModel()`,
#     `xgb_model = XGBoostModel()`, `nn_model = NNModel(input_dim=dh.input_dim)`).
# 6.  Run HPO: `nn_model.cross_validate(dh, tuner, objective_nn, n_trials=...)`.
#     (Ridge HPO is self-contained: `ridge.cross_validate(dh)`).
# 7.  Train Final Model: `nn_model.train_final_model(dh, ...)`.
# 8.  Evaluate: `results_df = nn_model.evaluate(dh)`.
# 9.  Compare evaluation results across different model handlers.
#
# --- File/Directory Structure ---
# - ./data/final_dataset.csv : Expected location of the input dataset.
# - ./models/ : Directory where trained models are saved.
# - ./results/ : Directory where HPO results (best params, summaries) and
#                evaluation outputs (predictions, aggregate scores) are saved.
#
# --- Key Features & Concepts ---
# - Weighted Scaling: Uses `WeightedStandardScaler` to account for county population (`P(C)`)
#   during feature scaling, applied correctly within CV folds.
# - Weighted Loss (NN): Uses a custom `weighted_cross_entropy_loss` static method
#   within `NNModel` for training NNs, weighting sample losses by `P(C)`.
# - Dynamic NN Architecture: `DynamicMLP` allows creating NNs with varying depth/width.
# - Hyperparameter Optimization: Leverages `Optuna` for efficient HPO for NN/XGBoost,
#   including pruning (Successive Halving/Hyperband via `HyperparameterTuner`).
# - External Configuration: `HyperparameterConfig` allows easy modification of HPO
#   search spaces from the calling notebook/script.
# - Consistent Interface: Model handlers aim for a similar API (`cross_validate`,
#   `train_final_model`, `load_model`, `evaluate`) for ease of use.
# - Aggregate Evaluation: Provides a consistent (though potentially model-specific
#   in interpretation) aggregate cross-entropy evaluation metric for comparing
#   national vote share predictions.
#
# =============================================================================

# =============================================================================
# Imports and Global Constants
# =============================================================================

# --- Standard Library ---
import os
import json
import copy
from typing import List, Dict, Tuple, Any, Type, Union, Optional, Callable
from abc import ABC, abstractmethod # Keep if using any abstract base classes (even if implicitly)

# --- Core Data Science / ML Libraries ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error # Used by Ridge CV / XGBoost Eval
import joblib

# --- HPO Library ---
import optuna

# --- File Paths ---
# It's often better to define these paths in the execution environment (e.g., notebook)
# or load from a config file, but defining them here is also common.
# Ensure these directories exist or are created by the script/notebook.
DATA_DIR = "./data"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Device Selection ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Default Training Hyperparameters ---
BATCH_SIZE: int = 64
# Note: Default HPO params/grids are now handled by HyperparameterConfig

# --- Mappings for Dynamic Instantiation ---
ACTIVATION_MAP = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid}
OPTIMIZER_MAP = {'AdamW': optim.AdamW, 'Adam': optim.Adam, 'SGD': optim.SGD, 'RMSprop': optim.RMSprop}
SCHEDULER_MAP = {'StepLR': optim.lr_scheduler.StepLR, 'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau}

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
# Hyperparameter Configuration Class
# =============================================================================

class HyperparameterConfig:
    """
    Manages hyperparameter search space configurations for different model types.

    Provides default search spaces and allows users to retrieve, update,
    and display the configurations for model types like Neural Networks ('NN'),
    Ridge Regression ('Ridge'), and XGBoost ('XGBoost').

    The search space for a parameter is defined by a dictionary specifying its
    'type' (e.g., 'float', 'int', 'categorical'), range ('low', 'high'),
    choices ('choices'), and distribution properties ('log').
    """
    def __init__(self):
        """Initializes with default search spaces."""
        # --- Default Search Spaces ---
        default_nn_space = {
            # Example Search Space for DynamicMLP
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.1, "log": False},
            "dropout_rate": {"type": "float", "low": 0.0, "high": 0.7, "log": False},
            "optimizer": {"type": "categorical", "choices": ["AdamW", "Adam", "SGD"]},
            "activation": {"type": "categorical", "choices": ["ReLU", "Tanh"]},
            # Example: assuming num_hidden_layers is fixed, suggest units per layer
            "n_units_l0": {"type": "int", "low": 8, "high": 128, "log": True},
            "n_units_l1": {"type": "int", "low": 8, "high": 128, "log": True},
            # Add more layers (l2, l3...) if tuning deeper networks
            # Example: Optional scheduler tuning (can be simplified by removing)
            # "scheduler": {"type": "categorical", "choices": ["None", "ReduceLROnPlateau"]},
            # "sched_factor": {"type": "float", "low": 0.1, "high": 0.8, "log": False}, # Only if ReduceLROnPlateau
            # "sched_patience": {"type": "int", "low": 3, "high": 15, "log": False},   # Only if ReduceLROnPlateau
        }
        default_ridge_space = {
            # Note: Ridge HPO will use its own simple grid search,
            # but we keep a placeholder config.
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True}
        }
        default_xgboost_space = {
            # Example Search Space for XGBoost
            "eta": {"type": "float", "low": 0.01, "high": 0.3, "log": True}, # learning_rate
            "max_depth": {"type": "int", "low": 3, "high": 10, "log": False},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False}, # Fraction of samples
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False}, # Fraction of features
            "gamma": {"type": "float", "low": 0.0, "high": 5.0, "log": False}, # Min loss reduction for split
            "lambda": {"type": "float", "low": 1e-2, "high": 10.0, "log": True}, # L2 regularization
            "alpha": {"type": "float", "low": 1e-2, "high": 10.0, "log": True}, # L1 regularization
            # n_estimators is handled by early stopping/pruning
        }

        # --- Store Active Configurations ---
        # Use .copy() to avoid modifying defaults unintentionally later
        self.active_spaces = {
            "NN": default_nn_space.copy(),
            "Ridge": default_ridge_space.copy(),
            "XGBoost": default_xgboost_space.copy()
        }

    def get_space(self, model_type: str) -> Dict:
        """
        Retrieves the active hyperparameter search space for a given model type.

        Args:
            model_type (str): The type of model ('NN', 'Ridge', 'XGBoost').

        Returns:
            Dict: The dictionary defining the search space configuration.

        Raises:
            KeyError: If the model_type is not found.
        """
        if model_type not in self.active_spaces:
            raise KeyError(f"Model type '{model_type}' not recognized. "
                           f"Available types: {list(self.active_spaces.keys())}")
        return self.active_spaces[model_type]

    def update_param_space(self, model_type: str, param_name: str, param_config: Dict):
        """
        Updates or adds a hyperparameter configuration within a model type's space.

        Args:
            model_type (str): The type of model ('NN', 'Ridge', 'XGBoost').
            param_name (str): The name of the hyperparameter (e.g., 'learning_rate').
            param_config (Dict): A dictionary defining the new search configuration
                                 for this parameter (e.g., {'type': 'float', ...}).

        Raises:
            KeyError: If the model_type is not found.
        """
        if model_type not in self.active_spaces:
            raise KeyError(f"Model type '{model_type}' not recognized.")
        # Add or overwrite the parameter's configuration
        self.active_spaces[model_type][param_name] = param_config
        print(f"Updated '{param_name}' config for model type '{model_type}'.")

    def display_space(self, model_type: str):
        """
        Prints the current hyperparameter search space configuration for a model type.

        Args:
            model_type (str): The type of model ('NN', 'Ridge', 'XGBoost').

        Raises:
            KeyError: If the model_type is not found.
        """
        if model_type not in self.active_spaces:
            raise KeyError(f"Model type '{model_type}' not recognized.")
        print(f"\n--- Active Search Space for '{model_type}' ---")
        # Use json.dumps for pretty-printing the dictionary
        print(json.dumps(self.active_spaces[model_type], indent=2))
        print("-" * (30 + len(model_type)))

# =============================================================================
# WeightedStandardScaler Class
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
# DataHandler Class
# =============================================================================

class DataHandler:
    """Handles loading data and creating PyTorch DataLoaders."""

    def __init__(self, 
                 test_year: int = 2020, 
                 features_to_drop = []):
        """
        Initializes the DataHandler and determines input/output dimensions.

        Args:
            test_year: Year to use for test data (default: 2020).
            features_to_drop: List of features to drop from the dataset (default: []).
            targets_to_drop: List of target columns to drop from the dataset (default: []).
        
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
                  fit_years: List[int], # list of years to fit StandardScalar
                  transform_year: int, # year to transform StandardScaler
                  ):
        data = pd.read_csv('data/final_dataset.csv')

        # make datasets with fit years and transform years
        df_fit = data[data['year'].isin(fit_years)].reset_index(drop=True)
        df_transform = data[data['year'] == transform_year].reset_index(drop=True)

        # make wts arrays
        wts_fit = df_fit['P(C)']
        wts_transform = df_transform['P(C)']

        # make y's (arrays of shape [n_samples, 4])
        y_fit = df_fit[targets].values
        y_transform = df_transform[targets].values

        # make X's (arrays of shape [n_samples, n_features])
        X_fit = df_fit[self.features].values
        X_transform = df_transform[self.features].values

        #apply weighted StandardScaler to X's
        scaler = WeightedStandardScaler()
        X_fit = scaler.fit_transform(X_fit, wts_fit.ravel())
        X_transform = scaler.transform(X_transform)

        return (X_fit, y_fit, wts_fit), (X_transform, y_transform, wts_transform)

    def _create_dataset(self, data: Tuple[np.ArrayLike, np.ArrayLike, np.Series]) -> TensorDataset:
        """Converts pandas DataFrames to a PyTorch TensorDataset."""
        X_tensor = torch.tensor(data[0], dtype=torch.float32)
        y_tensor = torch.tensor(data[1], dtype=torch.float32)
        wts_tensor = torch.tensor(data[2], dtype=torch.float32) # Already [:, 1] shape
        return TensorDataset(X_tensor, y_tensor, wts_tensor)

    def _create_dataloaders(self, train_data, val_data) -> Tuple[DataLoader, DataLoader]:
        """Creates DataLoaders for training and validation sets."""

        # create tensor datasets
        train_dataset = self._create_dataset(train_data)
        val_dataset = self._create_dataset(val_data)

        # set the manual seed to 42
        torch.manual_seed(42)

        train_loader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False)
        
        return train_loader, val_loader

# =============================================================================
# Dynamic MLP Class (Template for PyTorch MLP)
# =============================================================================

class DynamicMLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron (MLP) PyTorch module.

    This class constructs an MLP with a specified number of hidden layers,
    sizes for each hidden layer, activation function, and dropout rate.
    It dynamically builds the network architecture based on the provided
    parameters, allowing for easy creation of models ranging from simple
    Softmax Regression (no hidden layers) to deep MLPs. The final layer
    always applies a Softmax activation to produce probability distributions.

    Attributes:
        network (nn.Sequential): The sequential container holding the main
            layers of the network (Linear, Activation, Dropout) up to the
            final linear layer producing logits.
        softmax (nn.Softmax): The final Softmax activation layer applied
            to the logits.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_layers: List[int],
                 activation_fn: Type[nn.Module],
                 dropout_rate: float):
        """
        Initializes the DynamicMLP module.

        Args:
            input_dim (int): The number of features in the input data.
            hidden_layers (List[int]): A list where each element is an integer
                representing the number of neurons in a hidden layer. The
                order defines the network structure. An empty list [] creates
                a Softmax Regression model (no hidden layers).
            activation_fn (Type[nn.Module]): The PyTorch activation function
                *class* to use after each hidden linear layer (e.g., nn.ReLU,
                nn.Tanh, nn.Sigmoid).
            dropout_rate (float): The dropout probability to apply after each
                activation function in the hidden layers. Should be between
                0.0 and 1.0.
        """
        super().__init__() # Call the parent class constructor

        # --- Build the layer sequence dynamically ---
        layers = []
        current_dim = input_dim # Start with the input dimension

        # --- Handle Hidden Layers (if any) ---
        if hidden_layers:
            # First hidden layer
            layers.append(nn.Linear(current_dim, hidden_layers[0]))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_layers[0] # Update dimension for the next layer

            # Subsequent hidden layers (if more than one)
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(current_dim, hidden_layers[i]))
                layers.append(activation_fn())
                layers.append(nn.Dropout(dropout_rate))
                current_dim = hidden_layers[i] # Update dimension

        # --- Final Linear Layer (Logits) ---
        # Connects the last hidden layer (or input if no hidden layers) to the output units
        layers.append(nn.Linear(current_dim, 4)) # Assuming 4 output classes: Democrat, Republican, Other, Non-voter

        # --- Create the main network sequence (excluding final Softmax) ---
        # nn.Sequential automatically passes the output of one layer to the next
        self.network = nn.Sequential(*layers)

        # --- Define the final Softmax activation separately ---
        # This makes it explicit that the 'network' produces logits
        # and Softmax converts them to probabilities. dim=1 applies Softmax
        # across the class dimension for each sample in a batch.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: The output tensor of shape [batch_size, 4]
                          containing class probabilities after Softmax.
        """
        # Pass input through the sequential layers to get logits
        logits = self.network(x)
        # Apply Softmax to convert logits to probabilities
        probabilities = self.softmax(logits)
        return probabilities

# =============================================================================
# Objective Classes for Hyperparameter Optimization
# =============================================================================

# Reminder: ObjectiveRidge is not implemented here, assuming the existing
# RidgeModel.cross_validate() method handles Ridge HPO separately.

class ObjectiveNN:
    """
    Optuna Objective function for Neural Network (DynamicMLP) models.

    Evaluates NN hyperparameters using cross-validation, weighted cross-entropy loss,
    and Optuna's pruning features.
    """
    def __init__(self,
                 data_handler: 'DataHandler',
                 search_space: Dict[str, Dict],
                 fixed_params: Dict[str, Any]):
        """
        Initializes the NN Objective.

        Args:
            data_handler (DataHandler): Instance holding the CV data and input_dim.
            search_space (Dict[str, Dict]): Configuration defining the tunable NN parameters.
            fixed_params (Dict[str, Any]): Fixed parameters like 'num_hidden_layers', 'pruning_epochs', 'patience'.
        """
        self.data_handler = data_handler
        self.search_space = search_space
        self.fixed_params = fixed_params
        # Pre-calculate fixed number of hidden layers for easier suggestion loop
        self.num_hidden_layers = self.fixed_params.get('num_hidden_layers', 2)

    def _suggest_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Suggests parameters for a trial based on the search_space config."""
        params = {}
        for name, config in self.search_space.items():
            param_type = config['type']
            # Create kwargs dict excluding 'type'
            suggest_kwargs = {k: v for k, v in config.items() if k != 'type'}

            if param_type == 'float':
                params[name] = trial.suggest_float(name, **suggest_kwargs)
            elif param_type == 'int':
                params[name] = trial.suggest_int(name, **suggest_kwargs)
            elif param_type == 'categorical':
                params[name] = trial.suggest_categorical(name, **suggest_kwargs)
            # Add other types if needed
        return params

    def _train_one_fold(self, trial, model, train_loader, val_loader, optimizer, scheduler, pruning_checkpoints, patience) -> float:
        """Trains one fold with pruning. (Identical to previous version)."""
        best_val_loss = float('inf')
        epochs_no_improve = 0
        max_epochs_this_run = max(pruning_checkpoints) if pruning_checkpoints else self.fixed_params.get('max_epochs', 100)
        model.to(DEVICE)

        for epoch in range(max_epochs_this_run):
            # Training
            model.train()
            for features, targets, weights in train_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                outputs = model(features)
                loss = NNModel.weighted_cross_entropy_loss(outputs, targets, weights)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Validation
            model.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for features, targets, weights in val_loader:
                    features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                    outputs = model(features)
                    loss = NNModel.weighted_cross_entropy_loss(outputs, targets, weights)
                    val_loss_epoch += loss.item()
            avg_val_loss = val_loss_epoch / len(val_loader)

            # Early Stopping & Best Loss Update
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience: break

            # Pruning Check
            if (epoch + 1) in pruning_checkpoints:
                trial.report(avg_val_loss, step=epoch + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned() # Raise exception for pruning

            # Scheduler Step (optional)
            if scheduler:
                 if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                 else:
                    scheduler.step() # Step per epoch
        return best_val_loss

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Runs cross-validation for one NN trial."""
        # --- 1. Suggest Hyperparameters dynamically ---
        suggested_params = self._suggest_params(trial)

        # Extract specific parameters needed for model/optimizer setup
        lr = suggested_params['learning_rate']
        weight_decay = suggested_params['weight_decay']
        dropout_rate = suggested_params['dropout_rate']
        optimizer_name = suggested_params['optimizer']
        activation_name = suggested_params['activation']
        optimizer_cls = OPTIMIZER_MAP[optimizer_name]
        activation_fn = ACTIVATION_MAP[activation_name]

        hidden_layers = [suggested_params[f"n_units_l{i}"] for i in range(self.num_hidden_layers)]

        # Optional: Handle scheduler params if tuning them
        scheduler_name = suggested_params.get("scheduler", "None") # Default to None if not in space
        scheduler_params = {}
        if scheduler_name == "ReduceLROnPlateau":
             scheduler_params['factor'] = suggested_params['sched_factor']
             scheduler_params['patience'] = suggested_params['sched_patience']
             scheduler_params['mode'] = 'min'

        # --- Get fixed parameters ---
        input_dim = self.data_handler.input_dim
        pruning_checkpoints = self.fixed_params.get('pruning_epochs', [10, 20, 40])
        patience = self.fixed_params.get('patience', 10)

        # --- 2. Cross-Validation Loop ---
        fold_validation_losses = []
        for fold_idx, (train_data, val_data) in enumerate(self.data_handler.cv_data):
            # Data prep (as before)
            X_train_df, y_train_df, wts_train_df = train_data
            X_val_df, y_val_df, wts_val_df = val_data
            X_train = torch.tensor(X_train_df.values, dtype=torch.float32); y_train = torch.tensor(y_train_df.values, dtype=torch.float32); wts_train = torch.tensor(wts_train_df.values, dtype=torch.float32)
            X_val = torch.tensor(X_val_df.values, dtype=torch.float32); y_val = torch.tensor(y_val_df.values, dtype=torch.float32); wts_val = torch.tensor(wts_val_df.values, dtype=torch.float32)
            train_dataset = TensorDataset(X_train, y_train, wts_train); val_dataset = TensorDataset(X_val, y_val, wts_val)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True); val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Build model
            model = DynamicMLP(input_dim, hidden_layers, activation_fn, dropout_rate)
            optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Setup scheduler (optional)
            scheduler = None
            if scheduler_name != "None":
                 scheduler_cls = SCHEDULER_MAP[scheduler_name]
                 scheduler = scheduler_cls(optimizer, **scheduler_params)

            # Train fold
            try:
                best_fold_loss = self._train_one_fold(
                    trial, model, train_loader, val_loader, optimizer, scheduler,
                    pruning_checkpoints, patience
                )
                fold_validation_losses.append(best_fold_loss)
            except optuna.TrialPruned:
                # If pruning exception is caught, trial is automatically marked as pruned
                return float('inf') # Return high value

            # Memory cleanup
            del model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
            if DEVICE.type in ['cuda', 'mps']:
                torch.cuda.empty_cache() if DEVICE.type == 'cuda' else torch.mps.empty_cache()

        # --- 3/4. Aggregate and Return ---
        mean_cv_loss = np.mean(fold_validation_losses)
        return mean_cv_loss if not np.isnan(mean_cv_loss) else float('inf')

class ObjectiveXGBoost:
    """
    Optuna Objective function for XGBoost models.

    Evaluates XGBoost hyperparameters using cross-validation, RMSE as the
    evaluation metric during training, and Optuna's XGBoostPruningCallback.
    """
    def __init__(self,
                 data_handler: 'DataHandler',
                 search_space: Dict[str, Dict],
                 fixed_params: Dict[str, Any]):
        """
        Initializes the XGBoost Objective.

        Args:
            data_handler (DataHandler): Instance holding the CV data.
            search_space (Dict[str, Dict]): Config defining tunable XGBoost parameters.
            fixed_params (Dict[str, Any]): Fixed parameters like 'n_estimators' (max boosting rounds), 'early_stopping_rounds'.
        """
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
        # --- 1. Suggest Hyperparameters ---
        suggested_params = self._suggest_params(trial)

        # --- Get fixed parameters ---
        # Max rounds for pruning; final model might use more/less with early stopping
        n_estimators = self.fixed_params.get('n_estimators_max_pruning', 200)
        early_stopping_rounds = self.fixed_params.get('early_stopping_rounds', 20) # Patience for XGBoost

        # --- 2. Cross-Validation Loop ---
        fold_validation_rmses = []
        for fold_idx, ((X_train, y_train, wts_train), (X_val, y_val, wts_val)) in enumerate(self.data_handler.cv_data):
            # --- b. Build Model ---
            # Use multioutput regression capabilities of XGBoost
            model = xgb.XGBRegressor(
                objective='reg:squarederror', # Standard objective for regression
                n_estimators=n_estimators,    # Max rounds for this CV fold
                early_stopping_rounds=early_stopping_rounds,
                eval_metric='rmse',           # Metric for early stopping/pruning
                # tree_method='gpu_hist',     # Uncomment if using GPU
                **suggested_params           # Add tunable parameters
            )

            # --- c. Setup Pruning Callback ---
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial,
                observation_key='validation_0-rmse' # Monitor RMSE on the validation set
            )

            # --- d. Training with Pruning/Early Stopping ---
            try:
                model.fit(
                    X_train, y_train,
                    sample_weight=wts_train,
                    eval_set=[(X_val, y_val)], # Validation set for early stopping & pruning
                    sample_weight_eval_set=[wts_val], # Optional: weight validation loss
                    callbacks=[pruning_callback],
                    verbose=False # Suppress XGBoost training output during HPO
                )
                # Store the best RMSE achieved for this fold
                fold_validation_rmses.append(model.best_score)

            except optuna.TrialPruned:
                return float('inf') # Return high value if pruned
            except Exception as e:
                 # Catch potential XGBoost errors during fit
                 print(f"Warning: Trial {trial.number}, Fold {fold_idx+1} failed: {e}")
                 return float('inf') # Treat failed folds as bad trials


        # --- 3/4. Aggregate and Return ---
        # We want to minimize RMSE
        mean_cv_rmse = np.mean(fold_validation_rmses)
        return mean_cv_rmse if not np.isnan(mean_cv_rmse) else float('inf')

# =============================================================================
# Hyperparameter Tuner Class
# =============================================================================

class HyperparameterTuner:
    """
    Orchestrates hyperparameter optimization using Optuna.

    This class sets up and runs an Optuna study to find the best hyperparameters
    for a given model type by repeatedly calling a provided objective function.
    It handles study creation, pruner configuration, optimization execution,
    and retrieving the best results.

    Attributes:
        study_name (str): Name for the Optuna study.
        storage_path (Optional[str]): Path to save/load the study database
                                      (e.g., 'sqlite:///my_study.db'). If None,
                                      an in-memory study is used.
        pruner (optuna.pruners.BasePruner): The pruner instance to use for
                                            early stopping of unpromising trials.
        study (optuna.study.Study): The Optuna study object after tuning.
        best_params_ (Dict[str, Any]): Dictionary of the best hyperparameters found.
        best_value_ (float): The best objective value achieved.
    """
    def __init__(self,
                 study_name: str = "election_hpo_study",
                 storage_path: Optional[str] = None, # e.g., "sqlite:///hpo_study.db"
                 pruner: Optional[optuna.pruners.BasePruner] = None):
        """
        Initializes the HyperparameterTuner.

        Args:
            study_name (str): A name for the Optuna study.
            storage_path (Optional[str]): Database URL for study persistence.
                If None, study is in-memory only. Example: 'sqlite:///study.db'.
            pruner (Optional[optuna.pruners.BasePruner]): An Optuna pruner instance.
                If None, defaults to SuccessiveHalvingPruner.
        """
        self.study_name = study_name
        self.storage_path = storage_path
        # Default to SuccessiveHalvingPruner if none provided
        self.pruner = pruner if pruner is not None else optuna.pruners.SuccessiveHalvingPruner()

        self.study: Optional[optuna.study.Study] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_value_: Optional[float] = None

    def tune(self,
             objective: Callable[[optuna.trial.Trial], float],
             n_trials: int,
             direction: str = 'minimize',
             load_if_exists: bool = False):
        """
        Runs the hyperparameter optimization process.

        Args:
            objective (Callable[[optuna.trial.Trial], float]): The objective
                function (or callable class instance) to be optimized. It must
                take an optuna.trial.Trial object as input and return a float score.
            n_trials (int): The number of trials to run.
            direction (str): Direction of optimization ('minimize' or 'maximize').
            load_if_exists (bool): If True, tries to load an existing study with
                the same name from the storage path, otherwise creates a new one.
        """
        print(f"\n--- Starting Optuna HPO ---")
        print(f"Study Name: {self.study_name}")
        print(f"Number of Trials: {n_trials}")
        print(f"Pruner: {self.pruner.__class__.__name__}")
        print(f"Direction: {direction}")
        print(f"Storage: {'In-memory' if self.storage_path is None else self.storage_path}")

        # Create or load the Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_path,
            load_if_exists=load_if_exists,
            direction=direction,
            pruner=self.pruner
        )

        # Run the optimization
        try:
            self.study.optimize(
                objective,
                n_trials=n_trials,
                # n_jobs=-1, # Use all available CPU cores for parallel trials (if objective is safe)
                # Be cautious with n_jobs > 1 if using GPU or objective has shared state issues.
                # timeout=3600 # Optional: Set a time limit in seconds
            )
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
        except Exception as e:
             print(f"\nAn error occurred during optimization: {e}")
             # Optionally re-raise or handle more gracefully
             raise

        # Store the best results found
        if self.study.best_trial:
            self.best_params_ = self.study.best_params
            self.best_value_ = self.study.best_value
            print("\n--- Optuna HPO Finished ---")
            print(f"Best Trial Number: {self.study.best_trial.number}")
            print(f"Best Value ({direction}): {self.best_value_:.6f}")
            print("Best Parameters:")
            for key, value in self.best_params_.items():
                print(f"  {key}: {value}")
            print("-" * 30)
        else:
             print("\n--- Optuna HPO Finished ---")
             print("No successful trials completed.")
             print("-" * 30)
             self.best_params_ = {}
             self.best_value_ = float('inf') if direction == 'minimize' else float('-inf')

    def get_best_params(self) -> Dict[str, Any]:
        """Returns the best hyperparameters found during tuning."""
        if self.best_params_ is None:
            raise RuntimeError("Tuning has not been run or no successful trials completed.")
        return self.best_params_

    def get_best_value(self) -> float:
        """Returns the best objective value achieved during tuning."""
        if self.best_value_ is None:
             raise RuntimeError("Tuning has not been run or no successful trials completed.")
        return self.best_value_

    def save_study_results(self, filepath: Optional[str] = None):
        """
        Saves essential study results (like best params, value) to a file.

        Note: Saving the full Optuna study requires using a persistent storage
        (like SQLite or PostgreSQL) specified in `storage_path` during init.
        This method saves a summary if persistent storage wasn't used.

        Args:
            filepath (Optional[str]): Path to save the summary results (e.g., JSON).
                If None, uses a default path in RESULTS_DIR.
        """
        if self.study is None or self.best_params_ is None:
            print("Warning: Cannot save results, tuning not run or no successful trials.")
            return

        if filepath is None:
            # Define a default path if none provided
            filepath = os.path.join(RESULTS_DIR, f"{self.study_name}_summary.json")

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
            with open(filepath, 'w') as f:
                json.dump(results_summary, f, indent=2)
            print("Study summary saved successfully.")
        except Exception as e:
            print(f"Error saving study summary: {e}")

    def save_plots(self, results_dir: str = RESULTS_DIR):
        """Generates and saves Optuna visualization plots."""
        if not self.study:
            print("Warning: Study not available for plotting.")
            return
        try:
            import matplotlib # Import only when needed
            figures = [
                optuna.visualization.plot_optimization_history(self.study),
                optuna.visualization.plot_param_importances(self.study),
                optuna.visualization.plot_slice(self.study),
                # Add more plots as needed, e.g., plot_contour
            ]
            plot_names = ['optimization_history', 'param_importances', 'slice_plot']
            os.makedirs(results_dir, exist_ok=True)
            for fig, name in zip(figures, plot_names):
                fig.write_image(os.path.join(results_dir, f"{self.study_name}_{name}.png"))
            print(f"Saved Optuna plots to: {results_dir}")
        except ImportError:
            print("Warning: Cannot save plots. Please install matplotlib and plotly.")
        except Exception as e:
            print(f"Error generating Optuna plots: {e}")

# =============================================================================
# Model handler Classes
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
                model_fold.fit(X_train, y_train, sample_weight=wts_train.ravel())
                # Predict on validation data
                y_pred_val = model_fold.predict(X_val)

                # Use weighted MSE (weights = P(C) = wts_val) to compute validation loss
                weights_val = wts_val.squeeze() # Ensure weights is 1D array
                val_loss = mean_squared_error(y_val, y_pred_val, sample_weight=weights_val.ravel())

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
        self.model.fit(X_train, y_train, sample_weight=wts_train.ravel())
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
        print("Ridge model loaded successfully.")
        return self.model

    def predict(self,
                dh: DataHandler,
                save: bool = False) -> pd.DataFrame:
        """
        Generates raw predictions for the test set using the trained Ridge model.

        Args:
            dh (DataHandler): The DataHandler instance containing test data.
            save (bool, optional): If True, saves the predictions to a CSV file.
                                   Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing county-level raw predictions
                          for the four target classes (shape: [n_samples, 4]).
        """
        print(f"Generating predictions for {self.MODEL_NAME.upper()} on year {dh.test_year}...")

        # 1. Load model if not already loaded/trained
        if self.model is None:
            self.load_model()

        # 2. Get test features (NumPy array)
        _, (X_test, _, _) = dh.final_data # Only need features for prediction

        # 3. Perform prediction (outputs raw scores)
        y_pred_raw = self.model.predict(X_test) # Shape: [n_samples, 4]

        # 4. Create DataFrame
        pred_df = pd.DataFrame(y_pred_raw, columns=targets) # Use global 'targets' list

        # 5. Optionally save
        if save:
            pred_save_path = os.path.join(RESULTS_DIR, f"{self.MODEL_NAME}_{dh.test_year}_predictions.csv")
            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
            pred_df.to_csv(pred_save_path, index=False)
            print(f"  Raw predictions saved to: {pred_save_path}")

        print(f"Finished generating {self.MODEL_NAME.upper()} predictions.")
        return pred_df

class XGBoostModel:
    """
    Handles XGBoost model HPO orchestration (via Tuner/Objective),
    training, loading, and evaluation.
    """
    MODEL_NAME = "xgboost"

    def __init__(self):
        """Initializes the XGBoostModel handler."""
        self.model: Optional[xgb.XGBRegressor] = None
        self.best_params: Optional[Dict[str, Any]] = None
        # Store paths for convenience
        self.results_save_path = os.path.join(RESULTS_DIR, f"{self.MODEL_NAME}_cv_results.json") # Optuna study often saved as DB/JSON
        self.model_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_final_model.json") # XGBoost native save format
        # self.model_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_final_model.joblib") # Alternative save format

    def cross_validate(self,
                       dh: 'DataHandler',
                       tuner: 'HyperparameterTuner', # Assumes Tuner class exists
                       objective_xgb: ObjectiveXGBoost,
                       n_trials: int) -> Dict[str, Any]:
        """
        Performs Cross-Validation using Optuna via HyperparameterTuner.

        Args:
            dh (DataHandler): Data handler instance.
            tuner (HyperparameterTuner): The HPO orchestrator instance.
            objective_xgb (ObjectiveXGBoost): The objective function configured for XGBoost.
            n_trials (int): The number of HPO trials to run.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        print(f"\n--- Starting Cross-Validation (Optuna) for {self.MODEL_NAME.upper()} ---")

        # Run the tuning process
        tuner.tune(objective=objective_xgb, n_trials=n_trials) # Pruner defined within Tuner

        # Get and store the best parameters found by the tuner
        self.best_params = tuner.get_best_params()
        print(f"\nBest {self.MODEL_NAME.upper()} CV params: {self.best_params}")
        print(f"Best {self.MODEL_NAME.upper()} CV score (RMSE): {tuner.best_value_:.6f}")

        # Optionally save the Optuna study details (handled by Tuner.save_study)
        # tuner.save_study(self.results_save_path)
        # print(f"Optuna study results potentially saved by Tuner.")

        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")
        return self.best_params

    def train_final_model(self,
                          dh: 'DataHandler',
                          final_fit_params: Optional[Dict[str, Any]] = None
                         ) -> xgb.XGBRegressor:
        """
        Trains the final XGBoost model using the best parameters from CV.

        Args:
            dh (DataHandler): The data handler instance.
            final_fit_params (Optional[Dict[str, Any]]): Optional parameters for the
                final `.fit()` call (e.g., {'early_stopping_rounds': 50}).

        Returns:
            xgb.XGBRegressor: The trained final model.
        """
        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")

        # 1. Ensure best_params are available (e.g., loaded if not just run CV)
        if self.best_params is None:
            # Logic to load best params from saved study results if needed
            # Placeholder: Assume results_save_path contains simple JSON of best params
            try:
                with open(os.path.join(RESULTS_DIR, f"{self.MODEL_NAME}_best_params.json"), 'r') as f:
                     self.best_params = json.load(f)
                print(f"Loaded best params from file: {self.best_params}")
            except FileNotFoundError:
                 raise ValueError("Best params not found. Run cross_validate first or ensure results are saved.")
        else:
             print(f"Using best params from CV: {self.best_params}")

        # 2. Build model instance with best params + any fixed ones
        # Add objective, eval_metric etc. that weren't part of the tuning space
        fixed_model_params = {
             'objective': 'reg:squarederror',
             'eval_metric': 'rmse',
             # 'tree_method': 'gpu_hist', # If using GPU
             # 'n_estimators': 1000 # Set a high initial value, rely on early stopping
        }
        # Important: n_estimators found during HPO might not be optimal for the full dataset
        # It's often better to set a high number here and use early stopping.
        # Remove n_estimators from best_params if it was tuned, or override it.
        self.best_params.pop('n_estimators', None) # Remove if tuned, otherwise no effect
        n_estimators_final = final_fit_params.pop('n_estimators', 1000) if final_fit_params else 1000


        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators_final,
            **fixed_model_params,
            **self.best_params # Tuned parameters override others if names clash
            )

        # 3. Load combined training data and optional validation data for early stopping
        (X_train, y_train, wts_train), (X_test, y_test, wts_test) = dh.final_data

        fit_params = {
            'eval_set': [(X_test, y_test)],
            'sample_weight': wts_train,
            'verbose': False, # Can set to True or a number to see progress
        }
        # Add specific early stopping rounds for final training if provided
        if final_fit_params and 'early_stopping_rounds' in final_fit_params:
             fit_params['early_stopping_rounds'] = final_fit_params['early_stopping_rounds']


        print(f"Loaded final training data: {X_train.shape[0]} samples.")
        print(f"Fitting final model with n_estimators={n_estimators_final} (early stopping enabled)...")

        # 4. Fit the model
        self.model.fit(X_train, y_train, **fit_params)
        print("XGBoost model fitting complete.")
        print(f"  Best iteration: {self.model.best_iteration}, Best score (RMSE): {self.model.best_score:.6f}")


        # 5. Save the trained model (using native JSON format)
        self.model.save_model(self.model_save_path)
        # Or use joblib: joblib.dump(self.model, self.model_save_path)
        print(f"Saved final trained XGBoost model to: {self.model_save_path}")

        # Optionally save just the best parameters separately for easy access
        best_params_path = os.path.join(RESULTS_DIR, f"{self.MODEL_NAME}_best_params.json")
        with open(best_params_path, 'w') as f:
             json.dump(self.best_params, f, indent=2)
        print(f"Saved best parameters to: {best_params_path}")


        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        return self.model

    def load_model(self) -> xgb.XGBRegressor:
        """Loads a trained XGBoost model."""
        print(f"Loading trained XGBoost model from: {self.model_save_path}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(self.model_save_path)
        # Or use joblib: self.model = joblib.load(self.model_save_path)
        print("XGBoost model loaded successfully.")
        return self.model

    def predict(self,
                dh: DataHandler,
                save: bool = False) -> pd.DataFrame:
        """
        Generates raw predictions for the test set using the trained XGBoost model.

        Args:
            dh (DataHandler): The DataHandler instance containing test data.
            save (bool, optional): If True, saves the predictions to a CSV file.
                                   Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing county-level raw predictions
                          for the four target classes (shape: [n_samples, 4]).
        """
        print(f"Generating predictions for {self.MODEL_NAME.upper()} on year {dh.test_year}...")

        # 1. Load model if not already loaded/trained
        if self.model is None:
            self.load_model()

        # 2. Get test features (NumPy array)
        _, (X_test, _, _) = dh.final_data # Only need features

        # 3. Perform prediction (outputs raw scores)
        y_pred_raw = self.model.predict(X_test) # Shape: [n_samples, 4]

        # 4. Create DataFrame
        pred_df = pd.DataFrame(y_pred_raw, columns=targets) # Use global 'targets' list

        # 5. Optionally save
        if save:
            pred_save_path = os.path.join(RESULTS_DIR, f"{self.MODEL_NAME}_{dh.test_year}_predictions.csv")
            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
            pred_df.to_csv(pred_save_path, index=False)
            print(f"  Raw predictions saved to: {pred_save_path}")

        print(f"Finished generating {self.MODEL_NAME.upper()} predictions.")
        return pred_df

class NNModel:
    """
    Handles Neural Network (DynamicMLP) HPO orchestration, training, loading,
    and evaluation, providing a consistent interface similar to other model types.
    """
    MODEL_NAME = "NN" # Can be customized (e.g., "MLP") if needed

    def __init__(self,
                 input_dim: int):
        """
        Initializes the NNModel handler.

        Args:
            input_dim (int): The number of input features for the NN.
        """
        self.input_dim = input_dim
        self.model: Optional[DynamicMLP] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.fixed_params: Optional[Dict[str, Any]] = None # Store fixed params used during HPO
        self.final_loss_history: List[Dict[str, Any]] = []

        # Define paths
        model_name_lower = self.MODEL_NAME.lower()
        self.best_params_save_path = os.path.join(RESULTS_DIR, f"{model_name_lower}_best_params.json")
        self.model_save_path = os.path.join(MODELS_DIR, f"{model_name_lower}_final_model.pth")
        self.loss_save_path = os.path.join(RESULTS_DIR, f"{model_name_lower}_final_training_loss.csv")

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

    def _load_best_params_if_needed(self):
        """Loads best_params from file if they are not already set."""
        if self.best_params is None:
            print(f"Attempting to load best parameters from: {self.best_params_save_path}")
            if not os.path.exists(self.best_params_save_path):
                raise FileNotFoundError(f"Best parameters file not found at {self.best_params_save_path}. "
                                        "Run cross_validate first or ensure the file exists.")
            try:
                with open(self.best_params_save_path, 'r') as f:
                    saved_data = json.load(f)
                    # Compatibility: Handle simple dict or dict with fixed_params
                    if isinstance(saved_data, dict) and 'best_params' in saved_data and 'fixed_params' in saved_data:
                         self.best_params = saved_data['best_params']
                         self.fixed_params = saved_data['fixed_params'] # Load fixed params too
                         print("Loaded best_params and fixed_params from file.")
                    else:
                         # Assume old format (just best_params)
                         self.best_params = saved_data
                         print("Loaded best_params from file (assuming old format). Fixed params might be missing.")

            except Exception as e:
                 raise IOError(f"Error loading best parameters from {self.best_params_save_path}: {e}")

    def cross_validate(self,
                       dh: 'DataHandler',
                       tuner: 'HyperparameterTuner',
                       objective_nn: 'ObjectiveNN', # Pass the instantiated ObjectiveNN
                       n_trials: int) -> Dict[str, Any]:
        """
        Performs Cross-Validation using Optuna via HyperparameterTuner.

        Args:
            dh (DataHandler): Data handler instance.
            tuner (HyperparameterTuner): The HPO orchestrator instance.
            objective_nn (ObjectiveNN): The objective function configured for NNs.
            n_trials (int): The number of HPO trials to run.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        print(f"\n--- Starting Cross-Validation (Optuna) for {self.MODEL_NAME.upper()} ---")

        # Store the fixed params used by the objective for later use
        self.fixed_params = objective_nn.fixed_params

        # Run the tuning process
        # Ensure the tuner is set to minimize (ObjectiveNN returns loss)
        tuner.tune(objective=objective_nn, n_trials=n_trials, direction='minimize')

        # Get and store the best parameters found by the tuner
        self.best_params = tuner.get_best_params()
        print(f"\nBest {self.MODEL_NAME.upper()} CV params: {self.best_params}")
        print(f"Best {self.MODEL_NAME.upper()} CV score (Loss): {tuner.best_value_:.6f}")

        # Save best params and the fixed params used during HPO for reproducibility
        print(f"Saving best parameters to: {self.best_params_save_path}")
        os.makedirs(os.path.dirname(self.best_params_save_path), exist_ok=True)
        save_data = {
            'best_params': self.best_params,
            'fixed_params': self.fixed_params # Save fixed params context
        }
        with open(self.best_params_save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        # Optionally save the Optuna study details (handled by Tuner.save_study)
        # tuner.save_study_results(...)

        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")
        return self.best_params

    def train_final_model(self,
                          dh: 'DataHandler',
                          final_epochs: int = 150,
                          final_patience: int = 30
                         ) -> DynamicMLP:
        """
        Trains the final NN model using the best hyperparameters from CV.

        Args:
            dh (DataHandler): The data handler instance.
            final_epochs (int): Maximum number of epochs for final training.
            final_patience (int): Patience for early stopping in final training.

        Returns:
            DynamicMLP: The trained final model.
        """
        print(f"\n--- Starting Final Model Training for {self.MODEL_NAME.upper()} ---")

        # 1. Ensure best_params are available
        self._load_best_params_if_needed()
        if self.best_params is None: # Should be loaded by the helper now
             raise RuntimeError("Failed to load best_params.")
        print(f"Using best hyperparameters: {self.best_params}")
        if self.fixed_params is None:
            print("Warning: Fixed params used during HPO not found. Using defaults for layer count.")


        # 2. Prepare Data
        train_loader, val_loader = dh.final_dataloaders

        # 3. Build Model
        nn_params = self.best_params
        # Determine num_hidden_layers: Use saved fixed_params if possible, else default
        num_hidden_layers = (self.fixed_params or {}).get('num_hidden_layers', 2) # Default to 2
        hidden_layers = [nn_params[f"n_units_l{i}"] for i in range(num_hidden_layers)]
        activation_fn = ACTIVATION_MAP[nn_params['activation']]
        dropout_rate = nn_params['dropout_rate']
        self.model = DynamicMLP(self.input_dim, hidden_layers, activation_fn, dropout_rate).to(DEVICE)

        # 4. Setup Optimizer and Optional Scheduler
        optimizer_cls = OPTIMIZER_MAP[nn_params['optimizer']]
        optimizer = optimizer_cls(self.model.parameters(), lr=nn_params['learning_rate'], weight_decay=nn_params['weight_decay'])
        scheduler = None # Example: No scheduler for final training, or configure one here

        # 5. Training Loop with Early Stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state_dict = None
        self.final_loss_history = []

        print(f"Starting final training for max {final_epochs} epochs (patience={final_patience})...")
        for epoch in range(final_epochs):
            self.model.train()
            epoch_train_loss = 0.0
            for features, targets, weights in train_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                outputs = self.model(features)
                loss = NNModel.weighted_cross_entropy_loss(outputs, targets, weights)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_train_loss += loss.item()
            avg_train_loss = epoch_train_loss / len(train_loader)

            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for features, targets, weights in val_loader:
                    features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                    outputs = self.model(features)
                    loss = NNModel.weighted_cross_entropy_loss(outputs, targets, weights)
                    epoch_val_loss += loss.item()
            avg_val_loss = epoch_val_loss / len(val_loader)

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

            if scheduler: scheduler.step(avg_val_loss)

        if best_model_state_dict:
            self.model.load_state_dict(best_model_state_dict)
            print(f"Loaded best model state from epoch {epoch+1-epochs_no_improve} (Val Loss: {best_val_loss:.6f}).")
        else:
            print("Warning: No improvement observed during final training. Using model from last epoch.")
        self.model.eval()

        # 6. Save Model and History
        self._save_model()
        self._save_loss_history()

        print(f"--- Finished Final Model Training for {self.MODEL_NAME.upper()} ---")
        return self.model

    def _save_model(self):
        """Saves the trained final model state dict to disk."""
        if self.model is None:
            print("Error: No model instance available to save.")
            return
        print(f"Saving final {self.MODEL_NAME} model state_dict to: {self.model_save_path}")
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_save_path)
        print("Model state_dict saved successfully.")

    def _save_loss_history(self):
        """Saves the final training loss history to a CSV."""
        if not self.final_loss_history:
            print("Warning: No loss history recorded to save.")
            return
        print(f"Saving final training loss history to: {self.loss_save_path}")
        os.makedirs(os.path.dirname(self.loss_save_path), exist_ok=True)
        loss_df = pd.DataFrame(self.final_loss_history)
        loss_df.to_csv(self.loss_save_path, index=False)
        print("Loss history saved successfully.")

    def load_model(self) -> DynamicMLP:
        """
        Loads a pre-trained final NN model from disk.
        Requires best_params to reconstruct the architecture.
        """
        print(f"Loading trained {self.MODEL_NAME} model from: {self.model_save_path}")
        if not os.path.exists(self.model_save_path):
             raise FileNotFoundError(f"Model file not found at {self.model_save_path}")

        # 1. Ensure best_params are available to reconstruct architecture
        self._load_best_params_if_needed()
        if self.best_params is None:
             raise RuntimeError("Failed to load best_params needed for model reconstruction.")
        if self.fixed_params is None:
             print("Warning: Fixed params used during HPO not found. Using defaults for layer count.")

        # 2. Build model architecture
        nn_params = self.best_params
        num_hidden_layers = (self.fixed_params or {}).get('num_hidden_layers', 2) # Default to 2
        hidden_layers = [nn_params[f"n_units_l{i}"] for i in range(num_hidden_layers)]
        activation_fn = ACTIVATION_MAP[nn_params['activation']]
        dropout_rate = nn_params['dropout_rate']
        self.model = DynamicMLP(self.input_dim, hidden_layers, activation_fn, dropout_rate).to(DEVICE)

        # 3. Load the saved state dictionary
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=DEVICE))
        self.model.eval() # Set to evaluation mode
        print(f"{self.MODEL_NAME} model loaded successfully.")
        return self.model

    def predict(self,
                dh: DataHandler,
                save: bool = False) -> pd.DataFrame:
        """
        Generates scaled predictions for the test set using the trained NN model.

        The NN outputs probabilities summing to 1. These are scaled by the
        sum of the true targets for each county (P(18+|C)) to match the
        expected scale for evaluation.

        Args:
            dh (DataHandler): The DataHandler instance containing test data.
            save (bool, optional): If True, saves the scaled predictions to a CSV file.
                                   Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing county-level scaled predictions
                          for the four target classes (shape: [n_samples, 4]).
        """
        print(f"Generating predictions for {self.MODEL_NAME.upper()} on year {dh.test_year}...")

        # 1. Load model if not already loaded/trained
        if self.model is None:
            self.load_model() # This ensures model is on DEVICE and in eval mode

        # 2. Get test features and targets (need targets for scaling factor)
        _, (X_test, y_test, _) = dh.final_data

        # 3. Prepare tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        # Calculate scaling factor P(18+|C) from true targets
        y_tots_tensor = torch.tensor(y_test, dtype=torch.float32).sum(dim=1, keepdim=True) # Shape [n_samples, 1]

        # 4. Perform prediction (outputs probabilities summing to 1)
        self.model.eval() # Ensure evaluation mode
        with torch.no_grad():
            outputs = self.model(X_test_tensor) # Shape: [n_samples, 4]

            # 5. Scale predictions
            # Ensure outputs and y_tots_tensor are broadcastable: [n, 4] * [n, 1] -> [n, 4]
            y_pred_scaled = outputs * y_tots_tensor

        # 6. Convert scaled predictions to DataFrame (move to CPU first)
        y_pred_scaled = y_pred_scaled.numpy()
        pred_df = pd.DataFrame(y_pred_scaled, columns=targets) # Use global 'targets' list

        # 7. Optionally save
        if save:
            model_name_lower = self.MODEL_NAME.lower()
            pred_save_path = os.path.join(RESULTS_DIR, f"{model_name_lower}_{dh.test_year}_predictions.csv")
            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
            pred_df.to_csv(pred_save_path, index=False)
            print(f"  Scaled predictions saved to: {pred_save_path}")

        print(f"Finished generating {self.MODEL_NAME.upper()} predictions.")
        return pred_df


# =============================================================================
# Evaluation Function (Place AFTER all class definitions)
# =============================================================================

def evaluate_predictions(pred_dict: Dict[str, pd.DataFrame],
                         dh: DataHandler,
                         save: bool = False) -> pd.DataFrame:
    """
    Computes and aggregates evaluation metrics for predictions from multiple models.

    Calculates the aggregate true distribution and the aggregate predicted
    distribution for each model using county weights P(C). Computes self-entropy
    for the true distribution and cross-entropy for each model's prediction
    relative to the true distribution.

    Args:
        pred_dict (Dict[str, pd.DataFrame]): A dictionary where keys are model
            names (str) and values are pandas DataFrames containing the
            county-level predictions (raw for Ridge/XGBoost, scaled for NN).
            Each DataFrame must have columns matching the global 'targets' list.
        dh (DataHandler): The DataHandler instance containing the test data,
                          specifically the true targets (y_test) and weights (wts_test).
        save (bool, optional): If True, saves the resulting evaluation DataFrame
                               to a CSV file in the RESULTS_DIR. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame summarizing the evaluation.
            - Index: 'true' followed by model names (keys from pred_dict).
            - Columns:
                - 'P(democrat)', 'P(other)', 'P(republican)', 'P(non_voter)': Aggregate shares.
                - 'P(underage)': Calculated as 1 minus the sum of the four target shares.
                - 'entropy': Self-entropy for the 'true' row, cross-entropy vs. true
                             for the model rows.
    """
    print("\n--- Evaluating Model Predictions ---")

    # --- 1. Get True Targets and Weights ---
    _, (_, y_test, wts_test) = dh.final_data
    # Ensure weights are a column vector for broadcasting
    wts_test = wts_test.reshape(-1, 1)

    # --- 2. Calculate Aggregate True Distribution ---
    agg_true = (wts_test * y_test).sum(axis=0) # Shape: [4]
    print(f"Calculated aggregate true distribution using {y_test.shape[0]} samples.")

    # --- 3. Prepare Storage for Results ---
    results_data = {}
    epsilon = 1e-9 # For numerical stability in log

    # --- 4. Process True Distribution ---
    # Clip for entropy calculation
    agg_true = np.clip(agg_true, epsilon, 1.0)
    # Calculate self-entropy
    true_self_entropy = -np.sum(agg_true * np.log(agg_true))
    # Store true row data 
    results_data['true'] = list(agg_true) + [true_self_entropy]
    print(f"  Processed 'true' distribution. Self-entropy: {true_self_entropy:.6f}")


    # --- 5. Process Each Model's Predictions ---
    for model_name, pred_df in pred_dict.items():
        # Ensure predictions are NumPy array
        y_pred = pred_df.values # Shape: [n_samples, 4]

        # Calculate aggregate prediction
        agg_pred = (wts_test * y_pred).sum(axis=0) # Shape: [4]

        # Clip aggregate prediction to non-negative for entropy calc
        agg_pred = np.clip(agg_pred, epsilon, 1.0)

        # Calculate cross-entropy: - sum( P_true * log(P_pred) )
        cross_entropy = -np.sum(agg_true * np.log(agg_pred))

        # Store model row data (use original unclippped aggregate for reporting)
        results_data[model_name] = list(agg_pred) + [cross_entropy]
        print(f"    Aggregate prediction calculated. Cross-entropy vs true: {cross_entropy:.6f}")


    # --- 6. Assemble DataFrame ---
    # Define column names based on global 'targets'
    target_cols = [t.replace('|C)', ')') for t in targets] # e.g., 'P(democrat)'
    all_cols = target_cols + ['entropy']

    eval_df = pd.DataFrame.from_dict(results_data, orient='index', columns=all_cols)

    # --- 7. Calculate P(underage) ---
    # Ensure we only sum the actual target columns
    eval_df['P(underage)'] = 1.0 - eval_df[target_cols].sum(axis=1)

    # Reorder columns to have P(underage) before entropy
    final_cols_order = target_cols + ['P(underage)', 'entropy']
    eval_df = eval_df[final_cols_order]

    # --- 8. Optionally Save ---
    if save:
        eval_save_path = os.path.join(RESULTS_DIR, f"aggregate_evaluation_{dh.test_year}.csv")
        os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)
        eval_df.to_csv(eval_save_path, index=True, index_label='model') # Save index (model names)
        print(f"\nEvaluation summary saved to: {eval_save_path}")

    print("--- Finished Evaluation ---")
    return eval_df