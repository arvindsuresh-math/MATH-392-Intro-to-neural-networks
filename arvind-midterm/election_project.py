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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter          
from datetime import datetime  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import joblib
from typing import List, Dict, Tuple, Any, Type, Union
from abc import ABC, abstractmethod

# tidy default log‑dir:  logs/ridge_2020‑04‑18_14‑30‑25
def _tb_writer(model_name: str) -> SummaryWriter:
    timestamp = datetime.now().strftime("%Y‑%m‑d_%H‑%M‑%S")
    return SummaryWriter(log_dir=f"logs/{model_name}_{timestamp}")

# =============================================================================
# 0. All features and targets
# =============================================================================

years = [2008, 2012, 2016, 2020]

idx = ['year', 'gisjoin', 'state', 'county']

incomes= ['median_household_income','per_capita_income']

land = ['per_capita_area']

households = [
        'per_capita_households_income_under_10k',
        'per_capita_households_income_10k_15k',
        'per_capita_households_income_15k_25k',
        'per_capita_households_income_25k_plus'
                ]

sexes = ['P(persons_male|C)', 'P(persons_female|C)']

sex_maritals = [
        'P(male_never_married|C)',
        'P(male_married|C)',
        'P(male_separated|C)',
        'P(male_widowed|C)',
        'P(male_divorced|C)',
        'P(female_never_married|C)',
        'P(female_married|C)',
        'P(female_separated|C)',
        'P(female_widowed|C)',
        'P(female_divorced|C)'
                ]

sex_age_edus = [
        'P(male_18_24_less_than_9th|C)',
        'P(male_18_24_some_hs|C)',
        'P(male_18_24_hs_grad|C)',
        'P(male_18_24_some_college|C)',
        'P(male_18_24_associates|C)',
        'P(male_18_24_bachelors|C)',
        'P(male_18_24_graduate|C)',
        'P(male_25_34_less_than_9th|C)',
        'P(male_25_34_some_hs|C)',
        'P(male_25_34_hs_grad|C)',
        'P(male_25_34_some_college|C)',
        'P(male_25_34_associates|C)',
        'P(male_25_34_bachelors|C)',
        'P(male_25_34_graduate|C)',
        'P(male_35_44_less_than_9th|C)',
        'P(male_35_44_some_hs|C)',
        'P(male_35_44_hs_grad|C)',
        'P(male_35_44_some_college|C)',
        'P(male_35_44_associates|C)',
        'P(male_35_44_bachelors|C)',
        'P(male_35_44_graduate|C)',
        'P(male_45_64_less_than_9th|C)',
        'P(male_45_64_some_hs|C)',
        'P(male_45_64_hs_grad|C)',
        'P(male_45_64_some_college|C)',
        'P(male_45_64_associates|C)',
        'P(male_45_64_bachelors|C)',
        'P(male_45_64_graduate|C)',
        'P(male_65plus_less_than_9th|C)',
        'P(male_65plus_some_hs|C)',
        'P(male_65plus_hs_grad|C)',
        'P(male_65plus_some_college|C)',
        'P(male_65plus_associates|C)',
        'P(male_65plus_bachelors|C)',
        'P(male_65plus_graduate|C)',
        'P(female_18_24_less_than_9th|C)',
        'P(female_18_24_some_hs|C)',
        'P(female_18_24_hs_grad|C)',
        'P(female_18_24_some_college|C)',
        'P(female_18_24_associates|C)',
        'P(female_18_24_bachelors|C)',
        'P(female_18_24_graduate|C)',
        'P(female_25_34_less_than_9th|C)',
        'P(female_25_34_some_hs|C)',
        'P(female_25_34_hs_grad|C)',
        'P(female_25_34_some_college|C)',
        'P(female_25_34_associates|C)',
        'P(female_25_34_bachelors|C)',
        'P(female_25_34_graduate|C)',
        'P(female_35_44_less_than_9th|C)',
        'P(female_35_44_some_hs|C)',
        'P(female_35_44_hs_grad|C)',
        'P(female_35_44_some_college|C)',
        'P(female_35_44_associates|C)',
        'P(female_35_44_bachelors|C)',
        'P(female_35_44_graduate|C)',
        'P(female_45_64_less_than_9th|C)',
        'P(female_45_64_some_hs|C)',
        'P(female_45_64_hs_grad|C)',
        'P(female_45_64_some_college|C)',
        'P(female_45_64_associates|C)',
        'P(female_45_64_bachelors|C)',
        'P(female_45_64_graduate|C)',
        'P(female_65plus_less_than_9th|C)',
        'P(female_65plus_some_hs|C)',
        'P(female_65plus_hs_grad|C)',
        'P(female_65plus_some_college|C)',
        'P(female_65plus_associates|C)',
        'P(female_65plus_bachelors|C)',
        'P(female_65plus_graduate|C)'
            ]

sex_races = [
        'P(male_white|C)',
        'P(female_white|C)',
        'P(male_black|C)',
        'P(female_black|C)',
        'P(male_aian|C)',
        'P(female_aian|C)',
        'P(male_asian|C)',
        'P(female_asian|C)',
        'P(male_nhpi|C)',
        'P(female_nhpi|C)',
        'P(male_other|C)',
        'P(female_other|C)',
        'P(male_multi|C)',
        'P(female_multi|C)'
            ]

nativities = ['P(persons_native|C)', 'P(persons_foreign_born|C)']

labors = [
    'P(labor_force_total|C)',
    'P(labor_force_armed|C)',
    'P(labor_force_civilian|C)',
    'P(labor_force_employed|C)',
    'P(labor_force_unemployed|C)',
    'P(not_in_labor_force|C)'
        ]

misc = ['P(persons_hispanic|C)', 
        'P(persons_below_poverty|C)']

weight_feats = ['P(C)', 'P(18plus|C)']

targets = ['P(democrat|C)',
           'P(other|C)',
           'P(republican|C)',
           'P(non_voter|C)']

all_features = weight_feats \
        + incomes \
        + land \
        + households \
        + sexes \
        + sex_maritals \
        + sex_races \
        + sex_age_edus \
        + nativities \
        + labors \
        + misc

# =============================================================================
# 1. Global Constants and Configuration
# =============================================================================

# --- File Paths ---
DATA_DIR = "./data"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"

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
MAX_CV_EPOCHS: int = 100 # Max epochs for early stopping during CV
PATIENCE: int = 20      # Patience for early stopping during CV
FINAL_TRAIN_EPOCHS: int = 300 # Fixed epochs for final training
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

    def __init__(self, test_year: int = 2020, features_to_drop = []):
        """
        Initializes the DataHandler and determines input/output dimensions.

        Args:
            test_year: Year to use for test data (default: 2020).
            features_to_drop: List of features to drop from the dataset (default: []).
        """
        self.features = list(set(all_features) - set(features_to_drop))
        self.input_dim = len(self.features)
        self.test_year = test_year
        self.train_years = sorted(set(years) - {test_year})
        self.folds = [
                ([self.train_years[0], self.train_years[1]], self.train_years[2]),
                ([self.train_years[0], self.train_years[2]], self.train_years[1]),
                ([self.train_years[1], self.train_years[2]], self.train_years[0])
                    ]

        print(f"DataHandler initialized: n_features = {self.input_dim}")

    def load_data(self, 
                  fit_years: List[int], # list of years to fit StandardScalar
                  transform_year: int, # year to transform StandardScaler
                  ):
        data = pd.read_csv('data/final_dataset.csv')

        # make datasets with fit years and transform years
        df_fit = data[data['year'].isin(fit_years)].reset_index(drop=True)
        df_transform = data[data['year'] == transform_year].reset_index(drop=True)

        # make wts
        wts_fit = df_fit[['P(C)']]
        wts_transform = df_transform[['P(C)']]

        # make y's
        y_fit = df_fit[targets]
        y_transform = df_transform[targets]

        # make X's
        X_fit = df_fit[self.features]
        X_transform = df_transform[self.features]

        #apply StandardScalar to X's
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit) #fit and transform X_fit
        X_transform = scaler.transform(X_transform) #transform X_transform

        # wrap back into DataFrames so downstream .values works
        X_fit = pd.DataFrame(X_fit, columns=self.features)
        X_transform = pd.DataFrame(X_transform, columns=self.features)

        return X_fit, y_fit, wts_fit, X_transform, y_transform, wts_transform

    def _create_dataset(self, X_df: pd.DataFrame, y_df: pd.DataFrame, wts_df: pd.DataFrame) -> TensorDataset:
        """Converts pandas DataFrames to a PyTorch TensorDataset."""
        X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_df.values, dtype=torch.float32)
        wts_tensor = torch.tensor(wts_df.values, dtype=torch.float32) # Already [:, 1] shape
        return TensorDataset(X_tensor, y_tensor, wts_tensor)

    def create_dataloaders(self, train_years: List[int], val_year: int) -> Tuple[DataLoader, DataLoader]:
        """Creates DataLoaders for training and validation sets."""
        # load data for training and validation years
        X_train, y_train, wts_train, X_val, y_val, wts_val = self.load_data(train_years,val_year)

        # create tensor datasets
        train_dataset = self._create_dataset(X_train, y_train, wts_train)
        val_dataset = self._create_dataset(X_val, y_val, wts_val)

        # Use persistent_workers and pin_memory for potentially faster loading if CUDA available
        # num_workers > 0 can speed things up but sometimes causes issues on Windows/macOS
        num_workers = 2 if DEVICE.type == 'cuda' else 0 # Adjust as needed
        pin_memory = True if DEVICE.type == 'cuda' else False

        train_loader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True,
                                  num_workers=num_workers, 
                                  pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False,
                                num_workers=num_workers, 
                                pin_memory=pin_memory)

        print(f"Created DataLoaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
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
                       data_handler: DataHandler,
                       param_grid: List[float] = RIDGE_PARAM_GRID
                       ) -> None:
        """Performs CV for Ridge, saves results, and stores the best alpha."""
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        results_list = []
        best_score = float('inf')
        current_best_alpha = None

        for alpha in param_grid:
            print(f"  Testing alpha = {alpha}:")
            fold_val_losses = []
            for train_years, val_year in data_handler.folds:
                # Load data for the current fold
                X_train, y_train, wts_train, X_val, y_val, wts_val = data_handler.load_data(train_years,val_year)

                # fit and predict
                model_fold = Ridge(alpha=alpha)
                model_fold.fit(X_train.values, y_train.values, sample_weight=wts_train.values.ravel())
                y_pred_val = model_fold.predict(X_val.values)

                # Use weighted MSE (weights = P(C)) to compute validation loss
                weights_val = wts_val.values.squeeze() # Ensure weights is 1D array
                val_loss = mean_squared_error(y_val.values, y_pred_val, sample_weight=weights_val)

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
                          data_handler: DataHandler
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
        X_train, y_train, wts_train, _, _, _ = data_handler.load_data(data_handler.train_years, data_handler.test_year)
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

    def load_model(self) -> Ridge:
        """Loads a trained Ridge model."""
        print(f"Loading trained Ridge model from: {self.model_save_path}")
        self.model = joblib.load(self.model_save_path)
        print("Ridge model loaded successfully.")
        return self.model

    def evaluate(self,
                 data_handler: DataHandler
                 ) -> Tuple[np.ndarray, float]:
        """
        Evaluates the trained Ridge model on a specified test dataset, saves
        county-level raw predictions to CSV, calculates an approximate aggregate
        cross-entropy loss after applying Softmax to the aggregate predictions,
        and returns the Softmax-normalized aggregate prediction vector and the CE loss.
        (NO ERROR HANDLING)

        Calculation Steps:
        1. Load the entire test dataset (features, targets, weights) specified
           by `test_data_suffix` into NumPy arrays.
        2. Use the trained Ridge model to predict raw scores for each county.
        3. Save the county-level raw predictions `y_pred_raw` to a CSV file.
        4. Calculate the Aggregate Predicted Raw Scores by computing the
           weighted sum of county-level raw scores:
           AggPredRaw = sum_over_counties_C ( P(C) * prediction_raw_C )
        5. Calculate the Aggregate True Distribution similarly using true targets:
           AggTrue = sum_over_counties_C ( P(C) * target_C )
        6. Apply the Softmax function to the Aggregate Predicted Raw Scores to
           convert them into a probability distribution (AggPredSoftmax).
        7. Compute the cross-entropy between AggTrue and AggPredSoftmax:
           AggCE = - sum_over_classes_k ( AggTrue_k * log(AggPredSoftmax_k) )
        8. Return the Softmax-normalized aggregate prediction (as NumPy array) and the
           aggregate cross-entropy loss (as float).

        Args:
            test_data_suffix (str): The suffix identifying the test dataset.
            data_handler (DataHandler): An instance of the DataHandler class.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - agg_pred_np (np.ndarray): Vector of predicted national
                  vote shares after Softmax (shape: [output_dim]).
                - agg_ce (float): The calculated CE loss.
        """
        print(f"Evaluating {self.MODEL_NAME.upper()} on year '{data_handler.test_year}' with Aggregate Cross-Entropy (CPU)...")

        # --- All computations use NumPy (implicitly CPU) ---
        # Assume self.model exists and is loaded/trained before calling evaluate

        # 1. Load the entire test dataset into NumPy arrays
        # print(f"Loading test data for suffix: {test_data_suffix}")
        _, _, _, X_pd, y_pd, wts_pd = data_handler.load_data(data_handler.train_years,data_handler.test_year)

        X_np = X_pd.values
        y_np = y_pd.values # Shape: [num_samples, output_dim]
        wts_np = wts_pd.values # Shape: [num_samples, 1]

        print(f"Test data loaded: {X_np.shape[0]} samples.")

        # 2. Use the trained Ridge model to predict raw scores
        y_pred_np = self.model.predict(X_np) # Shape: [num_samples, output_dim]

        # 3. Save county-level raw predictions
        pred_df = pd.DataFrame(y_pred_np, columns=y_pd.columns) # Use target column names
        pred_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_{data_handler.test_year[0]}_predictions.csv")
        pred_df.to_csv(pred_save_path, index=False)
        print(f"  County-level raw predictions saved to: {pred_save_path}")

        # 4. Calculate Aggregate Predicted Raw Scores
        agg_pred_np = (wts_np * y_pred_np).sum(axis=0) # Shape: [output_dim]

        # 5. Calculate Aggregate True Distribution
        agg_true_np = (wts_np * y_np).sum(axis=0) # Shape: [output_dim]

        # --- Final Metric Computation ---
        # Use torch for softmax and CE calculation for consistency
        cpu_device = torch.device("cpu")
        agg_pred_tensor = torch.tensor(agg_pred_np, dtype=torch.float32, device=cpu_device)
        agg_true_tensor = torch.tensor(agg_true_np, dtype=torch.float32, device=cpu_device)

        # 6. Apply Softmax to aggregate raw predictions
        agg_pred_tensor = torch.softmax(agg_pred_tensor, dim=0)

        # Clamp probabilities
        epsilon = 1e-9
        agg_pred_tensor = torch.clamp(agg_pred_tensor, min=epsilon)

        # 7. Calculate cross-entropy between true and softmax predictions
        agg_ce = (-torch.sum(agg_true_tensor * torch.log(agg_pred_tensor))).item()

        # Prepare return values
        agg_pred_np = agg_pred_tensor.numpy()

        # Display the resulting aggregate distributions
        print(f"  Aggregate True Distribution: {agg_true_np}")
        print(f"  Aggregate Predicted Distribution (Softmax): {agg_pred_np}")
        print(f"  Aggregate Cross-Entropy: {agg_ce:.6f}")

        # 8. Return Softmax-normalized aggregate prediction and loss
        return agg_pred_np, agg_ce

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
        self.results_save_path = os.path.join(RESULTS_DIR, f"{self.MODEL_NAME}_cv_results.csv")
        self.state_dict_save_path = os.path.join(MODELS_DIR, f"{self.MODEL_NAME}_final_state_dict.pth")
        self.loss_save_path = os.path.join(RESULTS_DIR, f"{self.MODEL_NAME}_final_training_loss.csv")

    @staticmethod
    def weighted_cross_entropy_loss(outputs: torch.Tensor,
                                    targets: torch.Tensor,
                                    weights: torch.Tensor) -> torch.Tensor:
        """
        Calculates a custom weighted cross-entropy loss for a batch.
        Loss(C) = - sum_k ( target_k * log(output_k) ) / sum_k (target_k)
        Batch Loss = Expected sample loss = sum_C ( P(C) * Loss(C) ) / sum_C ( P(C) )

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
        # Calculate Cross-Entropy per sample: - sum(target * log(pred))/sum(target) over classes
        sample_ce_loss = -torch.sum(targets * torch.log(outputs_clamped), dim=1, keepdim=True)
        # Note: targets sum to less than 1, so need to normalize by the sum
        sample_ce_loss /= torch.sum(targets, dim=1, keepdim=True)
        # Ensure weights tensor has the same shape as sample_ce_loss for broadcasting
        weights_reshaped = weights.view_as(sample_ce_loss)
        # Apply weights: weight * sample_loss
        weighted_sample_losses = sample_ce_loss * weights_reshaped
        # Calculate the sum of weights for normalization
        total_weight = weights_reshaped.sum()
        # Calculate the batch loss. 
        batch_loss = weighted_sample_losses.sum() / total_weight
        return batch_loss

    @abstractmethod
    def _build_network(self, params: Dict, input_dim: int) -> nn.Module:
        """
        Subclasses must implement this method to return an initialized nn.Module
        based on the provided hyperparameters.

        Args:
            params (Dict): Dictionary containing hyperparameters (e.g., 'learning_rate', 'n_hidden').
            input_dim (int): Input dimension for the network.

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
                       max_epochs: int = MAX_CV_EPOCHS,
                       patience: int = PATIENCE,
                       optimizer_choice: Type[optim.Optimizer] = OPTIMIZER_CHOICE
                       ) -> None:
        """Performs CV for the NN model, saves results, and stores best params."""
        input_dim = data_handler.input_dim
        print(f"\n--- Starting Cross-Validation for {self.MODEL_NAME.upper()} ---")
        tb = _tb_writer(self.MODEL_NAME + "_cv")

        results_list = []
        best_mean_loss = float('inf')
        current_best_params = {}

        # Use the PARAM_GRID defined in the subclass
        for i, config in enumerate(ParameterGrid(self.PARAM_GRID), 1):
            print(f"  Testing config: {config}")
            fold_val_losses = []
            for j, (train_years, val_year) in enumerate(data_handler.folds, 1):
                print(f"    Fold {i}: Train Years: {train_years}, Val Year: {val_year}")
                # Create DataLoaders for this fold
                train_loader, val_loader = data_handler.create_dataloaders(train_years,val_year)

                # Build model using subclass implementation
                model_fold = self._build_network(config, input_dim).to(DEVICE)

                # Setup optimizer using parameters from config
                optimizer = optimizer_choice(model_fold.parameters(), 
                                             lr=config['learning_rate'], 
                                             weight_decay=config.get('weight_decay', 0))

                val_loss = self._train_one_fold(
                                                model_fold, 
                                                train_loader, 
                                                val_loader, 
                                                optimizer, 
                                                max_epochs, 
                                                patience
                                                    )
                fold_val_losses.append(val_loss)
                tb.add_scalar(f"config{i}/fold{j}_val", val_loss, i)
                # You can also log hyper‑params once:
                if j == 0:
                    tb.add_hparams(config, {"mean_val_loss": 0})   # placeholder

                # Clean up memory
                del train_loader, val_loader, model_fold, optimizer
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                elif DEVICE.type == 'mps': torch.mps.empty_cache()

            # Filter out non-finite scores before calculating mean
            finite_scores = [s for s in fold_val_losses if np.isfinite(s)]
            mean_val_loss = np.mean(finite_scores) if finite_scores else float('inf')

            print(f"    Avg Val Score (Weighted CE): {mean_val_loss:.6f}")
            results_list.append({**config, 'mean_cv_score': mean_val_loss})

            # Update best score and params if this fold is better
            if mean_val_loss < best_mean_loss:
                best_mean_loss = mean_val_loss
                current_best_params = config

        self.best_params = current_best_params # Store best params found
        results_df = pd.DataFrame(results_list).sort_values(by='mean_cv_score', ascending=True).reset_index(drop=True)
        results_df.to_csv(self.results_save_path, index=False)
        print(f"CV results saved to: {self.results_save_path}")

        print(f"\nBest {self.MODEL_NAME.upper()} CV params: {self.best_params} (Score: {best_mean_loss:.6f})")
        print(f"--- Finished Cross-Validation for {self.MODEL_NAME.upper()} ---")

    def train_final_model(self,
                          data_handler: DataHandler,
                          final_train_epochs: int = FINAL_TRAIN_EPOCHS,
                          optimizer_choice: Type[optim.Optimizer] = OPTIMIZER_CHOICE
                          ) -> nn.Module:
        """Trains the final NN model using the best hyperparams from CV."""
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
        self.model = self._build_network(self.best_params, data_handler.input_dim).to(DEVICE)

        # 3. Create final DataLoader
        final_train_loader,_ = data_handler.create_dataloaders(data_handler.train_years, data_handler.test_year)

        # 4. Setup Optimizer
        lr = self.best_params['learning_rate']
        wd = self.best_params.get('weight_decay', 0)
        optimizer = optimizer_choice(self.model.parameters(), lr=lr, weight_decay=wd)

        # 5. Training Loop
        self.model.train()
        self.final_loss_history = []
        print(f"Starting training for {final_train_epochs} epochs...")

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
        self.model = self._build_network(self.best_params, input_dim).to(DEVICE)

        # 3. Load the saved state dictionary
        self.model.load_state_dict(torch.load(self.state_dict_save_path, map_location=DEVICE))
        self.model.eval() # Set to evaluation mode
        print(f"{self.MODEL_NAME} model loaded successfully.")
        return self.model

    def evaluate(self,
                 data_handler: DataHandler
                 ) -> Tuple[np.ndarray, float]:
        """
        Evaluates the trained NN model on a specified test dataset, saves county-level
        predictions to CSV, and returns the aggregate national prediction vector
        and the aggregate cross-entropy loss. (NO ERROR HANDLING)

        Calculation Steps:
        1. Load the entire test dataset (features, targets, weights) specified
           by `test_data_suffix` into CPU tensors.
        2. Perform a single forward pass of the model (on CPU) on all test
           features to get county-level predictions (probability distributions).
        3. Save the county-level predictions `y_pred` to a CSV file.
        4. Calculate the Aggregate Predicted Distribution by computing the
           weighted sum of county-level predictions:
           AggPred = sum_over_counties_C ( P(C) * prediction_C )
        5. Calculate the Aggregate True Distribution similarly using true targets:
           AggTrue = sum_over_counties_C ( P(C) * target_C )
        6. Compute the cross-entropy between these two aggregate distributions:
           AggCE = - sum_over_classes_k ( AggTrue_k * log(AggPred_k) )
        7. Return the aggregate predicted distribution (as NumPy array) and the
           aggregate cross-entropy loss (as float).

        Args:
            test_data_suffix (str): The suffix identifying the test dataset.
            data_handler (DataHandler): An instance of the DataHandler class.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - aggregate_prediction (np.ndarray): Vector of predicted national
                  vote shares (shape: [output_dim]).
                - aggregate_cross_entropy (float): The calculated CE loss.
        """
        # print(f"Evaluating {self.MODEL_NAME.upper()} on '{test_data_suffix}' with Aggregate Cross-Entropy (CPU)...")

        # --- Force CPU ---
        cpu_device = torch.device("cpu")
        # Assume self.model exists and is loaded/trained before calling evaluate
        self.model.to(cpu_device) # Move model to CPU for evaluation

        # 1. Load the entire test dataset directly to CPU tensors
        # print(f"Loading test data for suffix: {test_data_suffix}")
        _,_,_, X_pd, y_pd, wts_pd = data_handler.load_data(data_handler.train_years, data_handler.test_year)

        # Convert pandas DataFrames to PyTorch tensors on CPU
        X_test = torch.tensor(X_pd.values, dtype=torch.float32, device=cpu_device)
        y_test = torch.tensor(y_pd.values, dtype=torch.float32, device=cpu_device)
        wts_test = torch.tensor(wts_pd.values, dtype=torch.float32, device=cpu_device)

        print(f"Test data loaded: {X_test.shape[0]} samples.")

        # Ensure model is in evaluation mode
        self.model.eval()

        # Disable gradient calculations
        with torch.no_grad():
            # 2. Perform forward pass on the entire test set (on CPU)
            y_pred_tensor = self.model(X_test) # Shape: [num_samples, output_dim]

            # 3. Save county-level predictions
            y_pred_np = y_pred_tensor.numpy() # Convert predictions to NumPy array
            pred_df = pd.DataFrame(y_pred_np, columns=y_pd.columns) # Use target column names
            pred_save_path = os.path.join(RESULTS_DIR, f"{self.MODEL_NAME}_{data_handler.test_year}_predictions.csv")
            pred_df.to_csv(pred_save_path, index=False)
            print(f"  County-level predictions saved to: {pred_save_path}")

            # 4. Calculate Aggregate True Distribution
            agg_y_true = (wts_test * y_test).sum(dim=0) # Shape: [output_dim]
            agg_y_true /= agg_y_true.sum() # Normalize to ensure it sums to 1

            # 5. Calculate Aggregate Predicted Distribution
            agg_y_pred = (wts_test * y_pred_tensor).sum(dim=0) # Shape: [output_dim]
            # clamp the predicted distribution to avoid log(0)
            epsilon = 1e-9
            agg_y_pred = torch.clamp(agg_y_pred, min=epsilon)

            # 6. Calculate cross-entropy between the two aggregate distributions
            agg_ce = (-torch.sum(agg_y_true * torch.log(agg_y_pred))).item()

            # Prepare return values
            agg_y_pred = agg_y_pred.numpy() # Return the unclamped version

            # Display the resulting aggregate distributions
            print(f"  Aggregate True Distribution: {agg_y_true.numpy()}")
            print(f"  Aggregate Predicted Distribution: {agg_y_pred}")
            print(f"  Aggregate Cross-Entropy: {agg_ce:.6f}")

        # 7. Return aggregate prediction and loss
        return agg_y_pred, agg_ce

# =============================================================================
# 6. SoftmaxModel Class
# =============================================================================

class SoftmaxModel(BaseNNModel):
    """Handles Softmax Regression (as NN) CV, training, loading, evaluation."""
    MODEL_NAME = "softmax"
    PARAM_GRID = SOFTMAX_PARAM_GRID # Use global constant

    class _SoftmaxNet(nn.Module):
        """Internal nn.Module definition for Softmax Regression."""
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 4)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            logits = self.linear(x)
            probabilities = self.softmax(logits)
            return probabilities

    def _build_network(self, params: Dict, input_dim: int) -> nn.Module:
        """Builds the Softmax Regression network."""
        # Softmax doesn't have structural hyperparameters in this setup
        return self._SoftmaxNet(input_dim)

# =============================================================================
# 7. MLP1Model Class
# =============================================================================

class MLP1Model(BaseNNModel):
    """Handles 1-Layer MLP CV, training, loading, and evaluation."""
    MODEL_NAME = "mlp1"
    PARAM_GRID = MLP1_PARAM_GRID # Use global constant

    class _MLP1Net(nn.Module):
        """Internal nn.Module definition for 1-Hidden-Layer MLP."""
        def __init__(self, input_dim, n_hidden, dropout_rate):
            super().__init__()
            self.layer_1 = nn.Linear(input_dim, n_hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.layer_2 = nn.Linear(n_hidden, 4)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.dropout(self.relu(self.layer_1(x)))
            logits = self.layer_2(x)
            probabilities = self.softmax(logits)
            return probabilities

    def _build_network(self, params: Dict, input_dim: int) -> nn.Module:
        """Builds the 1-Layer MLP network using hyperparameters from params."""
        n_hidden = int(params['n_hidden']) # Ensure integer type
        dropout_rate = params['dropout_rate']
        return self._MLP1Net(input_dim, n_hidden, dropout_rate)

# =============================================================================
# 8. MLP2Model Class
# =============================================================================

class MLP2Model(BaseNNModel):
    """Handles 2-Layer MLP CV, training, loading, and evaluation."""
    MODEL_NAME = "mlp2"
    PARAM_GRID = MLP2_PARAM_GRID # Use global constant

    class _MLP2Net(nn.Module):
        """Internal nn.Module definition for 2-Hidden-Layer MLP."""
        def __init__(self, input_dim, shared_hidden_size, dropout_rate):
            super().__init__()
            self.layer_1 = nn.Linear(input_dim, shared_hidden_size)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.layer_2 = nn.Linear(shared_hidden_size, shared_hidden_size)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)
            self.layer_3 = nn.Linear(shared_hidden_size, 4)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.dropout1(self.relu1(self.layer_1(x)))
            x = self.dropout2(self.relu2(self.layer_2(x)))
            logits = self.layer_3(x)
            probabilities = self.softmax(logits)
            return probabilities

    def _build_network(self, params: Dict, input_dim: int) -> nn.Module:
        """Builds the 2-Layer MLP network using hyperparameters from params."""
        shared_hidden_size = int(params['shared_hidden_size']) # Ensure integer type
        dropout_rate = params['dropout_rate']
        return self._MLP2Net(input_dim, shared_hidden_size, dropout_rate)

# =============================================================================
# End of Script (Ready for import and use in a notebook)
# =============================================================================