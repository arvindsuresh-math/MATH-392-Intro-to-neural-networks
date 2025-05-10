"""
Ridge Regression model handler for the election prediction project.

The RidgeModel class encapsulates:
- Cross-validation for alpha hyperparameter selection.
- Final model training.
- Saving and loading trained models.
- Generating predictions.
"""
import os
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from typing import List, Union

from .constants import MODELS_DIR, RESULTS_DIR, PREDS_DIR # Relative import
from .data_handling import DataHandler # For type hinting

# --- RidgeModel Class ---

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
