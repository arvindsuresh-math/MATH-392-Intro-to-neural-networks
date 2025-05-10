"""
XGBoost model handler for the election prediction project.

The XGBoostModel class encapsulates:
- Cross-validation with custom objective and evaluation metrics.
- Final model training.
- Saving and loading trained XGBoost boosters.
- Generating predictions.
"""
import os
import pandas as pd
import numpy as np
import json
import xgboost as xgb
import ast
import time
from sklearn.model_selection import ParameterGrid
from typing import List, Dict, Union

from .constants import MODELS_DIR, RESULTS_DIR, PREDS_DIR, DEVICE
from .data_handling import DataHandler # For type hinting
from .metrics import softmax, weighted_softprob_obj, weighted_cross_entropy_eval # Import metrics

# --- XGBoostModel Class ---

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
                obj = weighted_softprob_obj, # Custom objective
                custom_metric=weighted_cross_entropy_eval,
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
            custom_metric=weighted_cross_entropy_eval, # Needed if using evals list
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
        print(f"\n--- Loading Final Model for {self.model_name.upper()} ---")
        print(f"Loading model state from: {self.model_save_path}")
        self.model = xgb.Booster()
        self.model.load_model(self.model_save_path)
        print(f"{self.model_name.upper()} model loaded successfully.")

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
        y_pred = softmax(y_pred) 
        y_pred = y_pred * y_tots

        if save:
            pred_df = pd.DataFrame(y_pred, columns=dh.targets)
            pred_df.to_csv(self.pred_save_path, index=False)
            print(f"County-level scaled predictions saved to: {self.pred_save_path}")

        return y_pred


# Replace self.softmax, self.weighted_softprob_obj, etc. with imported functions.
# Ensure paths use constants from constants.py
# Example in cross_validate:
# obj=weighted_softprob_obj, custom_metric=weighted_cross_entropy_eval
# Example in predict:
# y_pred = softmax(y_pred)