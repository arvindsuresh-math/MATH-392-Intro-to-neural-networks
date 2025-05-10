"""
General utility functions for the election prediction project.
"""
import os
import pandas as pd
import numpy as np
import optuna 
from typing import Dict, Any

from .constants import RESULTS_DIR # Relative import
from .data_handling import DataHandler # For type hinting

# --- Evaluate Predictions Function ---

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

# --- Hyperparameter Suggestion Function ---

def suggest_hyperparameters(trial: optuna.trial.Trial, param_spaces: Dict[str, list]) -> Dict[str, Any]:
    suggested_params = {}

    if "float" in param_spaces:
        for name, low, high, is_log in param_spaces["float"]:
            suggested_params[name] = trial.suggest_float(name, low, high, log=is_log)

    if "int" in param_spaces:
        for name, low, high, step in param_spaces["int"]:
            suggested_params[name] = trial.suggest_int(name, low, high, step=step)

    return suggested_params

def suggest_mlp_params(trial: optuna.trial.Trial, depth):
    suggested_params = {}
    suggested_params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    suggested_params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    #depth-specific parameters
    for i in range(1, depth + 1):
        suggested_params[f'n_hidden_{i}'] = trial.suggest_int(f'n_hidden_{i}', 8, 128, step=8)
        suggested_params[f'dropout_rate_{i}'] = trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.5, log=False)
    return suggested_params