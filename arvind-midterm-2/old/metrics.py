# =============================================================================
# Metrics and Evaluation Utilities - Election Prediction Project
# =============================================================================
"""
Provides functions for calculating performance metrics, including loss functions
used during model training/HPO (weighted cross-entropy, weighted MSE) and
functions for evaluating final model predictions (aggregate evaluation).
"""

import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb # Needed for DMatrix type hint and potentially xgb classes
from typing import Dict, Optional, Tuple

# Imports from other modules (type hints or required values)
from data_handling import DataHandler # Type hint only
from utils import targets # Import global target list


# =============================================================================
# Loss/Metric Functions for Training/HPO
# =============================================================================

def weighted_cross_entropy_loss(outputs: torch.Tensor,
                                targets_batch: torch.Tensor,
                                weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates weighted cross-entropy loss for a batch (for NN training/eval).

    Loss(C) = - sum_k ( target_k * log(output_k) )
    Batch Loss = Expected sample loss = sum_C ( P(C) * Loss(C) ) / sum_C ( P(C) )

    Args:
        outputs (torch.Tensor): Model predictions (probabilities), shape [batch_size, num_classes].
        targets_batch (torch.Tensor): Ground truth targets, shape [batch_size, num_classes].
        weights (torch.Tensor): Sample weights ('P(C)'), shape [batch_size, 1].

    Returns:
        torch.Tensor: Scalar tensor representing the weighted average loss for the batch.
    """
    epsilon = 1e-9
    # Clamp model outputs (probabilities) before log
    outputs_clamped = torch.clamp(outputs, epsilon, 1. - epsilon)
    # Pointwise cross-entropy: - target * log(output)
    # Sum over classes for each sample
    sample_ce_loss = -torch.sum(targets_batch * torch.log(outputs_clamped), dim=1, keepdim=True) # Shape: [batch_size, 1]
    # Ensure weights have shape [batch_size, 1]
    weights_reshaped = weights.view_as(sample_ce_loss)
    # Weighted sample losses
    weighted_sample_losses = sample_ce_loss * weights_reshaped
    # Average over batch, weighted by sum of weights in batch
    sum_weights = weights_reshaped.sum()
    if sum_weights <= 0: return torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype) # Avoid division by zero, match type/device
    batch_loss = weighted_sample_losses.sum() / sum_weights
    return batch_loss


def weighted_mse_xgb(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """
    Custom XGBoost evaluation metric: Weighted Mean Squared Error (for XGBoost HPO/eval).

    Retrieves labels and weights from the DMatrix and calculates MSE weighted
    by the sample weights.

    Args:
        preds (np.ndarray): Predictions made by the model. Shape might be flattened.
        dtrain (xgb.DMatrix): The DMatrix containing the true labels and weights.

    Returns:
        Tuple[str, float]: A tuple containing the metric name ('weighted_mse')
                           and the calculated metric value.
    """
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    if weights is None or len(weights) == 0: weights = np.ones_like(labels)

    num_samples = len(weights)
    num_outputs = 4 # Assuming 4 outputs

    # Ensure labels and preds have correct shape [n_samples, n_outputs]
    # XGBoost might return flattened preds/labels depending on version/setup
    if preds.ndim == 1 or preds.shape[0] != num_samples or preds.shape[1] != num_outputs:
         preds = preds.reshape(num_samples, num_outputs)
    if labels.ndim == 1 or labels.shape[0] != num_samples or labels.shape[1] != num_outputs:
         labels = labels.reshape(num_samples, num_outputs)

    squared_error = (labels - preds)**2
    sum_weights = np.sum(weights)
    if sum_weights <= 0: return 'weighted_mse', np.inf

    # Weight the mean squared error *per sample*
    weighted_squared_error_per_sample = np.mean(squared_error, axis=1) * weights
    weighted_mse = np.sum(weighted_squared_error_per_sample) / sum_weights

    return 'weighted_mse', float(weighted_mse)


# =============================================================================
# Final Evaluation Function
# =============================================================================

def evaluate_predictions(pred_dict: Dict[str, pd.DataFrame],
                         dh: 'DataHandler', # Use DataHandler type hint
                         save_path: Optional[str] = None
                         ) -> pd.DataFrame:
    """
    Computes and aggregates evaluation metrics for predictions from multiple models.

    Calculates the aggregate true distribution and the aggregate predicted
    distribution for each model using county weights P(C). Computes self-entropy
    for the true distribution and cross-entropy for each model's prediction
    relative to the true distribution.

    Args:
        pred_dict (Dict[str, pd.DataFrame]): Dictionary where keys are model names
            and values are DataFrames of county-level predictions (raw for
            Ridge/XGBoost, scaled for NN). Columns must match 'targets'.
        dh (DataHandler): The DataHandler instance with test data (y_test, wts_test).
        save_path (Optional[str]): Full path to save the evaluation DataFrame CSV.
                                   If None, the DataFrame is not saved.

    Returns:
        pd.DataFrame: DataFrame summarizing evaluation with aggregate shares and entropy.
    """
    print("\n--- Evaluating Model Predictions ---")

    # --- 1. Get True Targets and Weights ---
    _, (_, y_test, wts_test) = dh.final_data # Get NumPy arrays
    wts_test = wts_test.reshape(-1, 1)

    # --- 2. Calculate Aggregate True Distribution ---
    agg_true_shares = (wts_test * y_test).sum(axis=0)
    print(f"Calculated aggregate true distribution using {y_test.shape[0]} samples.")

    # --- 3. Prepare Storage for Results ---
    results_data = {}
    epsilon = 1e-9

    # --- 4. Process True Distribution ---
    agg_true_clipped_shares = np.maximum(agg_true_shares, 0)
    agg_true_sum = agg_true_clipped_shares.sum()
    if agg_true_sum < epsilon:
        print("Warning: Sum of aggregate true shares is near zero.")
        agg_true_prob = np.ones_like(agg_true_shares) / len(agg_true_shares)
    else:
        agg_true_prob = agg_true_clipped_shares / agg_true_sum
    agg_true_prob_clamped = np.clip(agg_true_prob, epsilon, 1.0)
    true_self_entropy = -np.sum(agg_true_prob * np.log(agg_true_prob_clamped))
    results_data['true'] = list(agg_true_shares) + [true_self_entropy]
    print(f"  Processed 'true' distribution. Self-entropy: {true_self_entropy:.6f}")

    # --- 5. Process Each Model's Predictions ---
    for model_name, pred_df in pred_dict.items():
        print(f"  Processing model: {model_name}")
        y_pred = pred_df[targets].values # Ensure use of global targets list

        agg_pred_shares = (wts_test * y_pred).sum(axis=0)
        agg_pred_clipped_shares = np.maximum(agg_pred_shares, 0)
        agg_pred_sum = agg_pred_clipped_shares.sum()
        if agg_pred_sum < epsilon:
             print(f"  Warning: Sum of aggregate predicted shares for {model_name} is near zero.")
             agg_pred_prob = np.ones_like(agg_pred_shares) / len(agg_pred_shares)
        else:
             agg_pred_prob = agg_pred_clipped_shares / agg_pred_sum
        agg_pred_prob_clamped = np.clip(agg_pred_prob, epsilon, 1.0)
        cross_entropy = -np.sum(agg_true_prob * np.log(agg_pred_prob_clamped))
        results_data[model_name] = list(agg_pred_shares) + [cross_entropy]
        print(f"    Agg prediction calculated. Cross-entropy vs true: {cross_entropy:.6f}")

    # --- 6. Assemble DataFrame ---
    target_cols = [t.replace('|C)', ')') for t in targets]
    all_cols = target_cols + ['entropy']
    eval_df = pd.DataFrame.from_dict(results_data, orient='index', columns=all_cols)

    # --- 7. Calculate P(underage) ---
    eval_df['P(underage)'] = 1.0 - eval_df[target_cols].sum(axis=1)
    eval_df['P(underage)'] = np.maximum(eval_df['P(underage)'], 0)

    # Reorder columns
    final_cols_order = target_cols + ['P(underage)', 'entropy']
    eval_df = eval_df[final_cols_order]

    print("\nFinal Evaluation Summary:")
    with pd.option_context('display.float_format', '{:.6f}'.format): print(eval_df)

    # --- 8. Optionally Save ---
    if save_path:
        print(f"\nSaving evaluation summary to: {save_path}")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            eval_df.to_csv(save_path, index=True, index_label='model', float_format='%.8f')
            print("Evaluation summary saved successfully.")
        except Exception as e: print(f"Error saving evaluation summary: {e}")

    print("--- Finished Evaluation ---")
    return eval_df