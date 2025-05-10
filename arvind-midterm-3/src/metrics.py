"""
Custom metrics, loss functions, and objective functions for model training and evaluation.

Includes:
- weighted_cross_entropy_loss: For PyTorch models with soft labels and sample weights.
- softmax: Numerically stable softmax implementation.
- weighted_softprob_obj: Custom XGBoost objective for weighted cross-entropy.
- weighted_cross_entropy_eval: Custom XGBoost evaluation metric for weighted cross-entropy.
"""
import numpy as np
import torch
import xgboost as xgb # Primarily for type hinting DMatrix in XGBoost functions

# --- Weighted Cross-Entropy Loss for PyTorch ---

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

# --- Softmax for XGBoost Outputs ---

def softmax(x: np.ndarray):
    """Numerically stable softmax function."""
    # Subtract max for numerical stability before exp
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# --- XGBoost Custom Objective Function ---

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
    probs = softmax(preds)
    probs = probs * tots 

    # 4. Calculate Gradient: (q - p)
    grad = probs - labels

    # 5. Calculate Hessian (diagonal approximation): w * q * (1 - q)
    # hess = weights * probs * (tots - probs)
    # hess = np.maximum(hess, 1e-12) # Ensure non-negative hessian
    hess = probs * (1 - probs)
    hess = np.maximum(hess, 1e-12)

    return (grad,hess)

# --- XGBoost Custom Evaluation Metric ---

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
    probs = softmax(preds)
    probs = probs * tots

    # 4. Calculate weighted cross-entropy per sample
    epsilon = 1e-9
    probs = np.clip(probs, epsilon, 1. - epsilon)
    sample_surprisals = - labels * np.log(probs) #surprisal weighted by labels
    sample_loss = sample_surprisals.sum(axis=1) #sum across classes to get CE loss per sample

    # 5. Calculate average weighted cross-entropy
    weighted_avg_ce = np.average(sample_loss, weights=weights.flatten())

    return 'weighted-CE', weighted_avg_ce