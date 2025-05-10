import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import os
import json
import copy
import joblib
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional, Union, List, Tuple, Type

from .constants import DEVICE
from .data_handling import DataHandler
from .constants import RESULTS_DIR, MODELS_DIR, PREDS_DIR
from .metrics import weighted_cross_entropy_loss

def suggest_mlp_params(trial: optuna.trial.Trial, depth: int):
    """Suggests learning rate, weight decay, hidden layer sizes, and dropout rates for an Optuna trial for an MLP."""
    params = {}
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    for i in range(1, depth + 1):
        params[f'n_hidden_{i}'] = trial.suggest_int(f'n_hidden_{i}', 8, 128, step=8)
        params[f'dropout_rate_{i}'] = trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.5, log=False)
    return params

def build_network(input_dim: int, depth: int, hparams: Dict[str, Any]):
    """Returns an MLP (nn.Sequential) with the specified input dimension, depth, and hyperparameters."""
    layers = []
    current_dim = input_dim

    for i in range(1, depth + 1):
        n_hidden = hparams[f"n_hidden_{i}"]
        layers.append(nn.Linear(current_dim, n_hidden))
        layers.append(nn.ReLU()) 

        dropout_rate = hparams.get(f"dropout_rate_{i}", 0.0)
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        current_dim = n_hidden

    layers.append(nn.Linear(current_dim, 4))
    layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)

def objective(trial: optuna.trial.Trial, 
              input_dim: int,  
              depth: int, 
              dataloaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]):
    """
    Objective function for MLP hyperparam optimization using Optuna. Returns the val loss for the trial. 
    """
    # Get the hyperparams for the current trial
    hparams = suggest_mlp_params(trial, depth)

    # Make dict with (model, optimizer) pairs for each fold
    model = build_network(input_dim, depth, hparams)
    pairs = {}
    for i in range(3):
        model_i = copy.deepcopy(model)
        optimizer_i = optim.AdamW(model_i.parameters(),
                                  lr=hparams['learning_rate'],
                                  weight_decay=hparams['weight_decay'])
        pairs[i] = (model_i, optimizer_i)

    # Train the models in sets of 10 epochs
    epoch_checkpoints = [10*i for i in range(1, 15)]
    for checkpoint in epoch_checkpoints:
        fold_losses = []
        for i, (train_loader, val_loader) in enumerate(dataloaders): # 3 folds
            model, optimizer = pairs[i] 
            for _ in range(1,11): # Train for 10 epochs
                model.train()
                for features, targets, weights in train_loader:
                    features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = weighted_cross_entropy_loss(outputs, targets, weights)
                    loss.backward()
                    optimizer.step()

            # Validate every 10 epochs
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, targets, weights in val_loader:
                    features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                    outputs = model(features)
                    loss = weighted_cross_entropy_loss(outputs, targets, weights)
                    val_loss += loss.item()
            # Append val loss for this fold
            fold_losses.append(val_loss / len(dataloaders))
            pairs[i] = (model, optimizer) # For next checkpoint

        # Prune    
        checkpoint_loss = np.mean(fold_losses)
        trial.report(checkpoint_loss, checkpoint)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    # Return the final checkpoint loss
    return checkpoint_loss



