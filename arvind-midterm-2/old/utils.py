# =============================================================================
# Utilities - Election Prediction Project
# =============================================================================
"""
Contains shared utility functions, constants, and mappings used across
different modules of the election prediction project. This includes device
selection and mappings for PyTorch components. Global target list is also
defined here after loading from variables.json.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

# --- Device Selection ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Mappings for Dynamic Instantiation ---
ACTIVATION_MAP = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid}
OPTIMIZER_MAP = {'AdamW': optim.AdamW, 'Adam': optim.Adam, 'SGD': optim.SGD, 'RMSprop': optim.RMSprop}
SCHEDULER_MAP = {'StepLR': optim.lr_scheduler.StepLR, 'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau}

# --- Feature and Target Definitions ---
# Load variables from the JSON file to define targets globally
try:
    with open('variables.json', 'r') as f:
        _vars_data = json.load(f)
    targets: List[str] = _vars_data['targets']
    years: List[int] = _vars_data['years']
    idx: List[str] = _vars_data['idx']
    # Compute all_features if needed elsewhere, or handle in DataHandler
    _feature_keys = set(_vars_data.keys()) - set(['targets', 'years', 'idx'])
    all_features: List[str] = [item for key in _feature_keys for item in _vars_data[key]]
except FileNotFoundError:
    print("Error: variables.json not found. Please ensure it's in the correct directory.")
    targets = [] # Define as empty list to avoid NameError later if file missing
    years = []
    idx = []
    all_features = []
except KeyError as e:
    print(f"Error: Key {e} not found in variables.json.")
    targets = []
    years = []
    idx = []
    all_features = []