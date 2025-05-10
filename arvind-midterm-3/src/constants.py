"""
Global constants and device configuration for the election prediction project.

This module defines:
- Directory paths for data, models, results, predictions, and logs.
- The computing device (CPU, CUDA, MPS) to be used by PyTorch.
"""
import os
import torch

# --- File paths ---
DATA_DIR = "./data"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
LOGS_DIR = "./logs"
PREDS_DIR = "./preds"

# --- Device selection ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device")