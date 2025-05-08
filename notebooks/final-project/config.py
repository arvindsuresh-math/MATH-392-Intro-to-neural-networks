import os
from sklearn.model_selection import ParameterGrid

# --- Output Directory Paths ---
RESULTS_DIR = './results'
HPO_DIR = os.path.join(RESULTS_DIR, 'hpo')
FINAL_TRAINING_DIR = os.path.join(RESULTS_DIR, 'final_training')
EVALUATION_DIR = os.path.join(RESULTS_DIR, 'evaluation')

# --- Default Experiment Parameters ---
DEFAULT_HPO_EPOCHS = 15
DEFAULT_FINAL_TRAIN_EPOCHS = 50 # Max epochs for final training
DEFAULT_PATIENCE = 10           # Patience for early stopping in final training
DEFAULT_BATCH_SIZE = 64

# --- HPO Grid Definition ---
# Example Grid: Test a few Learning Rates and Weight Decay values
hpo_hyperparam_grid_definition = {
    'lr_head': [5e-3, 1e-3],
    'lr_backbone': [1e-4, 5e-5], # Include 0 for feature extraction comparison
    'weight_decay': [0.01, 0.001, 0]
}
HPO_CONFIG_LIST = list(ParameterGrid(hpo_hyperparam_grid_definition))

# --- Unfreeze Strategy Maps ---
RESNET_UNFREEZE_MAP = {
    'head': ['fc.'],
    'mid':  ['fc.', 'layer4.'],
    'deep': ['fc.', 'layer4.', 'layer3.']
}
MOBILENET_UNFREEZE_MAP = {
    'head': ['classifier.3.'],
    'mid':  ['classifier.3.', 'features.10.', 'features.11.', 'features.12.'],
    'deep': ['classifier.3.', 'features.8.', 'features.9.', 'features.10.', 'features.11.', 'features.12.']
}
UNFREEZE_MAPS = {
    'resnet': RESNET_UNFREEZE_MAP,
    'mobilenet': MOBILENET_UNFREEZE_MAP
}

# --- Helper Function ---
def parse_setup_id(setup_id_str: str):
    """
    Extracts model_name, unfreeze_key, and augment_str from a setup_id.
    Expected format: "modelname_unfreezekey_augmentstr"
    e.g., "resnet_mid_aug" or "mobilenet_head_noaug"
    """
    parts = setup_id_str.split('_')
    # Basic parsing, assumes correct format
    model_name = parts[0]
    unfreeze_key = parts[1]
    augment_str = parts[2] # 'aug' or 'noaug'
    return model_name, unfreeze_key, augment_str

def create_output_dirs():
    """Creates all necessary output directories if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HPO_DIR, exist_ok=True)
    os.makedirs(FINAL_TRAINING_DIR, exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    print("Output directories ensured.")