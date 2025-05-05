# -*- coding: utf-8 -*-
"""
Utility functions for the MATH-392 final project on fine-tuning.

Provides helper functions for:
- Model Loading: get_model
- Data Transformations: get_transforms
- Dataset Loading: get_datasets
- DataLoader Creation: get_dataloaders
- Model Head Adaptation: adapt_model_head
- Model Layer Unfreezing: apply_unfreeze_logic
- Optimizer Creation: get_optimizer
- JSON Saving/Loading: save_json, load_json
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import json
from typing import List, Dict, Tuple, Any # For type hinting arguments

# --- Constants ---
NUM_CLASSES = 37 # OxfordIIITPet dataset has 37 classes

# --- Model Loading ---

def get_model(model_name: str):
    """Loads a pre-trained model (ResNet18 or MobileNetV3-Small) and its weights object."""
    if model_name == 'resnet':
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
    elif model_name == 'mobilenet':
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        model = torchvision.models.mobilenet_v3_small(weights=weights)
    return model, weights

# --- Data Transformations ---

# --- Data Transformations (Simplified Version) ---

# --- Data Transformations (Simplified, Hardcoded Version) ---

def get_transforms(augment: bool):
    """Creates data transformations using hardcoded ImageNet stats and augmentations.

    Returns an augmented pipeline if augment=True, otherwise a simple resize/normalize pipeline.
    """
    # Hardcoded ImageNet stats and target image size
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    image_size = 224 # Standard size for ResNet/MobileNet

    # Define the standard validation/test transform (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), # Simple resize
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    if augment:
        # Define the specific augmented training transform
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
        return train_transform
    else:
        # Return the non-augmented transform for validation/test or non-augmented training
        return val_test_transform


# --- Dataset and DataLoader ---

# --- Dataset and DataLoader (with Train/Val Split) ---

def get_datasets(task: str,
                 root_dir: str = './data', 
                 augment_train: bool = False, 
                 weights_obj = None, 
                 val_split_ratio: float = 0.2, 
                 random_seed: int = 42):
    """Loads OxfordIIITPet, splits trainval into train/val, and returns train, val, test sets."""
    # Get transforms: potentially augmented for train, never for val/test
    train_transforms = get_transforms(augment=augment_train)
    val_test_transforms = get_transforms(augment=False)

    if task == 'test':
        # Load the test dataset (always uses non-augmented transforms)
        test_dataset = datasets.OxfordIIITPet(
            root=root_dir,
            split='test',
            download=True,
            transform=val_test_transforms
        )
        return test_dataset
    else:
        # Load the full trainval dataset *with non-augmented transforms* first
        full_trainval_dataset = datasets.OxfordIIITPet(
            root=root_dir,
            split='trainval',
            download=True,
            transform=val_test_transforms # Use non-augmented for splitting reference
        )

        # Split the dataset indices
        num_trainval = len(full_trainval_dataset)
        num_val = int(num_trainval * val_split_ratio)
        num_train = num_trainval - num_val
        generator = torch.Generator().manual_seed(random_seed) # For reproducible splits
        train_indices, val_indices = data.random_split(range(num_trainval), [num_train, num_val], generator=generator)

        # Create the train dataset using train_indices but applying train_transforms
        # We need to re-load the dataset with the correct transforms for the training subset
        train_dataset_reloaded = datasets.OxfordIIITPet(
            root=root_dir,
            split='trainval',
            download=False, # Already downloaded
            transform=train_transforms # Apply potentially augmented transforms
        )
        train_dataset = data.Subset(train_dataset_reloaded, train_indices)
        val_dataset = data.Subset(full_trainval_dataset, val_indices)
        print(f"Dataset split: Train samples = {len(train_dataset)}, Val samples = {len(val_dataset)}")
        return train_dataset, val_dataset 

def get_dataloaders(task: str, dataset, batch_size: int):
    """Creates train, validation, and test DataLoaders depending on task."""
    if task == 'test' or task == 'val':
        loader = data.DataLoader(dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    elif task == 'train':
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'train', 'val', or 'test'.")
    return loader

# --- Model Modification ---

def adapt_model_head(model, model_name: str):
    """Adapts the final classification layer of the model for NUM_CLASSES."""
    if model_name == 'resnet18':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    elif model_name == 'mobilenet_v3_small':
        # The classifier is a Sequential module; replace the last layer
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def apply_unfreeze_logic(model, layers_to_unfreeze: list):
    """Freezes all model layers, then unfreezes specified layers by name substring."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze parameters whose names contain any specified substring
    for name, param in model.named_parameters():
        for layer_substring in layers_to_unfreeze:
            if layer_substring in name:
                param.requires_grad = True
                break # Stop checking substrings for this parameter once unfrozen
    return model

# --- Optimizer ---

def get_optimizer(model, lr_head: float, lr_backbone: float, weight_decay: float):
    """Creates an AdamW optimizer with differential learning rates."""
    # Separate parameters into head (new classifier) and backbone (rest)
    head_params = []
    backbone_params = []
    head_param_names = ['fc.', 'classifier.'] # Substrings identifying head parameters

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Skip frozen parameters

        is_head_param = any(head_name in name for head_name in head_param_names)

        if is_head_param:
            head_params.append(param)
        else:
            backbone_params.append(param)

    # Create parameter groups for the optimizer
    param_groups = []
    if head_params: # Ensure the list is not empty
        param_groups.append({'params': head_params, 'lr': lr_head})
    if backbone_params: # Ensure the list is not empty
         param_groups.append({'params': backbone_params, 'lr': lr_backbone})
    # If backbone LR is 0 or very small, this effectively freezes those parts during optimization

    # Create the optimizer
    optimizer = optim.AdamW(param_groups, lr=lr_head, weight_decay=weight_decay)
    return optimizer

# --- File I/O ---

def save_json(data: dict, filename: str):
    """Saves a dictionary to a JSON file in the current directory."""
    filepath = f"./{filename}" 
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filename: str):
    """Loads a dictionary from a JSON file in the current directory."""
    filepath = f"./{filename}" 
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data