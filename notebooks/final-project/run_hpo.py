import torch
import torch.nn as nn
import time
import os
import argparse
from tqdm import tqdm # Use standard tqdm for scripts
import copy # For deepcopy if needed, though less critical here

import utils
import config # Import our new config file

def perform_hpo_for_setup(
        setup_id: str,
        hpo_grid: list,
        num_epochs_hpo: int,
        batch_size: int,
        device
    ):
    """
    Runs HPO for a single setup.
    Saves detailed hpo_history (including epoch-wise train/val loss/acc for each trial)
    and best_params to config.HPO_DIR.
    Returns: dict of best hyperparameters or None if HPO fails.
    """
    print(f"--- Starting HPO for Setup ID: {setup_id} ---")
    model_name, unfreeze_key, augment_str = config.parse_setup_id(setup_id)
    layers_to_unfreeze = config.UNFREEZE_MAPS[model_name][unfreeze_key]
    augment_train_bool = (augment_str == 'aug')

    hpo_run_history = [] # To store results of all trials for this setup
    best_overall_val_accuracy = -1.0
    best_overall_params = None

    for trial_idx, hpo_config_params in enumerate(hpo_grid):
        print(f"  Trial {trial_idx+1}/{len(hpo_grid)} with params: {hpo_config_params}")
        trial_start_time = time.time()
        
        # Model setup for each trial (to ensure fresh weights if not careful, though pre-trained are fixed)
        model = utils.get_model(model_name)
        model = utils.adapt_model_head(model, model_name)
        model = utils.apply_unfreeze_logic(model, layers_to_unfreeze)
        model.to(device)

        # Data setup
        train_dataset, val_dataset = utils.get_datasets(
            task='trainval',
            augment_train=augment_str, # Pass string 'aug' or 'noaug'
            val_split_ratio=0.2 # Consistent validation split
        )
        train_loader = utils.get_dataloaders(task='train', dataset=train_dataset, batch_size=batch_size)
        val_loader = utils.get_dataloaders(task='val', dataset=val_dataset, batch_size=batch_size)

        # Optimizer
        current_lr_backbone = hpo_config_params['lr_backbone']
        if unfreeze_key == 'head' and current_lr_backbone > 0: # Ensure backbone LR is 0 if only head unfrozen
            print(f"    Adjusting lr_backbone to 0 for 'head' unfreeze strategy.")
            current_lr_backbone = 0

        optimizer = utils.get_optimizer(
            model=model,
            lr_head=hpo_config_params['lr_head'],
            lr_backbone=current_lr_backbone,
            weight_decay=hpo_config_params['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()

        trial_epoch_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        current_trial_best_val_acc = -1.0 # For this specific trial

        for epoch in range(num_epochs_hpo):
            # --- Training Step ---
            model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0
            train_pbar = tqdm(train_loader, desc=f"    Epoch {epoch+1} Train", leave=False, ncols=100)
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                train_pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
            
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            epoch_train_acc = 100 * correct_train / total_train
            trial_epoch_history['train_loss'].append(epoch_train_loss)
            trial_epoch_history['train_acc'].append(epoch_train_acc)

            # --- Validation Step ---
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            val_pbar = tqdm(val_loader, desc=f"    Epoch {epoch+1} Val  ", leave=False, ncols=100)
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            epoch_val_acc = 100 * correct_val / total_val
            trial_epoch_history['val_loss'].append(epoch_val_loss)
            trial_epoch_history['val_acc'].append(epoch_val_acc)
            
            print(f"    Epoch {epoch+1}/{num_epochs_hpo} - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")
            current_trial_best_val_acc = epoch_val_acc # Use accuracy of the final HPO epoch for this trial

        trial_end_time = time.time()
        hpo_run_history.append({
            'config_params': hpo_config_params,
            'final_val_accuracy_for_trial': current_trial_best_val_acc, # Accuracy at the end of this trial's HPO epochs
            'epoch_by_epoch_history': trial_epoch_history,
            'time_taken_secs': trial_end_time - trial_start_time
        })

        if current_trial_best_val_acc > best_overall_val_accuracy:
            best_overall_val_accuracy = current_trial_best_val_acc
            best_overall_params = hpo_config_params
            print(f"    New best HPO params for {setup_id}: {best_overall_params} with Val Acc: {best_overall_val_accuracy:.2f}%")

    # Save HPO history and best params
    hpo_history_filepath = os.path.join(config.HPO_DIR, f"hpo_history_{setup_id}.json")
    utils.save_json(hpo_run_history, hpo_history_filepath)
    print(f"  Full HPO history for {setup_id} saved to: {hpo_history_filepath}")

    if best_overall_params:
        best_params_filepath = os.path.join(config.HPO_DIR, f"best_params_{setup_id}.json")
        utils.save_json(best_overall_params, best_params_filepath)
        print(f"  Best hyperparameters for {setup_id} saved to: {best_params_filepath}")
    else:
        print(f"  No best parameters found for {setup_id} after HPO trials.")

    print(f"--- Finished HPO for Setup ID: {setup_id} ---")
    return best_overall_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Hyperparameter Optimization for specified setups.")
    parser.add_argument('--setup_ids', nargs='+', required=True, help='List of setup_ids to run (e.g., resnet_mid_aug).')
    parser.add_argument('--hpo_epochs', type=int, default=config.DEFAULT_HPO_EPOCHS, help='Number of epochs for each HPO trial.')
    parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE, help='Batch size for HPO.')
    args = parser.parse_args()

    config.create_output_dirs() # Ensure output directories exist

    # Determine device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"Using device: {dev}")

    for sid in args.setup_ids:
        perform_hpo_for_setup(
            setup_id=sid,
            hpo_grid=config.HPO_CONFIG_LIST,
            num_epochs_hpo=args.hpo_epochs,
            batch_size=args.batch_size,
            device=dev
        )
    print("\nAll HPO tasks complete for specified setups.")