import torch
import torch.nn as nn
import time
import os
import argparse
from tqdm import tqdm
import copy 

import utils
import config

def perform_final_training_for_setup(
        setup_id: str,
        max_epochs: int,
        patience: int,
        batch_size: int,
        device
    ):
    """
    Runs final training for a single setup using best HPO params.
    Saves best_model .pth and final_training_history .json.
    """
    print(f"--- Starting Final Training for Setup ID: {setup_id} ---")
    model_name, unfreeze_key, augment_str = config.parse_setup_id(setup_id)
    layers_to_unfreeze = config.UNFREEZE_MAPS[model_name][unfreeze_key]

    # Load best HPO params
    best_params_filepath = os.path.join(config.HPO_DIR, f"best_params_{setup_id}.json")
    if not os.path.exists(best_params_filepath):
        print(f"  Error: Best HPO parameters file not found: {best_params_filepath}. Skipping final training.")
        return None
    best_hpo_params = utils.load_json(best_params_filepath)
    print(f"  Loaded best HPO params: {best_hpo_params}")

    # Model setup
    model = utils.get_model(model_name)
    model = utils.adapt_model_head(model, model_name)
    model = utils.apply_unfreeze_logic(model, layers_to_unfreeze)
    model.to(device)

    # Data setup (using 'train' task from utils.get_datasets for the full trainval set)
    train_dataset = utils.get_datasets(
        task='train', # This gets the full 'trainval' split from OxfordIIITPet
        augment_train=augment_str
    )
    train_loader = utils.get_dataloaders(task='train', dataset=train_dataset, batch_size=batch_size)

    # Optimizer
    optimizer = utils.get_optimizer(
        model=model,
        lr_head=best_hpo_params['lr_head'],
        lr_backbone=best_hpo_params['lr_backbone'],
        weight_decay=best_hpo_params['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    history = {'train_loss': [], 'train_acc': []}
    best_model_wts = None
    best_train_loss = float('inf')
    epochs_no_improve = 0
    saved_model_path = os.path.join(config.FINAL_TRAINING_DIR, f"best_model_{setup_id}.pth") 

    total_start_time = time.time()
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        print(f"  Epoch {epoch+1}/{max_epochs}")
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc="    Training", leave=False, ncols=100)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        epoch_end_time = time.time()
        print(f"    Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Time: {epoch_end_time - epoch_start_time:.2f}s")

        if epoch_train_loss < best_train_loss:
            print(f"    Training loss improved ({best_train_loss:.4f} -> {epoch_train_loss:.4f}). Saving model...")
            best_train_loss = epoch_train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_wts, saved_model_path) # Save to predefined path
            print(f"    Best model (lowest training loss) weights saved to {saved_model_path}")
        else:
            epochs_no_improve += 1
            print(f"    No improvement in training loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"    Early stopping triggered after {epoch+1} epochs based on training loss.")
                break
    total_end_time = time.time()
    print(f"  Training Finished. Total time: {(total_end_time - total_start_time)/60:.2f} minutes")
    print(f"  Lowest Training Loss Achieved: {best_train_loss:.4f}")

    # Save final training history
    history_filepath = os.path.join(config.FINAL_TRAINING_DIR, f"final_training_history_{setup_id}.json")
    utils.save_json(history, history_filepath)
    print(f"  Final training history for {setup_id} saved to: {history_filepath}")
    
    print(f"--- Finished Final Training for Setup ID: {setup_id} ---")
    return saved_model_path if best_model_wts else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Final Model Training for specified setups.")
    parser.add_argument('--setup_ids', nargs='+', required=True, help='List of setup_ids to run.')
    parser.add_argument('--final_train_epochs', type=int, default=config.DEFAULT_FINAL_TRAIN_EPOCHS, help='Max epochs for final training.')
    parser.add_argument('--patience', type=int, default=config.DEFAULT_PATIENCE, help='Early stopping patience.')
    parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE, help='Batch size for final training.')
    args = parser.parse_args()

    config.create_output_dirs()

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"Using device: {dev}")

    for sid in args.setup_ids:
        perform_final_training_for_setup(
            setup_id=sid,
            max_epochs=args.final_train_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=dev
        )
    print("\nAll Final Training tasks complete for specified setups.")