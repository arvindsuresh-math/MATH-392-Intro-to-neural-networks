from sklearn.model_selection import ParameterGrid

hyperparams = {'lr_head': [5e-3, 1e-3],
               'lr_backbone': [1e-4, 5e-5],
               'weight_decay': [0.01, 0]}
hpo_configs = list(ParameterGrid(hyperparams))

print(f"Generated {len(hpo_configs)} hyperparameter configurations to test:")
hpo_configs

def run_hpo_trial(config: dict, hpo_epochs: int):
    """
    Trains a model configuration for a fixed number of epochs
    and returns the final validation accuracy.
    """
    print(f"--- Testing Config: {config} ---")
    start_time_trial = time.time()

    # 1. Load Model, adapt head, apply unfreeze-strategy, move to device
    model = utils.get_model(MODEL_NAME)
    model = utils.adapt_model_head(model, MODEL_NAME)
    model = utils.apply_unfreeze_logic(model, layers_to_unfreeze)
    model.to(device) 

    # 2. Load Datasets and DataLoaders
    train_dataset, val_dataset = utils.get_datasets(
        task = 'trainval',
        augment_train=AUGMENT_TRAIN
        )
    train_loader = utils.get_dataloaders(task='train', dataset=train_dataset, batch_size=BATCH_SIZE)
    val_loader = utils.get_dataloaders(task='val', dataset=val_dataset, batch_size=BATCH_SIZE)

    # 3. Setup Optimizer using current config
    optimizer = utils.get_optimizer(
        model=model,
        lr_head=config['lr_head'],
        lr_backbone=config['lr_backbone'],
        weight_decay=config['weight_decay']
    )

    # 4. Define Loss Function
    criterion = nn.CrossEntropyLoss()

    # 5. Training loop
    for epoch in range(hpo_epochs):
        # -- Training Step --
        model.train()
        running_loss = 0.0
        # Simple progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hpo_epochs} Train", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        # -- Validation Step -- (record the last one for HPO)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        # Simple progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{hpo_epochs} Val", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate validation accuracy for this epoch
        epoch_val_acc = 100 * correct / total
        epoch_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{hpo_epochs} - Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        # Store the results of the *last* epoch for this trial
        if epoch == hpo_epochs - 1:
             final_val_acc = epoch_val_acc
             final_val_loss = epoch_val_loss


    end_time_trial = time.time()
    print(f"--- Finished Config. Time: {end_time_trial - start_time_trial:.2f}s. Final Val Acc: {final_val_acc:.2f}% ---")

    # Return results for this configuration trial
    return {
        'config': config,
        'final_val_accuracy': final_val_acc,
        'final_val_loss': final_val_loss,
        'time_taken_secs': end_time_trial - start_time_trial
    }

# --- Main HPO Loop ---
# Find the best configuration based on final validation accuracy
best_trial = None
best_val_accuracy = -1.0

for trial in hpo_results:
    if trial['final_val_accuracy'] > best_val_accuracy:
        best_val_accuracy = trial['final_val_accuracy']
        best_trial = trial

# Extract the best hyperparameter configuration
best_hyperparameters = best_trial['config'] if best_trial else None

print("\n--- HPO Results ---")
if best_hyperparameters:
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Best Hyperparameters Found: {best_hyperparameters}")
else:
    print("No successful trials completed.")

# Define filenames for saving results
hpo_history_filename = f"hpo_history_{SETUP_ID}.json"
best_params_filename = f"best_params_{SETUP_ID}.json"

# Save the full HPO history (list of dictionaries)
utils.save_json(hpo_results, hpo_history_filename)
print(f"Full HPO history saved to: {hpo_history_filename}")

# Save only the best hyperparameter configuration
if best_hyperparameters:
    utils.save_json(best_hyperparameters, best_params_filename)
    print(f"Best hyperparameters saved to: {best_params_filename}")

# Find the best configuration based on final validation accuracy
best_trial = None
best_val_accuracy = -1.0

for trial in hpo_results:
    if trial['final_val_accuracy'] > best_val_accuracy:
        best_val_accuracy = trial['final_val_accuracy']
        best_trial = trial

best_hyperparameters = best_trial['config'] if best_trial else None

print("\n--- HPO Results ---")
if best_hyperparameters:
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Best Hyperparameters Found: {best_hyperparameters}")
else:
    print("No successful trials completed.")

# Define filenames for saving results
hpo_history_filename = f"hpo_history_{SETUP_ID}.json"
best_params_filename = f"best_params_{SETUP_ID}.json"

# Save the full HPO history (list of dictionaries)
utils.save_json(hpo_results, hpo_history_filename)
print(f"Full HPO history saved to: {hpo_history_filename}")

# Save only the best hyperparameter configuration
if best_hyperparameters:
    utils.save_json(best_hyperparameters, best_params_filename)
    print(f"Best hyperparameters saved to: {best_params_filename}")


# --- Final Model Training ---

# Load the best hyperparameters identified from hpo
print(f"Loading best hyperparameters from: {best_params_filename}")
# Add check if file exists? No error handling per request.
best_hyperparams = utils.load_json(best_params_filename)

print(f"Using hyperparameters: {best_hyperparams}")

# Extract specific hyperparameters for clarity
LR_HEAD = best_hyperparams['lr_head']
LR_BACKBONE = best_hyperparams['lr_backbone']
WEIGHT_DECAY = best_hyperparams['weight_decay']

# 1. Load Model, adapt Head, apply Unfreeze Strategy
model = utils.get_model(MODEL_NAME)
model = utils.adapt_model_head(model, MODEL_NAME)
model = utils.apply_unfreeze_logic(model, layers_to_unfreeze)
model.to(device) # Move model to device *after* modifications
print(f"Model '{MODEL_NAME}' adapted and layers unfrozen according to '{UNFREEZE_STRATEGY}'.")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params}, Trainable params: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")

# 2. Load Datasets and DataLoaders
train_dataset = utils.get_datasets(
    task = 'train',
    augment_train=AUGMENT_TRAIN
    )
train_loader = utils.get_dataloaders(task='train', dataset=train_dataset, batch_size=BATCH_SIZE)
print(f"DataLoader created. Train batches: {len(train_loader)}")

# 4. Setup Optimizer using loaded best hyperparameters
optimizer = utils.get_optimizer(
    model=model,
    lr_head=LR_HEAD,
    lr_backbone=LR_BACKBONE,
    weight_decay=WEIGHT_DECAY
)
print("Optimizer created with loaded best hyperparameters.")

# 5. Define Loss Function
criterion = nn.CrossEntropyLoss()
print("Loss function (CrossEntropyLoss) defined.")


def train_final_model(model, criterion, optimizer, train_loader, device, max_epochs, patience):
    """
    Trains the model on the provided training data.
    Implements early stopping based on training loss.
    Saves the model state dictionary with the lowest training loss in memory.
    Returns the best model state_dict and the training history.
    """
    history = {'train_loss': [], 'train_acc': []} # Only training metrics
    best_model_wts = None
    best_train_loss = float('inf') # Initialize best training loss to infinity
    epochs_no_improve = 0          # Counter for early stopping

    total_start_time = time.time()

    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{max_epochs} ---")

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc="Training", leave=False)

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # Accumulate loss weighted by batch size
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_pbar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        epoch_end_time = time.time()
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Time: {epoch_end_time - epoch_start_time:.2f}s")

        # --- Check for Improvement in Training Loss & Early Stopping ---
        # Note: Monitoring training loss for "improvement" (decrease) is unusual for early stopping.
        # Usually, one looks for validation loss to stop *increasing* or validation accuracy to stop *improving*.
        if epoch_train_loss < best_train_loss:
            print(f"Training loss improved ({best_train_loss:.4f} -> {epoch_train_loss:.4f}). Saving model...")
            best_train_loss = epoch_train_loss
            best_model_wts = copy.deepcopy(model.state_dict()) # Save the best weights
            epochs_no_improve = 0 # Reset counter
            # Save best model weights immediately to file (based on lowest training loss)
            torch.save(best_model_wts, best_model_filename) # best_model_filename defined in cell 1
            print(f"Best model (lowest training loss) weights saved to {best_model_filename}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in training loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs based on training loss.")
                break # Exit training loop

    total_end_time = time.time()
    print(f"\nTraining Finished. Total time: {(total_end_time - total_start_time)/60:.2f} minutes")
    print(f"Lowest Training Loss Achieved: {best_train_loss:.4f}")

    # Load best model weights back into model before returning
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return model, history


print("Starting final training process...")
final_model, training_history = train_final_model(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    device=device,
    max_epochs=2,
    patience=PATIENCE
)

print("Final training complete.")

# Save the full training history (list of dictionaries)
utils.save_json(training_history, train_history_filename)
print(f"Training history saved to: {train_history_filename}")


# 1. Instantiate Model Architecture
model = utils.get_model(MODEL_NAME) # get_model now only returns model
model = utils.adapt_model_head(model, MODEL_NAME)
model = utils.apply_unfreeze_logic(model, layers_to_unfreeze)
print(f"Model '{MODEL_NAME}' architecture instantiated for evaluation.")

# 2. Load Saved Best Model Weights
print(f"Loading best model weights from: {best_model_filename}")
model.load_state_dict(torch.load(best_model_filename, map_location=device))
model.to(device)
model.eval() # Set to evaluation mode
print("Best model weights loaded and model set to evaluation mode.")

# 3. Load Test Dataset 
test_dataset = utils.get_datasets(task='test')
print(f"Test Dataset loaded with {len(test_dataset)} images.")


# Get class names (assuming OxfordIIITPet structure)
if hasattr(test_dataset, '_breeds'): # OxfordIIITPet specific attribute
    class_names = test_dataset._breeds
elif hasattr(test_dataset, 'classes'): # Generic torchvision dataset attribute
    class_names = test_dataset.classes
else: # Fallback if class names are not directly accessible
    class_names = [str(i) for i in range(utils.NUM_CLASSES)]
print(f"Number of classes: {len(class_names)}")

print("Generating predictions on the test set (iterating directly over dataset)...")
all_predictions = []
all_true_labels = []
# Image indices will just be 0 to len(test_dataset)-1

with torch.no_grad():
    # Iterate directly over the dataset
    for i in tqdm(range(len(test_dataset)), desc="Testing"):
        image, true_label = test_dataset[i] # Get image and label

        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(device) # Add batch dimension for single image
        # label is already a scalar

        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

        all_predictions.append(predicted_class.item()) # Get scalar value
        all_true_labels.append(true_label)

# Store predictions in a more structured format
prediction_results = []
for i, (true_label, pred_label) in enumerate(zip(all_true_labels, all_predictions)):
    prediction_results.append({
        'image_id': i, # Simple index
        'true_label_idx': int(true_label),
        'predicted_label_idx': int(pred_label),
        'true_label_name': class_names[int(true_label)],
        'predicted_label_name': class_names[int(pred_label)]
    })

print(f"Predictions generated for {len(prediction_results)} test images.")

# Save predictions to csv
predictions_df = pd.DataFrame(prediction_results)
predictions_df.to_csv(predictions_filename, index=False)
print(f"Predictions saved to: {predictions_filename}")

print(f"\n--- Evaluation Metrics for Setup: {SETUP_ID_TO_EVAL} ---")

y_true = np.array(all_true_labels)
y_pred = np.array(all_predictions)

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0, output_dict=True)
print("\nClassification Report:")

# get accuracy and remove from the report
total_accuracy = report['accuracy']
print(f"\nOverall Test Accuracy: {total_accuracy * 100:.2f}%")
del report['accuracy'] 

# convert to DataFrame and make it more readable
report_df = pd.DataFrame(report).T
# convert to percentage with 2 decimal places
report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].apply(lambda x: x * 100).round(2)
# Convert support to int
report_df['support'] = report_df['support'].astype(int) 

# save to csv
report_df.to_csv(eval_metrics_filename)
print(f"Evaluation metrics saved to: {eval_metrics_filename}")

print(report_df)

