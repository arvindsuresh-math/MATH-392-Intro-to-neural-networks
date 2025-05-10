class MLPModel:
    """Handles Multi-Layer Perceptron CV, training, loading, and prediction."""

    def __init__(self, model_name: str = 'mlp'):
        """
        Initializes the MLPModel handler for a specific model configuration.
        Args:
            model_name (str): A unique name for this model configuration
                              (e.g., "softmax", "mlp1", "mlp2"). Used for saving files.
        """
        self.model_name: str = model_name
        self.model: Union[nn.Module, None] = None
        self.best_params: Dict = {}
        self.final_loss_history: List[Dict[str, Any]] = []

        # Define paths based on the specific model_name
        self.results_save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_cv_results.csv")
        self.state_dict_save_path = os.path.join(MODELS_DIR, f"{self.model_name}_final_state_dict.pth")
        self.loss_save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_final_training_loss.csv")
        self.pred_save_path = os.path.join(PREDS_DIR, f"{self.model_name}_predictions.csv")
 
    def train_final_model(self,
                          dh: DataHandler,
                          final_train_epochs: int,
                          batch_size: int = 64,
                          final_patience: int = 50 # For early stopping
                          ):
        """Trains the final NN model using the best hyperparams from CV."""
        print(f"\n--- Starting Final Model Training for {self.model_name.upper()} ---")

        if not self.best_params:
            self.best_params = self._parse_best_params_from_csv()
        print(f"Using best hyperparameters from CV: {self.best_params}")

        self.model = self._build_network(self.best_params, dh.input_dim).to(DEVICE)
        final_train_loader = dh.get_nn_data('train', batch_size)

        optimizer = optim.AdamW(self.model.parameters(),
                                lr=config['learning_rate'],
                                weight_decay=config.get('weight_decay', 0))

        best_loss = float('inf')
        epochs_no_improve = 0
        best_state_dict = None 

        self.model.train() 
        self.final_loss_history = []
        print(f"Starting final training for up to {final_train_epochs} epochs (Patience: {final_patience})...")

        for epoch in range(final_train_epochs):
            epoch_loss_sum = 0.0
            for features, targets, weights in final_train_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.weighted_cross_entropy_loss(outputs, targets, weights)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item()

            # Calculate mean batch loss for the epoch
            avg_epoch_loss = epoch_loss_sum / len(final_train_loader) 
            self.final_loss_history.append({'epoch': epoch + 1, 'loss': avg_epoch_loss})

            # Check for improvement based on average epoch loss
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_no_improve = 0
                best_state_dict = copy.deepcopy(self.model.state_dict()) # Save best state
            else:
                epochs_no_improve += 1

            # Log progress periodically
            if (epoch + 1) % 10 == 0 or epoch == final_train_epochs - 1 or epochs_no_improve >= final_patience:
                print(f"  Epoch {epoch+1}/{final_train_epochs} - Loss: {avg_epoch_loss:.6f} "
                      f"(Best Loss: {best_loss:.6f}, Epochs No Improve: {epochs_no_improve})")

            # Early stopping check
            if epochs_no_improve >= final_patience:
                print(f"  Early stopping triggered at epoch {epoch+1}.")
                break

        # Load the best performing model state before saving
        self.model.load_state_dict(best_state_dict)
        print(f"Loaded model state from epoch with best loss: {best_loss:.6f}")

        torch.save(self.model.state_dict(), self.state_dict_save_path)
        print(f"Saved best model state_dict to: {self.state_dict_save_path}")

        loss_df = pd.DataFrame(self.final_loss_history)
        loss_df.to_csv(self.loss_save_path, index=False)
        print(f"Saved training loss history to: {self.loss_save_path}")
        print(f"{self.model_name.upper()} training complete.")

        self.model.eval()

    def load_model(self, input_dim: int):
        """Loads a trained NN model state_dict, using hyperparameters from CV results."""
        print(f"Loading state_dict from: {self.state_dict_save_path}")
        print(f"Reading hyperparameters from: {self.results_save_path}")

        # 1. Build model architecture using subclass method and best params
        self.best_params = self._parse_best_params_from_csv()
        self.model = self._build_network(self.best_params, input_dim).to(DEVICE)

        # 2. Load the saved state dictionary onto the same device
        self.model.load_state_dict(torch.load(self.state_dict_save_path, map_location=DEVICE))
        print(f"{self.model_name.upper()} model loaded successfully onto {DEVICE}.")
        
        return self.model

    def predict(self, dh: DataHandler, save: bool = False):
        """
        Generates raw predictions (numpy array of shape [n_samples, 4]) for the test set using the trained NN model. If `save` is True, then saves the predictions to a CSV file.
        """
        print(f"\n--- Generating Predictions for {self.model_name.upper()} on Year {dh.test_year} ---")

        X_test, y_test = dh.get_nn_data('test', batch_size=1) 
        y_tots = y_test.sum(axis = 1, keepdims=True) # Probability of being 18plus by county

        X_test, y_test, y_tots = X_test.to(DEVICE), y_test.to(DEVICE), y_tots.to(DEVICE)
        self.model.eval()
        self.model.to(DEVICE)

        with torch.no_grad():
            outputs = self.model(X_test) 
            y_pred = outputs * y_tots # Shape [n_samples, 4]

        y_pred = y_pred.cpu().numpy()

        if save:
            pred_df = pd.DataFrame(y_pred, columns=dh.targets) # Use global 'targets'
            pred_df.to_csv(self.pred_save_path, index=False)
            print(f"Predictions for {self.model_name.upper()} on Year {dh.test_year} are saved to: {self.pred_save_path}")

        return y_pred



def objective(trial: optuna.trial.Trial, depth: int, dh: DataHandler):
    """
    Objective function for MLP hyperparam optimization using Optuna. Writes the trial results to "mlp_cv_history.json". Returns the best val loss for the trial. 
    """
    # Get the hyperparams and dataloaders for the current trial
    hparams = suggest_mlp_params(trial, depth)
    dataloaders = dh.get_nn_data('cv', batch_size=128)

    # Use joblib to parallelize the training across the 3 folds
    start_time = time.time()
    results = joblib.Parallel(n_jobs=3)(
                        joblib.delayed(train_one_fold)(dh.input_dim,
                                                    depth,
                                                    hparams,
                                                    train_loader,
                                                    val_loader,
                                                    150,
                                                    30)
                        for train_loader, val_loader in dataloaders
                        )
    best_val_loss = np.mean([result[0] for result in results])
    train_loss_at_best = np.mean([result[1] for result in results])
    best_epoch = np.mean([result[2] for result in results])
    time_taken = time.time() - start_time

    # update the JSON file with the current trial's results
    with open('mlp_cv_history.json', 'r') as f:
        history = json.load(f)
    history[str(trial.number)] = {
        'hparams': hparams,
        'best_val_loss': best_val_loss,
        'train_loss_at_best': train_loss_at_best,
        'best_epoch': best_epoch,
        'time_taken': time_taken
    }
    with open('mlp_cv_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    return best_val_loss


def train_one_fold(input_dim: int,
                   depth: int,
                   hparams: Dict[str, Any],
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   epochs: int,
                   patience: int):
    """Trains the model for a single fold in cross-validation. Returns a tuple (best_val_loss, train_loss_at_best, best_epoch)."""

    model = build_network(input_dim, depth, hparams).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),
                            lr=hparams['learning_rate'],
                            weight_decay=hparams['weight_decay'])

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        for features, targets, weights in train_loader:
            features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = weighted_cross_entropy_loss(outputs, targets, weights)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets, weights in val_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                outputs = model(features)
                loss = weighted_cross_entropy_loss(outputs, targets, weights)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience: break

    return best_val_loss