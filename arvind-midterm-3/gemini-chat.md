**High-Level Overview of `election_project.py`:**

The script sets up a framework to predict US presidential election outcomes (probabilities for Democrat, Republican, Other candidates, and Non-voters) at the county level using demographic data from 2008, 2012, 2016, and 2020. It aims to compare the performance of Ridge Regression, Multi-Layer Perceptrons (MLPs), and XGBoost. The code is organized into classes for data handling and for each model type, promoting modularity and reusability. It includes functionalities for cross-validation, final model training, prediction, and evaluation.

**Class and Method Summaries:**

1.  **`WeightedStandardScaler` Class:**
    *   **Purpose:** Implements a custom StandardScaler that considers sample weights (`P(C)`) when calculating the mean and variance for feature scaling.
    *   **`__init__()`**: Initializes scaler attributes.
    *   **`fit(X, weights)`**: Computes weighted mean and standard deviation from `X` using `weights`.
    *   **`transform(X)`**: Scales `X` using the fitted mean and standard deviation.
    *   **`fit_transform(X, weights)`**: Combines fitting and transforming.
    *   **`inverse_transform(X)`**: Reverses the scaling transformation.

2.  **`DataHandler` Class:**
    *   **Purpose:** Manages all aspects of data loading, preprocessing (scaling), and splitting for different model types and operational phases (CV, training, testing).
    *   **`__init__(test_year, features_to_drop)`**: Loads raw data, defines features, targets, input/output dimensions, and year splits for training, testing, and 3-fold cross-validation. Initializes scalers for CV folds and the final training set.
    *   **`_load_data(train_years, test_year)`**: Internal helper to load and split data into (X, y, weights) tuples for given training and test/validation years.
    *   **`_fit_scaler(fit_years)`**: Internal helper to fit a `WeightedStandardScaler` on data from specified years.
    *   **`_create_tensors(data)`**: Internal helper to convert NumPy arrays (X, y, weights) into PyTorch tensors.
    *   **`_create_dataloader(data, batch_size, shuffle)`**: Internal helper to create PyTorch `DataLoader` instances.
    *   **`_create_dmatrix(data, only_X)`**: Internal helper to create `xgb.DMatrix` instances for XGBoost.
    *   **`get_ridge_data(task)`**: Returns appropriately scaled data (X, y, weights as NumPy arrays) for Ridge regression, based on `task` ('cv', 'train', 'test').
    *   **`get_nn_data(task, batch_size)`**: Returns data for Neural Networks. For 'cv', it provides a list of (train\_loader, val\_loader) tuples. For 'train', a train\_loader. For 'test', (X\_tensor, y\_tensor).
    *   **`get_xgb_data(task)`**: Returns `xgb.DMatrix` instances for XGBoost. For 'cv' or 'train', a DMatrix with all training data. For 'test', a DMatrix with only test features.

3.  **`RidgeModel` Class:**
    *   **Purpose:** Handles Ridge Regression, including cross-validation for alpha, final model training, saving/loading, and prediction.
    *   **`__init__(model_name)`**: Initializes the model name and paths for saving/loading.
    *   **`cross_validate(dh, param_grid, save)`**: Performs 3-fold cross-validation across the `param_grid` (list of alphas) using data from `DataHandler`. Stores the best alpha and saves CV results. Uses weighted MSE for evaluation.
    *   **`train_final_model(dh)`**: Trains the Ridge model on all training years using the best alpha (from attribute or loaded from CV results). Saves the trained model.
    *   **`load_model()`**: Loads a pre-trained Ridge model from a `.joblib` file.
    *   **`predict(dh, save)`**: Generates predictions on the test set using the trained model. Optionally saves predictions.

4.  **`MLPModel` Class:**
    *   **Purpose:** Manages Multi-Layer Perceptron models, including dynamic network creation, hyperparameter tuning via Successive Halving Algorithm (SHA), training with early stopping, and prediction.
    *   **`__init__(model_name)`**: Initializes model name, paths for saving/loading models, results, and loss history.
    *   **`weighted_cross_entropy_loss(outputs, targets, weights)` (static)**: Computes a custom batch-wise weighted cross-entropy loss.
    *   **`_build_network(params, input_dim)`**: Internal helper to construct an `nn.Sequential` MLP based on `hidden_layers` and `dropout_rate` in `params`.
    *   **`_parse_best_params_from_csv(results_df)`**: Internal helper to read and parse the best hyperparameters from the saved CV results CSV.
    *   **`_train_one_config(...)`**: Internal helper for SHA; trains and validates a single hyperparameter configuration across CV folds for a specified number of epochs (`rung_epochs`), managing model state and early stopping within the rung.
    *   **`_prune_config_history(cv_history_dict, n_promote)`**: Internal helper for SHA; prunes less promising configurations.
    *   **`cross_validate(dh, param_grid, rung_schedule, ...)`**: Implements SHA for hyperparameter tuning (e.g., hidden layers, dropout, learning rate, weight decay). Saves CV results and stores the best parameters.
    *   **`train_final_model(dh, final_train_epochs, ...)`**: Trains the final MLP on all training years using the best hyperparameters. Implements early stopping based on training loss. Saves the model's `state_dict` and training loss history.
    *   **`load_model(input_dim)`**: Loads a pre-trained MLP's `state_dict` and reconstructs the model using hyperparameters from CV results.
    *   **`predict(dh, save)`**: Generates predictions on the test set. Predictions are scaled by county-level `P(18plus)` (derived from `y_test`). Optionally saves predictions.

5.  **`XGBoostModel` Class:**
    *   **Purpose:** Handles XGBoost models, including cross-validation with custom objective/evaluation, final model training, saving/loading, and prediction.
    *   **`__init__(model_name)`**: Initializes model name and paths for saving/loading.
    *   **`softmax(x)` (static)**: Numerically stable softmax function.
    *   **`weighted_softprob_obj(preds, dtrain)` (static)**: Custom XGBoost objective function for weighted cross-entropy with soft probability labels.
    *   **`weighted_cross_entropy_eval(preds, dtrain)` (static)**: Custom XGBoost evaluation metric corresponding to the objective.
    *   **`_parse_best_params_from_csv()`**: Internal helper to read best hyperparameters and optimal boosting rounds from CV results.
    *   **`cross_validate(dh, param_grid, ...)`**: Performs 3-fold cross-validation using `xgb.cv` with the custom objective and evaluation metric. Uses early stopping to find optimal boosting rounds for each hyperparameter set. Saves CV results and stores the best parameters and rounds.
    *   **`train_final_model(dh)`**: Trains the final XGBoost model on all training years using the best hyperparameters and optimal boosting rounds. Saves the model.
    *   **`load_model()`**: Loads a pre-trained XGBoost model from a `.json` file.
    *   **`predict(dh, save)`**: Generates predictions on the test set. Applies softmax to raw outputs and scales by county-level `P(18plus)`. Optionally saves predictions.

6.  **`evaluate_predictions(pred_dict, dh, save)` Function:**
    *   **Purpose:** Calculates and compares performance metrics for predictions from multiple models.
    *   **Functionality:** Takes a dictionary of model predictions (`pred_dict`), true values from `DataHandler`. It calculates aggregate national vote shares for each party/non-voter, `P(underage)`, overall cross-entropy, and KL divergence against the true distribution for each model. Returns a summary DataFrame and optionally saves it.

**Workflow Logic:**

The intended workflow, as suggested by your "Example Usage" and class structure, is:

1.  **Initialization:**
    *   Create necessary directories (`./data`, `./models`, `./results`, `./preds`, `./logs`).
    *   Ensure `final_dataset.csv` and `variables.json` are in `./data/`.
    *   Instantiate `DataHandler`, specifying the `test_year`.

2.  **For each model type (Ridge, MLP, XGBoost):**
    *   **Instantiate Model Handler:** e.g., `ridge_handler = RidgeModel()`.
    *   **Cross-Validation:** Call `ridge_handler.cross_validate(data_handler, param_grid=ridge_alphas)`. This tunes hyperparameters and saves results to `./results/`.
    *   **Final Training:** Call `ridge_handler.train_final_model(data_handler)`. This uses the best hyperparameters from CV to train on all training years and saves the model to `./models/`.

3.  **Prediction Generation:**
    *   For each trained model, call its `predict(data_handler, save=True)` method. This loads the final model (if not already in memory), predicts on the test year, and saves county-level predictions to `./preds/`. Collect these predictions.

4.  **Evaluation:**
    *   Pass the collected predictions (as a dictionary `{'model_name': predictions_array, ...}`) and the `DataHandler` instance to `evaluate_predictions()`. This will compute and display/save a comparative performance summary.

This workflow allows for systematic model development, from data preparation and hyperparameter tuning to final training, prediction, and comparative evaluation, with clear separation of concerns between data handling and model-specific logic.

**Evaluation of Proposed Changes:**

1.  **Multiple Scripts (Separation of Concerns):**
    *   **Pros:**
        *   **Improved Readability & Maintainability:** Code becomes easier to understand, navigate, and modify.
        *   **Modularity:** Components (data handling, metrics, models) are independent, facilitating easier testing and reuse.
        *   **Collaboration:** Easier for multiple people to work on different parts simultaneously (less relevant for a solo project but good practice).
    *   **Cons:**
        *   **Increased Complexity (Slightly):** More files to manage, requires careful import organization. This is a minor trade-off for the benefits.

2.  **Joblib for Parallel CV (NN & XGBoost):**
    *   **Pros:**
        *   **Speed-up:** Significantly faster cross-validation, especially for NNs where training each fold/config can be time-consuming.
        *   **Efficient Resource Utilization:** Makes better use of multi-core CPUs.
    *   **Cons:**
        *   **Pickling Issues:** As you correctly pointed out, `joblib` (and Python's `multiprocessing`) requires functions and their arguments to be picklable. Instance methods of classes can sometimes be tricky if the class itself or its state isn't easily picklable. Moving core logic to top-level functions or static methods is a common solution.
        *   **Overhead:** For very fast individual tasks, the overhead of parallelization might negate benefits, but this is unlikely for model training folds.
        *   **XGBoost's `xgb.cv`:** This function already has built-in support for parallelizing fold computation (typically controlled by `nthread` or implied by `n_jobs` in parameters). You might find it simpler to leverage `xgb.cv`'s internal parallelism for folds and use `joblib` to parallelize the iteration over different *hyperparameter sets* if you have many. For NNs, `joblib` for folds/configs is ideal.

3.  **CLI for Running Processes (using `argparse`):**
    *   **Pros:**
        *   **Automation & Scripting:** Enables running experiments from the terminal, making it easy to script multiple runs, sweeps, or integrate into automated workflows.
        *   **Reproducibility:** Command-line arguments explicitly define the conditions of an experiment.
        *   **Decoupling from Notebooks:** Notebooks are great for exploration, but CLI is better for systematic runs and production-like execution.
        *   **Server/Cluster Execution:** Essential if you plan to run jobs on remote servers or clusters.
    *   **Cons:**
        *   **Initial Setup:** Requires writing parsing logic and designing a clear command-line interface.
        *   **Parameter Complexity:** Very complex hyperparameter grids might be cumbersome to pass entirely via CLI; often paired with configuration files.

**Overall, your proposed changes are highly beneficial and will significantly enhance your project's quality and utility.**

**High-Level Overview of Refactoring Implementation:**

Here's a step-by-step approach, without specific code:

**Phase 1: Restructure Files and Basic Separation**

1.  **Plan New Directory Structure:**
    *   `your_project_root/`
        *   `data/` (contains `final_dataset.csv`, `variables.json`)
        *   `models/` (for saved model artifacts)
        *   `results/` (for CV results, final evals)
        *   `preds/` (for prediction CSVs)
        *   `src/`
            *   `__init__.py`
            *   `constants.py` (for `DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`, `PREDS_DIR`, `LOGS_DIR`, `DEVICE` selection)
            *   `data_handling.py` (for `WeightedStandardScaler`, `DataHandler`)
            *   `metrics.py` (for `weighted_cross_entropy_loss`, `weighted_softprob_obj`, `weighted_cross_entropy_eval`)
            *   `ridge_model.py` (`RidgeModel` class)
            *   `mlp_model.py` (`MLPModel` class, `_build_network`)
            *   `xgboost_model.py` (`XGBoostModel` class, `softmax` if only used here)
            *   `utils.py` (optional, for common helper functions like directory creation, parsing params from CSVs)
        *   `run_pipeline.py` (main script with `argparse` for CV, training)
        *   `make_predictions.py` (script with `argparse` for prediction)
        *   `evaluate_results.py` (script with `argparse` for evaluation)

2.  **Move Code:**
    *   Relocate global constants and `DEVICE` logic to `src/constants.py`.
    *   Move `WeightedStandardScaler` and `DataHandler` to `src/data_handling.py`.
    *   Move custom loss/objective/eval functions to `src/metrics.py`.
    *   Separate each model class (`RidgeModel`, `MLPModel`, `XGBoostModel`) into its own file (`src/ridge_model.py`, etc.).
    *   The `evaluate_predictions` function could go into `evaluate_results.py` or a utility module.

3.  **Update Imports:** Adjust all import statements in the newly created files to reflect the new structure (e.g., `from src.data_handling import DataHandler`).

**Phase 2: Implement CLI with `argparse`**

1.  **`run_pipeline.py`:**
    *   Import `argparse`, `DataHandler`, model classes, etc.
    *   Define CLI arguments:
        *   `--model_type` (e.g., 'ridge', 'mlp', 'xgboost') - required.
        *   `--action` (e.g., 'cv', 'train_final') - required.
        *   `--test_year` (for `DataHandler` init).
        *   `--params_file` (optional path to a JSON/YAML file for HPO grids or fixed params to avoid overly long CLIs).
        *   Model-specific parameters for `cross_validate` or `train_final_model` (e.g., `--mlp_rung_schedule`, `--ridge_alphas`).
    *   Main logic:
        *   Parse args.
        *   Initialize `DataHandler`.
        *   Instantiate the chosen model handler (e.g., `model = MLPModel(...)`).
        *   If `action == 'cv'`, call `model.cross_validate(...)`, passing relevant arguments.
        *   If `action == 'train_final'`, call `model.train_final_model(...)`.

2.  **`make_predictions.py`:**
    *   CLI arguments: `--model_type`, `--test_year`, `--model_name` (to load the correct files), `--input_dim` (if needed for MLP model loading).
    *   Logic: Init `DataHandler`, init model handler, load model, call `predict`.

3.  **`evaluate_results.py`:**
    *   CLI arguments: `--test_year`, paths to prediction files (or a directory).
    *   Logic: Init `DataHandler`, load predictions, call `evaluate_predictions`.

**Phase 3: Refactor for `joblib` Parallelization (Focus on MLP CV)**

1.  **Identify Pickling-Sensitive Code:** The primary target is the CV loop within `MLPModel.cross_validate`, specifically the part that trains and evaluates one configuration on one fold (or `_train_one_config` if it handles multiple folds for one config).

2.  **Refactor MLP's `_train_one_config` (or its core):**
    *   **Goal:** Make the core logic (train/eval one config on one fold, or one config across all folds if SHA structure is kept that way) a top-level function in `mlp_model.py` or a static method.
    *   Example: `_train_mlp_config_fold_worker(config_params, train_loader, val_loader, input_dim, model_build_fn, loss_fn, device, epochs, patience) -> fold_results`.
        *   `model_build_fn` could be `MLPModel._build_network` (if static or passed appropriately).
        *   `loss_fn` would be `weighted_cross_entropy_loss` imported from `metrics.py`.
    *   This worker function will contain the training loop for a specific fold and configuration.

3.  **Modify `MLPModel.cross_validate`:**
    *   The outer SHA logic (managing rungs, promoting configs) will remain.
    *   Within each rung, when evaluating a set of configurations:
        *   For each configuration:
            *   Prepare data for each of the 3 CV folds.
            *   Use `joblib.Parallel(n_jobs=-1)(joblib.delayed(_train_mlp_config_fold_worker)(params_for_fold_i) for fold_i in folds)`.
            *   Collect results from parallel jobs and aggregate for that configuration.
        *   Alternatively, if `_train_one_config` itself manages the folds sequentially, you can parallelize the execution of `_train_one_config` for *different configurations* within a SHA rung.

4.  **XGBoost CV:**
    *   As discussed, `xgb.cv` handles its own fold parallelization.
    *   If your `param_grid` for XGBoost is large, the loop *iterating through this grid* in `XGBoostModel.cross_validate` can be parallelized using `joblib`. Each job would call `xgb.cv` for one set of hyperparameters.
    *   Example: `_run_xgb_cv_worker(config_params, dtrain, num_boost_round, ...) -> cv_result_for_config`.
    *   `XGBoostModel.cross_validate` would then use `joblib.Parallel` to call this worker for each config.

**Important Considerations:**

*   **State and Paths:** Functions moved outside classes or made static must receive all necessary information (like `input_dim`, `model_name` for file paths, `DEVICE`) as arguments.
*   **Configuration Files:** For complex HPO grids (especially for MLP and XGBoost), consider using a configuration file (e.g., JSON or YAML) that `argparse` can load. This keeps the CLI cleaner. Your `run_pipeline.py` would parse this file and pass the parameters to the respective model methods.
*   **Logging:** Implement proper logging (using Python's `logging` module) throughout your CLI scripts to track progress and errors, especially for long-running CV or training jobs.
*   **Testing:** With more modularity, you can write unit tests for individual components more easily (e.g., test `DataHandler` methods, test metric functions).

This refactoring is a significant but very worthwhile effort. Start with Phase 1 to get the structure right, then Phase 2 for CLI, and finally Phase 3 for parallelization, focusing on the most time-consuming parts first (likely MLP CV).