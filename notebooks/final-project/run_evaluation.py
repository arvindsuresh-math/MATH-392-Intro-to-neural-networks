# run_evaluation.py
import torch
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

import utils
import config

def perform_evaluation_for_setup(
        setup_id: str,
        device
    ):
    """
    Evaluates a trained model on the test set.
    Saves predictions .csv and eval_metrics .csv.
    """
    print(f"--- Starting Evaluation for Setup ID: {setup_id} ---")
    model_name, unfreeze_key, _ = config.parse_setup_id(setup_id) # augment_str not needed for eval model structure
    layers_to_unfreeze = config.UNFREEZE_MAPS[model_name][unfreeze_key]

    # Model setup
    model_path = os.path.join(config.FINAL_TRAINING_DIR, f"best_model_{setup_id}.pth")
    model = utils.get_model(model_name)
    model = utils.adapt_model_head(model, model_name)
    model = utils.apply_unfreeze_logic(model, layers_to_unfreeze)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Loaded model from {model_path}")

    # Data setup
    test_dataset = utils.get_datasets(task='test', augment_train='noaug')
    
    # Get class names
    if hasattr(test_dataset, '_breeds'): class_names = test_dataset._breeds
    elif hasattr(test_dataset, 'classes'): class_names = test_dataset.classes
    else: class_names = [str(i) for i in range(utils.NUM_CLASSES)]

    # --- Prediction Loop ---
    all_predictions_indices = []
    all_true_labels_indices = []
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="    Testing", leave=False, ncols=100):
            image, true_label = test_dataset[i]
            image = image.unsqueeze(0).to(device)
            outputs = model(image)
            _, predicted_class = torch.max(outputs, 1)
            all_predictions_indices.append(predicted_class.item())
            all_true_labels_indices.append(true_label)

    prediction_results_list_of_dicts = []
    for i, (true_idx, pred_idx) in enumerate(zip(all_true_labels_indices, all_predictions_indices)):
        prediction_results_list_of_dicts.append({
            'image_id': i,
            'true_label_idx': int(true_idx),
            'predicted_label_idx': int(pred_idx),
            'true_label_name': class_names[int(true_idx)],
            'predicted_label_name': class_names[int(pred_idx)]
        })
    
    predictions_df = pd.DataFrame(prediction_results_list_of_dicts)
    predictions_filepath = os.path.join(config.EVALUATION_DIR, f"predictions_{setup_id}.csv")
    predictions_df.to_csv(predictions_filepath, index=False)
    print(f"  Predictions saved to: {predictions_filepath}")

    # --- Metrics Calculation ---
    y_true = np.array(all_true_labels_indices)
    y_pred = np.array(all_predictions_indices)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0, output_dict=True)
    
    overall_accuracy = report_dict['accuracy']
    del report_dict['accuracy'] # Remove to avoid confusion in DataFrame
    
    report_df = pd.DataFrame(report_dict).T
    report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].apply(lambda x: x * 100).round(2)
    report_df['support'] = report_df['support'].astype(int)
    
    # Add overall accuracy back as a specific row or separate entry if desired for clarity
    # For now, it's implicitly part of the classification_report output_dict
    # Or save it in the main comparison dataframe later.

    eval_metrics_filepath = os.path.join(config.EVALUATION_DIR, f"eval_metrics_{setup_id}.csv")
    report_df.to_csv(eval_metrics_filepath)
    print(f"  Evaluation metrics saved to: {eval_metrics_filepath}")
    print(f"  Overall Test Accuracy for {setup_id}: {overall_accuracy * 100:.2f}%")
    print(f"--- Finished Evaluation for Setup ID: {setup_id} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Model Evaluation for specified setups.")
    parser.add_argument('--setup_ids', nargs='+', required=True, help='List of setup_ids to evaluate.')
    # No batch_size needed here as we iterate one by one
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
        perform_evaluation_for_setup(
            setup_id=sid,
            device=dev
        )
    print("\nAll Evaluation tasks complete for specified setups.")