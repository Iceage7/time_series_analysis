#!/usr/bin/env python3
"""
Visualize ML training results from JSON file.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_results(json_file):
    """Load training results from a JSON file."""
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results

def plot_predictions(dates, y_true, y_pred, title, save_path, color='red'):
    """Plot predictions versus ground truth."""
    plt.figure(figsize=(14,4))
    plt.plot(dates, y_true, label='Ground Truth', marker='o', linestyle='-', color='blue')
    plt.plot(dates, y_pred, label='Prediction', marker='x', linestyle='--', color=color)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".pdf")
    plt.close()

def plot_confusion(y_true, y_pred, labels, title, save_path):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".pdf")
    plt.close()

def plot_feature_importances(feature_importances, save_path, top_n=10):
    """Plot top N feature importances as a horizontal bar chart."""
    if not feature_importances:
        return
    sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    plt.figure(figsize=(10,6))
    sns.barplot(x=values, y=names, palette="viridis")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".pdf")
    plt.close()

def main(args):
    """Main visualization pipeline."""
    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.input_file)
    
    dates = pd.to_datetime(results['holdout_dates'])
    y_true = np.array(results['holdout_actuals'])
    y_pred_model = np.array(results['holdout_predictions'])
    y_pred_baseline = np.array(results.get('holdout_baseline_predictions', []))
    feature_importances = results.get('feature_importances', [])
    
    plot_predictions(dates, y_true, y_pred_model,
                     "Model Predictions vs Ground Truth",
                     os.path.join(args.output_dir, "model_vs_ground_truth"))
    
    if len(y_pred_baseline) == len(y_true):
        plot_predictions(dates, y_true, y_pred_baseline,
                         "Baseline Predictions vs Ground Truth",
                         os.path.join(args.output_dir, "baseline_vs_ground_truth"),
                         color='orange')
    
    plot_confusion(y_true, y_pred_model, labels=[0,1],
                   title="Confusion Matrix: Model",
                   save_path=os.path.join(args.output_dir, "confusion_matrix_model"))
    
    if len(y_pred_baseline) == len(y_true):
        plot_confusion(y_true, y_pred_baseline, labels=[0,1],
                       title="Confusion Matrix: Baseline",
                       save_path=os.path.join(args.output_dir, "confusion_matrix_baseline"))
    
    plot_feature_importances(feature_importances,
                             save_path=os.path.join(args.output_dir, "feature_importances"),
                             top_n=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model training results.")
    parser.add_argument("--input-file", type=str, default="training_results.json", help="Path to the training results JSON file.")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save output plots.")
    args = parser.parse_args()
    main(args)
