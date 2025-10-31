import pandas as pd
import numpy as np
import json
import argparse
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def plot_confusion_matrix(cm, labels, filepath):
    """Plots and saves a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Holdout Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved confusion matrix to {filepath}")

def plot_predictions(dates, actuals, preds, proba, filepath):
    """Plots actuals vs. predictions on a timeline."""
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Actual': actuals,
        'Predicted': preds,
        'Probability_Up': proba
    })
    df.set_index('Date', inplace=True)
    
    plt.figure(figsize=(15, 7))
    
    # Plot actuals as points
    plt.plot(df.index, df['Actual'], 'bo-', label='Actual Direction (1=Up, 0=Down)', markersize=8)
    # Plot predictions as 'x'
    plt.plot(df.index, df['Predicted'], 'rx', label='Predicted Direction', markersize=8, alpha=0.7)
    
    # Plot probability
    plt.bar(df.index, df['Probability_Up'], width=3, label='P(Up)', color='green', alpha=0.3)
    
    plt.title('Actual vs. Predicted Price Direction (Holdout Set)')
    plt.legend()
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved prediction plot to {filepath}")

def main(args):
    """
    Main visualization pipeline.
    """
    try:
        with open(args.input_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.input_file}")
        print("Please run train.py first.")
        sys.exit(1)
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. Print Key Metrics ---
    print("--- Model Performance (Holdout Set) ---")
    print(f"  Average Validation F1: {results['avg_validation_f1']:.4f}")
    print(f"  Holdout F1 (Model):   {results['holdout_f1']:.4f}")
    print(f"  Holdout F1 (Baseline): {results['holdout_baseline_f1']:.4f}")
    print(f"  Model Beat Baseline:   {results['model_beat_baseline']}")
    print(f"  Holdout Accuracy:      {results['holdout_accuracy']:.4f}")
    print("\n--- Classification Report (Holdout Set) ---")
    # Re-generate report from lists for clean printing
    report = classification_report(
        results['holdout_actuals'], 
        results['holdout_predictions'], 
        target_names=['Down (0)', 'Up (1)']
    )
    print(report)
    
    # --- 2. Plot Confusion Matrix ---
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    cm = np.array(results['holdout_confusion_matrix'])
    plot_confusion_matrix(cm, labels=['Down (0)', 'Up (1)'], filepath=cm_path)
    
    # --- 3. Plot Predictions Timeline ---
    pred_path = os.path.join(args.output_dir, "prediction_timeline.png")
    plot_predictions(
        dates=results['holdout_dates'],
        actuals=results['holdout_actuals'],
        preds=results['holdout_predictions'],
        proba=results['holdout_proba_up'],
        filepath=pred_path
    )
    
    # --- 4. Print Top Features ---
    print("\n--- Top 10 Feature Importances ---")
    try:
        for feature, importance in results['feature_importances'][:10]:
            print(f"  {feature:<30}: {importance:.4f}")
    except KeyError:
        print("  Feature importances not found in results file.")
        
    print("\nVisualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model training results.")
    parser.add_argument("--input-file", type=str, default="training_results.json", help="Path to the training results JSON file.")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save output plots.")
    
    args = parser.parse_args()
    main(args)
