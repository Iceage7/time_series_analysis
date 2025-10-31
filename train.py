import pandas as pd
import numpy as np
import json
import argparse
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.exceptions import NotFittedError

def create_baseline_preds(y_series):
    """
    Creates the 'last-sign persistence' baseline predictions.
    Predicts this week's direction will be the same as last week's.
    y_baseline(t) = y_actual(t-1)
    """
    # Shift by 1, fill the first NaN with 0 (Down)
    return y_series.shift(1).fillna(0).astype(int)

def main(args):
    """
    Main training and evaluation pipeline.
    """
    try:
        df = pd.read_csv(args.input_file, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Featured data file not found at {args.input_file}")
        print("Please run feature_engineering.py first.")
        sys.exit(1)
        
    print(f"Loaded featured data with {len(df)} rows.")
    
    # 1. Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Create Holdout Split (Time-based)
    # The holdout set is the *last* N samples.
    test_size_n = int(args.test_size)
    if test_size_n >= len(X):
        print(f"Error: Test size ({test_size_n}) is larger than total dataset ({len(X)}).")
        sys.exit(1)
        
    X_train_val = X.iloc[:-test_size_n]
    y_train_val = y.iloc[:-test_size_n]
    X_test = X.iloc[-test_size_n:]
    y_test = y.iloc[-test_size_n:]
    
    print(f"Total samples: {len(X)}")
    print(f"Training/Validation samples: {len(X_train_val)} (from {X_train_val.index.min().date()} to {X_train_val.index.max().date()})")
    print(f"Holdout Test samples: {len(X_test)} (from {X_test.index.min().date()} to {X_test.index.max().date()})")
    
    # 3. Scale Data
    # CRITICAL: Fit scaler ONLY on X_train_val, then transform both.
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Cross-Validation (on training/validation set)
    # We use TimeSeriesSplit to respect temporal order.
    print("Running TimeSeriesSplit cross-validation...")
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    val_f1_scores = []
    for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_scaled)):
        X_train, X_val = X_train_val_scaled[train_index], X_train_val_scaled[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
        
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        fold_f1 = f1_score(y_val, val_preds)
        val_f1_scores.append(fold_f1)
        print(f"  Fold {fold+1}/{args.n_splits} - Validation F1: {fold_f1:.4f}")
        
    avg_val_f1 = np.mean(val_f1_scores)
    print(f"Average Validation F1: {avg_val_f1:.4f}")
    
    # 5. Final Training
    # Train the model on the *entire* training/validation set.
    print("Training final model on all training/validation data...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    final_model.fit(X_train_val_scaled, y_train_val)
    
    # 6. Evaluate on Holdout Test Set
    print("Evaluating model on holdout test set...")
    test_preds = final_model.predict(X_test_scaled)
    test_proba_up = final_model.predict_proba(X_test_scaled)[:, 1] # Probability of class 1 (Up)
    
    test_f1 = f1_score(y_test, test_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_cm = confusion_matrix(y_test, test_preds)
    test_report = classification_report(y_test, test_preds, target_names=['Down (0)', 'Up (1)'], output_dict=True)
    
    print(f"Holdout Test F1 Score: {test_f1:.4f}")
    print(f"Holdout Test Accuracy: {test_accuracy:.4f}")
    
    # 7. Evaluate Baseline Model
    print("Evaluating baseline model...")
    baseline_preds_all = create_baseline_preds(y)
    baseline_preds_test = baseline_preds_all.loc[y_test.index] # Align with test set
    
    baseline_f1 = f1_score(y_test, baseline_preds_test)
    print(f"Holdout Baseline F1 Score: {baseline_f1:.4f}")
    
    # 8. Feature Importances
    try:
        importances = final_model.feature_importances_
        feature_names = X.columns
        feature_importance_list = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    except NotFittedError:
        print("Warning: Model not fitted, cannot get feature importances.")
        feature_importance_list = []
        
    # 9. Save all results
    print(f"Saving training results to {args.output_file}...")
    results = {
        'avg_validation_f1': avg_val_f1,
        'holdout_f1': test_f1,
        'holdout_accuracy': test_accuracy,
        'holdout_baseline_f1': baseline_f1,
        'model_beat_baseline': test_f1 > baseline_f1,
        'holdout_confusion_matrix': test_cm.tolist(),
        'holdout_classification_report': test_report,
        'holdout_predictions': test_preds.tolist(),
        'holdout_proba_up': test_proba_up.tolist(),
        'holdout_actuals': y_test.tolist(),
        'holdout_dates': y_test.index.strftime('%Y-%m-%d').tolist(),
        'feature_importances': feature_importance_list,
        'model_params': final_model.get_params(),
        'training_boundaries': {
            'train_val_start': X_train_val.index.min().strftime('%Y-%m-%d'),
            'train_val_end': X_train_val.index.max().strftime('%Y-%m-%d'),
            'test_start': X_test.index.min().strftime('%Y-%m-%d'),
            'test_end': X_test.index.max().strftime('%Y-%m-%d'),
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Training pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate PP price prediction model.")
    parser.add_argument("--input-file", type=str, default="featured_data.csv", help="Path to the featured data CSV.")
    parser.add_argument("--output-file", type=str, default="training_results.json", help="Path to save the training results JSON.")
    parser.add_argument("--test-size", type=int, default=24, help="Number of weekly samples for the holdout test set (e.g., 24 = ~6 months).")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of splits for TimeSeriesSplit cross-validation.")
    
    args = parser.parse_args()
    main(args)
