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
import joblib
import os

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def create_baseline_preds(y_series):
    """
    Creates the 'last-sign persistence' baseline predictions.
    """
    return y_series.shift(1).fillna(0).astype(int)

def save_artifacts(scaler, final_model, X, model_dir):
    """
    Save trained artifacts (scaler, model, feature names, mask) to the specified directory.
    """
    os.makedirs(model_dir, exist_ok=True)
    print(f"Saving artifacts to directory: {model_dir}")

    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(final_model, os.path.join(model_dir, 'final_model.pkl'))

    feature_names = np.array(X.columns)
    feature_mask = np.ones(len(feature_names), dtype=bool)

    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
    joblib.dump(feature_mask, os.path.join(model_dir, 'feature_mask.pkl'))

    print("Artifacts saved successfully.")

def main(args):
    """
    Main training and evaluation pipeline.
    """
    try:
        df = pd.read_csv(args.input_file, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Featured data file not found at {args.input_file}")
        sys.exit(1)

    X = df.drop('target', axis=1)
    y = df['target']

    test_size_n = int(args.test_size)
    if test_size_n >= len(X):
        print(f"Error: Test size ({test_size_n}) is larger than total dataset ({len(X)}).")
        sys.exit(1)

    X_train_val = X.iloc[:-test_size_n]
    y_train_val = y.iloc[:-test_size_n]
    X_test = X.iloc[-test_size_n:]
    y_test = y.iloc[-test_size_n:]

    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    val_f1_scores = []

    n_pos = (y_train_val == 1).sum()
    n_neg = (y_train_val == 0).sum()
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    if XGB_AVAILABLE:
        base_model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss',
                                   scale_pos_weight=scale_pos_weight, random_state=42)
    else:
        base_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_scaled)):
        X_train, X_val = X_train_val_scaled[train_index], X_train_val_scaled[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        m = base_model.__class__(**base_model.get_params())
        m.fit(X_train, y_train)
        val_preds = m.predict(X_val)
        fold_f1 = f1_score(y_val, val_preds)
        val_f1_scores.append(fold_f1)
        print(f"  Fold {fold+1}/{args.n_splits} - Validation F1: {fold_f1:.4f}")

    avg_val_f1 = np.mean(val_f1_scores)
    print(f"Average Validation F1: {avg_val_f1:.4f}")

    if XGB_AVAILABLE:
        final_model = XGBClassifier(n_estimators=300, use_label_encoder=False, eval_metric='logloss',
                                    scale_pos_weight=scale_pos_weight, random_state=42)
    else:
        final_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')

    final_model.fit(X_train_val_scaled, y_train_val)

    test_preds = final_model.predict(X_test_scaled)
    test_proba_up = final_model.predict_proba(X_test_scaled)[:, 1]
    test_f1 = f1_score(y_test, test_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_cm = confusion_matrix(y_test, test_preds)
    test_report = classification_report(y_test, test_preds, target_names=['Down (0)', 'Up (1)'], output_dict=True)

    baseline_preds_all = create_baseline_preds(y)
    baseline_preds_test = baseline_preds_all.loc[y_test.index]
    baseline_f1 = f1_score(y_test, baseline_preds_test)

    try:
        importances = final_model.feature_importances_
        feature_names = X.columns
        feature_importance_list = sorted(
            [(str(f), float(imp)) for f, imp in zip(feature_names, importances)],
            key=lambda x: x[1], reverse=True
        )
    except NotFittedError:
        feature_importance_list = []

    results = {
        'avg_validation_f1': avg_val_f1,
        'holdout_f1': test_f1,
        'holdout_accuracy': test_accuracy,
        'holdout_baseline_f1': baseline_f1,
        'model_beat_baseline': test_f1 > baseline_f1,
        'holdout_confusion_matrix': test_cm.tolist(),
        'holdout_classification_report': test_report,
        'holdout_predictions': np.asarray(test_preds).tolist(),
        'holdout_proba_up': np.asarray(test_proba_up).astype(float).tolist(),
        'holdout_actuals': np.asarray(y_test).astype(int).tolist(),
        'holdout_dates': y_test.index.strftime('%Y-%m-%d').tolist(),
        'feature_importances': feature_importance_list,
        'model_params': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                         for k, v in final_model.get_params().items()},
        'training_boundaries': {
            'train_val_start': X_train_val.index.min().strftime('%Y-%m-%d'),
            'train_val_end': X_train_val.index.max().strftime('%Y-%m-%d'),
            'test_start': X_test.index.min().strftime('%Y-%m-%d'),
            'test_end': X_test.index.max().strftime('%Y-%m-%d'),
        }
    }

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_sanitize(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return _sanitize(obj.tolist())
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(args.output_file, 'w') as f:
        json.dump(_sanitize(results), f, indent=4)

    save_artifacts(scaler, final_model, X, args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate PP price prediction model.")
    parser.add_argument("--input-file", type=str, default="featured_data.csv", help="Path to the featured data CSV.")
    parser.add_argument("--output-file", type=str, default="training_results.json", help="Path to save the training results JSON.")
    parser.add_argument("--test-size", type=int, default=24, help="Number of weekly samples for the holdout test set.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of splits for TimeSeriesSplit cross-validation.")
    parser.add_argument("--model-dir", type=str, default="model_artifacts", help="Directory to save trained model artifacts.")

    args = parser.parse_args()
    main(args)
