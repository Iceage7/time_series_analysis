import pandas as pd
import numpy as np
import argparse
import sys
import joblib

try:
    from preprocess import load_pp_data, load_exog_data
    from create_features import (
        add_lags, 
        add_rolling_features, 
        add_technical_indicators, 
        add_spreads_and_ratios, 
        detrend_and_decompose
    )
except ImportError:
    print("Error: Could not import from preprocess.py or create_features.py.")
    sys.exit(1)

def generate_inference_features(pp_path, brent_path, gas_path):
    pp_df = load_pp_data(pp_path)
    brent_df = load_exog_data(brent_path, 'brent')
    gas_df = load_exog_data(gas_path, 'gas')

    df_main = pp_df.join(brent_df, how='left')
    df_main = df_main.join(gas_df, how='left')

    df_main.ffill(inplace=True)
    df_main.dropna(inplace=True)

    price_cols = [col for col in df_main.columns if col.endswith('_price')]
    features_df = pd.DataFrame(index=df_main.index)

    for col in price_cols:
        temp_df = pd.DataFrame(index=df_main.index)
        temp_df[col] = df_main[col]

        if col == 'pp_price':
            df_main = detrend_and_decompose(df_main, col=col, period=52)
            temp_df[f'{col}_trend'] = df_main[f'{col}_trend']
            temp_df[f'{col}_seasonal'] = df_main[f'{col}_seasonal']
            temp_df[f'{col}_resid'] = df_main[f'{col}_resid']

        temp_df = add_lags(temp_df, col, lags=[1,2,3,4,8,13])
        temp_df = add_rolling_features(temp_df, col, windows=[4,8,12,16])
        temp_df = add_technical_indicators(temp_df, col)
        temp_df[f'{col}_diff_1'] = temp_df[col].diff(1)
        temp_df[f'{col}_diff_2'] = temp_df[col].diff(2)
        features_df = features_df.join(temp_df)

    features_df = add_spreads_and_ratios(features_df)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return features_df

def main(args):
    print("Loading artifacts...")
    try:
        model = joblib.load(args.model_file)
        scaler = joblib.load(args.scaler_file)
        feature_names = joblib.load(args.features_file)
    except FileNotFoundError as e:
        print(f"Error: Missing artifact file. {e}")
        sys.exit(1)

    print("Generating features from latest raw data...")
    all_features = generate_inference_features(args.pp_file, args.brent_file, args.gas_file)
    all_features.dropna(inplace=True)

    if all_features.empty:
        print("Error: No data remaining after feature engineering and NaN removal.")
        sys.exit(1)

    X_latest_unscaled = all_features.iloc[[-1]]
    last_date = X_latest_unscaled.index[0]

    try:
        X_latest_aligned = X_latest_unscaled[feature_names]
    except KeyError as e:
        print(f"Error: Feature mismatch. Missing feature: {e}")
        sys.exit(1)

    print(f"\nMaking prediction for week ending: {last_date.date()}...")
    X_latest_scaled = scaler.transform(X_latest_aligned)

    prediction = model.predict(X_latest_scaled)[0]
    probabilities = model.predict_proba(X_latest_scaled)[0]
    proba_up = probabilities[model.classes_ == 1][0]
    label = "Up" if prediction == 1 else "Down"

    importances = model.feature_importances_
    feature_importance_list = sorted(
        zip(feature_names, importances), 
        key=lambda x: x[1], 
        reverse=True
    )

    print("\n--- INFERENCE OUTPUT ---")
    print(f"  P(Up):                 {proba_up:.4f}")
    print(f"  Predicted Class:       {label}")
    print("\n  Top 5 Feature Drivers:")
    for feature, importance in feature_importance_list[:5]:
        print(f"    {feature:<30}: {importance:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for PP price direction.")
    parser.add_argument("--pp-file", type=str, required=True, help="Path to PP-BOPP-Film CSV file.")
    parser.add_argument("--brent-file", type=str, required=True, help="Path to Brent Oil Futures CSV file.")
    parser.add_argument("--gas-file", type=str, required=True, help="Path to Natural Gas Futures CSV file.")
    parser.add_argument("--model-file", type=str, default="final_model.pkl", help="Path to saved model file.")
    parser.add_argument("--scaler-file", type=str, default="scaler.pkl", help="Path to saved scaler file.")
    parser.add_argument("--features-file", type=str, default="feature_names.pkl", help="Path to saved feature names list.")
    args = parser.parse_args()
    main(args)
