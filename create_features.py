import pandas as pd
import numpy as np
import ta  # Technical Analysis library
import argparse
import sys

def add_lags(df, col, lags=[1, 2, 4, 8]):
    """Adds lagged features for a given column."""
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def add_rolling_features(df, col, windows=[4, 8, 12]):
    """Adds rolling mean and rolling std dev features."""
    for window in windows:
        df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
        df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
    return df

def add_technical_indicators(df, col):
    """Adds technical indicators like RSI, MACD, and Bollinger Bands."""
    # RSI
    df[f'{col}_rsi'] = ta.momentum.RSIIndicator(df[col], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df[col], window_slow=26, window_fast=12, window_sign=9)
    df[f'{col}_macd'] = macd.macd()
    df[f'{col}_macd_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df[col], window=20, window_dev=2)
    df[f'{col}_bb_high_ind'] = bollinger.bollinger_hband_indicator()
    df[f'{col}_bb_low_ind'] = bollinger.bollinger_lband_indicator()
    
    # Weekly Returns
    df[f'{col}_return'] = df[col].pct_change()
    
    return df

def add_spreads_and_ratios(df):
    """Adds spreads and ratios between different prices."""
    # Check if columns exist before creating spreads
    if 'pp_price' in df.columns and 'brent_price' in df.columns:
        df['pp_brent_spread'] = df['pp_price'] - df['brent_price']
        df['pp_brent_ratio'] = df['pp_price'] / df['brent_price']
        
    if 'pp_price' in df.columns and 'gas_price' in df.columns:
        df['pp_gas_spread'] = df['pp_price'] - df['gas_price']
        
    if 'brent_price' in df.columns and 'gas_price' in df.columns:
        df['brent_gas_ratio'] = df['brent_price'] / df['gas_price']
        
    return df

def main(args):
    """
    Main feature engineering pipeline.
    """
    try:
        df = pd.read_csv(args.input_file, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {args.input_file}")
        print("Please run preprocessing.py first.")
        sys.exit(1)
        
    print(f"Loaded processed data. Starting feature engineering...")
    
    # Separate target variable. We MUST NOT generate features from it.
    if 'target' not in df.columns:
        print(f"Error: Target column not in {args.input_file}. Check preprocessing.py")
        sys.exit(1)
        
    target = df[['target']]
    # Features will be generated from the price columns
    price_cols = [col for col in df.columns if col.endswith('_price')]
    
    # Initialize features dataframe
    features_df = pd.DataFrame(index=df.index)
    
    # Generate features for all price columns
    for col in price_cols:
        print(f"Generating features for: {col}")
        # Make a temporary df for calculations to avoid fragmentation
        temp_df = pd.DataFrame(index=df.index)
        temp_df[col] = df[col] # Add the base price itself as a feature
        
        temp_df = add_lags(temp_df, col)
        temp_df = add_rolling_features(temp_df, col)
        temp_df = add_technical_indicators(temp_df, col)
        
        # Add all newly generated features to the main features_df
        features_df = features_df.join(temp_df)
    
    print("Generating spreads and ratios...")
    features_df = add_spreads_and_ratios(features_df)
    
    # Combine engineered features with the target
    final_df = features_df.join(target)
    
    # --- Critical Step: Clean NaNs ---
    # Features like rolling means and TIs create NaNs at the start.
    # We must drop these rows to have a clean dataset for the model.
    initial_rows = len(final_df)
    final_df.dropna(inplace=True)
    print(f"Dropped {initial_rows - len(final_df)} rows due to NaN values from feature generation.")
    
    # Replace any infinite values (e.g., from ratios) with NaN and drop again
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(inplace=True)
    
    if final_df.empty:
        print("Error: No data remaining after feature engineering and NaN removal.")
        print("Consider using shorter lookback windows or providing more initial data.")
        sys.exit(1)
        
    # Save the final dataset
    final_df.to_csv(args.output_file)
    print(f"Successfully engineered features and saved to {args.output_file}")
    print(f"Final featured data has {len(final_df)} rows and {len(final_df.columns)} columns (including target).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer features for PP price prediction.")
    parser.add_argument("--input-file", type=str, default="processed_data.csv", help="Path to the processed data CSV.")
    parser.add_argument("--output-file", type=str, default="featured_data.csv", help="Path to save the final featured CSV.")
    
    args = parser.parse_args()
    main(args)
