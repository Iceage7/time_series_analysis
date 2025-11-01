import pandas as pd
import numpy as np
import ta
import argparse
import sys
from statsmodels.tsa.seasonal import STL
from scipy import stats

def add_lags(df, col, lags=[1, 2, 4, 8]):
    """Adds lagged features for a given column."""
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def add_rolling_features(df, col, windows=[4, 8, 12]):
    """Adds rolling mean, std, and slope features."""
    for window in windows:
        df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
        df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
        df[f'{col}_roll_slope_{window}'] = df[col].rolling(window=window).apply(
            lambda x: np.nan if x.isnull().any() else np.polyfit(np.arange(len(x)), x, 1)[0]
        )
    return df

def add_technical_indicators(df, col):
    """Adds technical indicators like RSI, MACD, Bollinger Bands, returns, and EMAs."""
    df[f'{col}_rsi'] = ta.momentum.RSIIndicator(df[col], window=14).rsi()
    macd = ta.trend.MACD(df[col], window_slow=26, window_fast=12, window_sign=9)
    df[f'{col}_macd'] = macd.macd()
    df[f'{col}_macd_signal'] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(df[col], window=20, window_dev=2)
    df[f'{col}_bb_high_ind'] = bollinger.bollinger_hband_indicator()
    df[f'{col}_bb_low_ind'] = bollinger.bollinger_lband_indicator()
    df[f'{col}_return'] = df[col].pct_change()
    df[f'{col}_ema_4'] = df[col].ewm(span=4, adjust=False).mean()
    df[f'{col}_ema_12'] = df[col].ewm(span=12, adjust=False).mean()
    df[f'{col}_ema_diff'] = df[f'{col}_ema_4'] - df[f'{col}_ema_12']
    return df

def add_spreads_and_ratios(df):
    """Adds spreads and ratios between different prices."""
    if 'pp_price' in df.columns and 'brent_price' in df.columns:
        df['pp_brent_spread'] = df['pp_price'] - df['brent_price']
        df['pp_brent_ratio'] = df['pp_price'] / df['brent_price']
    if 'pp_price' in df.columns and 'gas_price' in df.columns:
        df['pp_gas_spread'] = df['pp_price'] - df['gas_price']
    if 'brent_price' in df.columns and 'gas_price' in df.columns:
        df['brent_gas_ratio'] = df['brent_price'] / df['gas_price']
    return df


def detrend_and_decompose(df, col='pp_price', period=52, window=50):
    trend, seasonal, resid = [], [], []
    for i in range(len(df)):
        if i < window:
            trend.append(np.nan)
            seasonal.append(np.nan)
            resid.append(np.nan)
            continue
        series = df[col].iloc[i-window:i]  # only past values
        if series.isna().any():
            trend.append(np.nan)
            seasonal.append(np.nan)
            resid.append(np.nan)
            continue
        res = STL(series, period=period, robust=True).fit()
        trend.append(res.trend.iloc[-1])
        seasonal.append(res.seasonal.iloc[-1])
        resid.append(res.resid.iloc[-1])
    df[f'{col}_trend'] = trend
    df[f'{col}_seasonal'] = seasonal
    df[f'{col}_resid'] = resid
    return df


def main(args):
    """Main feature engineering pipeline."""
    try:
        df = pd.read_csv(args.input_file, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {args.input_file}")
        print("Please run preprocessing.py first.")
        sys.exit(1)
    print(f"Loaded processed data. Starting feature engineering...")
    if 'target' not in df.columns:
        print(f"Error: Target column not in {args.input_file}. Check preprocessing.py")
        sys.exit(1)
    target = df[['target']]
    price_cols = [col for col in df.columns if col.endswith('_price')]
    features_df = pd.DataFrame(index=df.index)
    for col in price_cols:
        print(f"Generating features for: {col}")
        temp_df = pd.DataFrame(index=df.index)
        temp_df[col] = df[col]
        if col == 'pp_price':
            # pass
            # lekage free 
            df = detrend_and_decompose(df, col=col, period=52)
            temp_df[f'{col}_trend'] = df[f'{col}_trend']
            temp_df[f'{col}_seasonal'] = df[f'{col}_seasonal']
            temp_df[f'{col}_resid'] = df[f'{col}_resid']
        temp_df = add_lags(temp_df, col, lags=[1,2,3,4,8,13])
        temp_df = add_rolling_features(temp_df, col, windows=[4,8,12,16])
        temp_df = add_technical_indicators(temp_df, col)
        temp_df[f'{col}_diff_1'] = temp_df[col].diff(1)
        temp_df[f'{col}_diff_2'] = temp_df[col].diff(2)
        features_df = features_df.join(temp_df)
    print("Generating spreads and ratios...")
    features_df = add_spreads_and_ratios(features_df)
    final_df = features_df.join(target)
    initial_rows = len(final_df)
    final_df.dropna(inplace=True)
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(inplace=True)
    if final_df.empty:
        print("Error: No data remaining after feature engineering and NaN removal.")
        print("Consider using shorter lookback windows or providing more initial data.")
        sys.exit(1)
    final_df.to_csv(args.output_file)
    print(f"Successfully engineered features and saved to {args.output_file}")
    print(f"Final featured data has {len(final_df)} rows and {len(final_df.columns)} columns (including target).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer features for PP price prediction.")
    parser.add_argument("--input-file", type=str, default="processed_data.csv", help="Path to the processed data CSV.")
    parser.add_argument("--output-file", type=str, default="featured_data.csv", help="Path to save the final featured CSV.")
    args = parser.parse_args()
    main(args)
