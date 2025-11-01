import pandas as pd
import argparse
import sys

def load_pp_data(filepath):
    """
    Loads the core Polypropylene (PP) weekly data.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    print(f"Loaded PP data with {len(df)} rows.")
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df[['Price']].rename(columns={'Price': 'pp_price'})
    df = df.asfreq('W-FRI')
    print(f"Resampled PP data to {len(df)} rows (W-FRI).")
    return df

def load_exog_data(filepath, name):
    """
    Loads and resamples daily exogenous data (like Oil or Gas) to weekly (Friday-ending).
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    print(f"Loaded {name} data with {len(df)} rows.")
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    if df['Price'].dtype == 'object':
        df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
    df_weekly = df[['Price']].resample('W-FRI').last()
    df_weekly = df_weekly.rename(columns={'Price': f"{name}_price"})
    print(f"Resampled {name} data to {len(df_weekly)} rows (W-FRI).")
    return df_weekly

def define_target(df, col='pp_price'):
    """
    Defines the binary target variable y_t.
    y_t = 1 if PP_{t+1} > PP_t
    y_t = 0 if PP_{t+1} <= PP_t
    """
    print("Defining target variable...")
    df['pp_price_next_week'] = df[col].shift(-1)
    df['target'] = (df['pp_price_next_week'] > df[col]).astype(int)
    df.dropna(subset=['pp_price_next_week'], inplace=True)
    df.drop(columns=['pp_price_next_week'], inplace=True)
    print("Target variable 'target' created.")
    return df

def main(args):
    """
    Main preprocessing pipeline.
    """
    pp_df = load_pp_data(args.pp_file)
    brent_df = load_exog_data(args.brent_file, 'brent')
    gas_df = load_exog_data(args.gas_file, 'gas')
    print("Joining all dataframes...")
    df_main = pp_df.join(brent_df, how='left')
    df_main = df_main.join(gas_df, how='left')
    df_main.ffill(inplace=True)
    initial_rows = len(df_main)
    df_main.dropna(inplace=True)
    if initial_rows > len(df_main):
        print(f"Dropped {initial_rows - len(df_main)} rows with initial NaNs.")
    if df_main.empty:
        print("Error: No data remaining after cleaning. Check input files and date ranges.")
        sys.exit(1)
    df_processed = define_target(df_main)
    df_processed.to_csv(args.output_file)
    print(f"Successfully preprocessed data and saved to {args.output_file}")
    print(f"Final processed data has {len(df_processed)} rows and columns: {list(df_processed.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess time series data for PP price prediction.")
    parser.add_argument("--pp-file", type=str, required=True, help="Path to the PP-BOPP-Film CSV file.")
    parser.add_argument("--brent-file", type=str, required=True, help="Path to the Brent Oil Futures CSV file.")
    parser.add_argument("--gas-file", type=str, required=True, help="Path to the Natural Gas Futures CSV file.")
    parser.add_argument("--output-file", type=str, default="processed_data.csv", help="Path to save the combined and processed CSV.")
    args = parser.parse_args()
    main(args)
