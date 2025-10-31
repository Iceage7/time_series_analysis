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
    # Rename 'Price' to 'pp_price' for clarity
    df = df[['Price']].rename(columns={'Price': 'pp_price'})
    # Ensure data is on a consistent weekly frequency (Friday)
    df = df.asfreq('W-FRI')
    print(f"Resampled PP data to {len(df)} rows (W-FRI).")
    return df

def load_exog_data(filepath, name):
    """
    Loads and resamples daily exogenous data (like Oil or Gas) to weekly (Friday-ending).
    We use .last() to get the closing price of the week.
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
    
    # Clean price data
    if df['Price'].dtype == 'object':
        df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
        
    # Resample daily data to weekly, taking the last price of the week
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
    # Shift price by -1 to get next week's price
    df['pp_price_next_week'] = df[col].shift(-1)
    
    # Define target
    df['target'] = (df['pp_price_next_week'] > df[col]).astype(int)
    
    # Drop the last row, which will have no target
    df.dropna(subset=['pp_price_next_week'], inplace=True)
    
    # We can now drop the helper column
    df.drop(columns=['pp_price_next_week'], inplace=True)
    print("Target variable 'target' created.")
    return df

def main(args):
    """
    Main preprocessing pipeline.
    """
    # 1. Load data
    pp_df = load_pp_data(args.pp_file)
    brent_df = load_exog_data(args.brent_file, 'brent')
    gas_df = load_exog_data(args.gas_file, 'gas')
    
    # 2. Combine dataframes
    print("Joining all dataframes...")
    # Start with the core PP data, then join the others
    df_main = pp_df.join(brent_df, how='left')
    df_main = df_main.join(gas_df, how='left')
    
    # 3. Handle missing values
    # We use forward fill (ffill) assuming that if a price is missing
    # for a week, it's likely the same as the previous week's close.
    df_main.ffill(inplace=True)
    
    # Drop any remaining NaNs (e.g., at the very beginning of the series)
    initial_rows = len(df_main)
    df_main.dropna(inplace=True)
    if initial_rows > len(df_main):
        print(f"Dropped {initial_rows - len(df_main)} rows with initial NaNs.")
        
    if df_main.empty:
        print("Error: No data remaining after cleaning. Check input files and date ranges.")
        sys.exit(1)

    # 4. Define target variable
    df_processed = define_target(df_main)
    
    # 5. Save processed data
    df_processed.to_csv(args.output_file)
    print(f"Successfully preprocessed data and saved to {args.output_file}")
    print(f"Final processed data has {len(df_processed)} rows and columns: {list(df_processed.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess time series data for PP price prediction.")
    
    # File paths from user
    parser.add_argument("--pp-file", type=str, required=True, help="Path to the PP-BOPP-Film CSV file.")
    parser.add_argument("--brent-file", type=str, required=True, help="Path to the Brent Oil Futures CSV file.")
    parser.add_argument("--gas-file", type=str, required=True, help="Path to the Natural Gas Futures CSV file.")
    
    # Output path
    parser.add_argument("--output-file", type=str, default="processed_data.csv", help="Path to save the combined and processed CSV.")
    
    args = parser.parse_args()
    main(args)
