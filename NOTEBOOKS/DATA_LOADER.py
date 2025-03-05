import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
import logging  # Import the logging module

# Configure logging - set up basic logging to console for now
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(filename)s:%(lineno)d - %(message)s') # More info in log


def load_futures_data(data_directory):
    """
    Loads, merges, and cleans futures data from text files in the specified directory.
    Includes logging for potentially erroneous low prices and filters them out.

    Args:
        data_directory (str): Path to the directory containing .txt data files.

    Returns:
        pandas.DataFrame: Combined and cleaned futures data, or None if no valid data is found.
    """
    data_dir = data_directory  # Use the function argument

    print("\n===== LOADING FUTURES DATA =====")
    print(f"Looking for text files in: {data_dir}")

    # Use glob to find all .txt files in the directory
    file_paths = glob.glob(os.path.join(data_dir, '*.txt'))
    print(f"Found {len(file_paths)} text files.")

    if not file_paths:
        print(f"Warning: No text files found in {data_dir}. Please check the directory.")
        return None

    # Initialize a list to store dataframes from each file
    all_dfs = []

    # Process each file
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing file: {file_name}")

        try:
            df = pd.read_table(file_path, sep=';', header=None, names=['date_time_str', 'open', 'high', 'low', 'close', 'volume'])
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            continue
        except pd.errors.ParserError as e:
            print(f"Error parsing file {file_name} with pandas: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error reading file {file_name}: {str(e)}")
            continue

        print(f"Successfully loaded file with {len(df)} rows.")

        try:
            df['timestamp'] = pd.to_datetime(df['date_time_str'], format='%Y%m%d %H%M%S', errors='raise') # errors='raise' to halt on bad dates
            df = df.drop('date_time_str', axis=1)
            df.set_index('timestamp', inplace=True)
            df['date'] = df.index.date  # Create 'date' column from the Timestamp index
        except ValueError as e:
            print(f"Error converting timestamp in {file_name}: {e}")
            print("Skipping file due to timestamp conversion error.")
            continue

        df['contract'] = file_name.replace('.Last.txt', '')  # Store contract name
        df['session'] = 'ETH'  # Default session
        df.loc[(df.index.hour >= 9) & (df.index.hour < 16) & ~((df.index.hour == 9) & (df.index.minute < 30)), 'session'] = 'RTH' #Vectorized session assignment

        # --- Enhanced Logging and Filtering for Potentially Erroneous Low Prices ---
        low_price_threshold = 1000  # Threshold for filtering - adjust as needed
        initial_len = len(df) # Store initial length for comparison

        df_filtered = df[df['open'] >= low_price_threshold] # Filter out low open prices
        df_filtered = df_filtered[df_filtered['low'] >= low_price_threshold] # Filter out low low prices

        removed_count = initial_len - len(df_filtered) # Calculate removed rows
        if removed_count > 0:
            print(f"Warning: Removed {removed_count} rows with open/low prices below {low_price_threshold} from {file_name}.")
            logging.warning(f"Removed {removed_count} rows with open/low prices below {low_price_threshold} from {file_name}. Total removed: {removed_count}") # Log removal count
            df = df_filtered # Replace df with filtered version


        # --- End of Enhanced Logging and Filtering ---


        print(f"Data shape for {file_name}: {df.shape}")

        if df.index.duplicated().any():
            print(f"Warning: Found duplicate timestamps in {file_name}. Removing duplicates (keeping first).")
            df = df[~df.index.duplicated(keep='first')] #Vectorized duplicate removal

        missing_values = df.isna().sum()
        if missing_values.sum() > 0:
            print(f"Warning: Missing values found in {file_name}:\n{missing_values}")

        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            print("Warning: Found zero or negative prices in {file_name}")

        if (df['high'] < df['low']).any():
            print("Warning: Found high < low inconsistencies in {file_name}")

        ohlc_issues = ((df['open'] > df['high']) | (df['open'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low']))
        if ohlc_issues.any():
            print(f"Warning: Found {ohlc_issues.sum()} OHLC relationship inconsistencies in {file_name}")

        df['bar_range'] = df['high'] - df['low'] #Vectorized calculation
        df['bar_return'] = df['close'].pct_change() #Vectorized calculation

        all_dfs.append(df)
        print(f"Successfully processed {file_name}")

    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=0)
        combined_df.sort_index(inplace=True) # Sort combined dataframe

        print("\n===== Combined Dataset Summary =====")
        print(f"Total records: {len(combined_df)}")
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Number of contracts: {combined_df['contract'].nunique()}")
        print(f"Contracts: {', '.join(combined_df['contract'].unique())}")
        print(f"RTH sessions: {(combined_df['session'] == 'RTH').sum()}")
        print(f"ETH sessions: {(combined_df['session'] == 'ETH').sum()}")

        print("\nPrice statistics:")
        print(combined_df[['open', 'high', 'low', 'close']].describe())

        print("\nVolume statistics:")
        print(combined_df['volume'].describe())

        return combined_df
    else:
        print("No valid data was processed from any file.")
        return None


if __name__ == "__main__":
    # ========================================================================
    # =====  IMPORTANT: UPDATE THIS PATH TO YOUR ACTUAL DATA DIRECTORY!  =====
    # ========================================================================
    test_data_dir = '/home/lr/Documents/FUTURUES_PROJECT/futures_vpoc_backtest/DATA/'
    # ========================================================================
    # =====  IMPORTANT: UPDATE THIS PATH TO YOUR ACTUAL DATA DIRECTORY!  =====
    # ========================================================================

    if not os.path.exists(test_data_dir):
        print(f"Error: Test data directory '{test_data_dir}' not found. Please create it or adjust the path in data_loader.py.")
    else:
        futures_data = load_futures_data(test_data_dir)
        if futures_data is not None:
            print("\nSuccessfully loaded futures data for testing.")
            print(futures_data.head()) # Print the first few rows to verify
        else:
            print("\nFailed to load futures data during testing.")