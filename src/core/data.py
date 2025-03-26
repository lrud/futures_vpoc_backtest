import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging
from typing import Optional, List, Dict, Union, Tuple

from src.utils.logging import get_logger

class FuturesDataManager:
    """
    Data loading and preprocessing for futures data.
    Provides methods to load, clean, and validate futures data.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the DataLoader with settings.
        
        Args:
            data_dir: Path to the directory containing data files
        """
        self.logger = get_logger(__name__)
        
        # Try to get data_dir from settings if not provided
        if data_dir is None:
            try:
                from src.config.settings import settings
                self.data_dir = settings.DATA_DIR
            except (ImportError, AttributeError):
                self.data_dir = os.path.join(os.getcwd(), 'DATA')
        else:
            self.data_dir = data_dir
            
        self.logger.info(f"Initialized with data directory: {self.data_dir}")
        
        # Get additional settings if available
        try:
            from src.config.settings import settings
            self.low_price_threshold = getattr(settings, 'LOW_PRICE_THRESHOLD', 1000)
        except (ImportError, AttributeError):
            self.low_price_threshold = 1000
    
    def load_futures_data(self, data_directory=None) -> Optional[pd.DataFrame]:
        """
        Loads, merges, and cleans futures data from text files in the specified directory.
        Includes logging for potentially erroneous low prices and filters them out.

        Args:
            data_directory: Path to the directory containing .txt data files.

        Returns:
            Combined and cleaned futures data, or None if no valid data is found.
        """
        data_dir = data_directory or self.data_dir
        
        self.logger.info(f"Loading futures data from: {data_dir}")
        print(f"\n===== LOADING FUTURES DATA =====")
        print(f"Looking for text files in: {data_dir}")

        # Use glob to find all .txt files in the directory
        file_paths = glob.glob(os.path.join(data_dir, '*.txt'))
        print(f"Found {len(file_paths)} text files.")

        if not file_paths:
            self.logger.warning(f"No text files found in {data_dir}")
            print(f"Warning: No text files found in {data_dir}. Please check the directory.")
            return None

        # Initialize a list to store dataframes from each file
        all_dfs = []

        # Process each file
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            print(f"\nProcessing file: {file_name}")

            try:
                df = pd.read_table(file_path, sep=';', header=None, 
                                  names=['date_time_str', 'open', 'high', 'low', 'close', 'volume'])
            except FileNotFoundError:
                self.logger.error(f"File not found at {file_path}")
                print(f"Error: File not found at {file_path}")
                continue
            except pd.errors.ParserError as e:
                self.logger.error(f"Error parsing file {file_name}: {e}")
                print(f"Error parsing file {file_name} with pandas: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error reading file {file_name}: {str(e)}")
                print(f"Unexpected error reading file {file_name}: {str(e)}")
                continue

            print(f"Successfully loaded file with {len(df)} rows.")
            self.logger.info(f"Loaded {file_name} with {len(df)} rows")

            try:
                # Convert timestamp and handle date formatting
                df['timestamp'] = pd.to_datetime(df['date_time_str'], format='%Y%m%d %H%M%S', errors='raise')
                df = df.drop('date_time_str', axis=1)
                df.set_index('timestamp', inplace=True)
                df['date'] = df.index.date
            except ValueError as e:
                self.logger.error(f"Error converting timestamp in {file_name}: {e}")
                print(f"Error converting timestamp in {file_name}: {e}")
                print("Skipping file due to timestamp conversion error.")
                continue

            # Add contract and session information
            df['contract'] = file_name.replace('.Last.txt', '')
            df['session'] = 'ETH'  # Default session
            
            # Vectorized session assignment (RTH is 9:30 AM to 4:00 PM)
            df.loc[(df.index.hour >= 9) & (df.index.hour < 16) & 
                   ~((df.index.hour == 9) & (df.index.minute < 30)), 'session'] = 'RTH'

            # --- Enhanced Logging and Filtering for Potentially Erroneous Low Prices ---
            low_price_threshold = self.low_price_threshold
            initial_len = len(df)
            
            # Filter out low prices using vectorized operations
            df_filtered = df[df['open'] >= low_price_threshold]
            df_filtered = df_filtered[df_filtered['low'] >= low_price_threshold]

            removed_count = initial_len - len(df_filtered)
            if removed_count > 0:
                self.logger.warning(
                    f"Removed {removed_count} rows with prices below {low_price_threshold} from {file_name}")
                print(f"Warning: Removed {removed_count} rows with open/low prices below {low_price_threshold} from {file_name}.")
                df = df_filtered

            print(f"Data shape for {file_name}: {df.shape}")

            # Remove duplicate timestamps with vectorized operation
            if df.index.duplicated().any():
                dupes = df.index.duplicated().sum()
                self.logger.warning(f"Found {dupes} duplicate timestamps in {file_name}")
                print(f"Warning: Found duplicate timestamps in {file_name}. Removing duplicates (keeping first).")
                df = df[~df.index.duplicated(keep='first')]

            # Check for missing values
            missing_values = df.isna().sum()
            if missing_values.sum() > 0:
                self.logger.warning(f"Missing values found in {file_name}: {missing_values}")
                print(f"Warning: Missing values found in {file_name}:\n{missing_values}")

            # Check for zero or negative prices
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                self.logger.warning(f"Zero or negative prices found in {file_name}")
                print(f"Warning: Found zero or negative prices in {file_name}")

            # Check for high < low inconsistencies
            if (df['high'] < df['low']).any():
                self.logger.warning(f"High < low inconsistencies found in {file_name}")
                print(f"Warning: Found high < low inconsistencies in {file_name}")

            # Check for OHLC relationship inconsistencies
            ohlc_issues = ((df['open'] > df['high']) | (df['open'] < df['low']) | 
                          (df['close'] > df['high']) | (df['close'] < df['low']))
                          
            if ohlc_issues.any():
                self.logger.warning(f"Found {ohlc_issues.sum()} OHLC relationship inconsistencies in {file_name}")
                print(f"Warning: Found {ohlc_issues.sum()} OHLC relationship inconsistencies in {file_name}")

            # Calculate additional metrics with vectorized operations
            df['bar_range'] = df['high'] - df['low']
            df['bar_return'] = df['close'].pct_change()

            all_dfs.append(df)
            print(f"Successfully processed {file_name}")
            self.logger.info(f"Successfully processed {file_name}")

        # Combine all dataframes and sort by timestamp
        if all_dfs:
            combined_df = pd.concat(all_dfs, axis=0)
            combined_df.sort_index(inplace=True)

            # Log dataset summary
            self._log_dataset_summary(combined_df)
            
            return combined_df
        else:
            self.logger.warning("No valid data was processed from any file")
            print("No valid data was processed from any file.")
            return None
    
    def _log_dataset_summary(self, df: pd.DataFrame) -> None:
        """
        Logs and prints a summary of the combined dataset.
        
        Args:
            df: Combined DataFrame to summarize
        """
        print("\n===== Combined Dataset Summary =====")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Number of contracts: {df['contract'].nunique()}")
        print(f"Contracts: {', '.join(df['contract'].unique())}")
        print(f"RTH sessions: {(df['session'] == 'RTH').sum()}")
        print(f"ETH sessions: {(df['session'] == 'ETH').sum()}")

        print("\nPrice statistics:")
        print(df[['open', 'high', 'low', 'close']].describe())

        print("\nVolume statistics:")
        print(df['volume'].describe())
        
        self.logger.info(f"Loaded combined dataset with {len(df)} records")
        self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        self.logger.info(f"Contracts: {df['contract'].nunique()} unique contracts")
    
    def save_cleaned_data(self, df: pd.DataFrame, output_directory=None, 
                         file_name=None) -> str:
        """
        Save cleaned data to a CSV file.
        
        Args:
            df: DataFrame to save
            output_directory: Directory to save the file in
            file_name: Name of the output file
            
        Returns:
            Path to the saved file
        """
        # Try to get output directory from settings if not provided
        if output_directory is None:
            try:
                from src.config.settings import settings
                output_dir = settings.CLEANED_DATA_DIR
            except (ImportError, AttributeError):
                output_dir = os.path.join(os.getcwd(), 'DATA', 'CLEANED')
        else:
            output_dir = output_directory
            
        os.makedirs(output_dir, exist_ok=True)
        
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"cleaned_futures_data_{timestamp}.csv"
            
        output_path = os.path.join(output_dir, file_name)
        
        try:
            df.to_csv(output_path)
            self.logger.info(f"Saved cleaned data to {output_path}")
            print(f"Saved cleaned data to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving cleaned data: {str(e)}")
            print(f"Error saving cleaned data: {str(e)}")
            return ""
    
    def filter_by_contract(self, df: pd.DataFrame, contract_prefix: str) -> pd.DataFrame:
        """
        Filter data for a specific futures contract.
        
        Args:
            df: DataFrame to filter
            contract_prefix: Contract prefix to filter (e.g., 'ES')
            
        Returns:
            DataFrame with data for the specified contract
        """
        if df is None or df.empty:
            self.logger.warning("Cannot filter empty DataFrame by contract")
            return pd.DataFrame()
            
        # Filter for the specified contract
        contract_df = df[df['contract'].str.startswith(contract_prefix)]
        
        if len(contract_df) == 0:
            self.logger.warning(f"No data found for contract prefix '{contract_prefix}'")
            print(f"Warning: No data found for contract prefix '{contract_prefix}'")
            return pd.DataFrame()
            
        self.logger.info(f"Filtered to {len(contract_df)} records for contract '{contract_prefix}'")
        print(f"Filtered to {len(contract_df)} records for contract '{contract_prefix}'")
        
        return contract_df
    
    def filter_by_session(self, df: pd.DataFrame, session_type: str = 'RTH') -> pd.DataFrame:
        """
        Filter data by session type.
        
        Args:
            df: DataFrame to filter
            session_type: Session type to filter for ('RTH' or 'ETH')
            
        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty:
            self.logger.warning("Cannot filter empty DataFrame by session")
            return pd.DataFrame()
            
        if 'session' not in df.columns:
            self.logger.warning("DataFrame does not contain 'session' column. Cannot filter by session.")
            print("Warning: DataFrame does not contain 'session' column. Cannot filter by session.")
            return df
            
        filtered_df = df[df['session'] == session_type]
        
        self.logger.info(f"Filtered to {session_type} session: {len(filtered_df)} records")
        print(f"Filtered to {session_type} session: {len(filtered_df)} records")
        
        return filtered_df
    
    def filter_by_date_range(self, df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            df: DataFrame to filter
            start_date: Start date for filtering (inclusive)
            end_date: End date for filtering (inclusive)
            
        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty:
            self.logger.warning("Cannot filter empty DataFrame by date range")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # Convert date strings to datetime if necessary
        if start_date and isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if end_date and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Apply filters
        if start_date:
            filtered_df = filtered_df[filtered_df['date'] >= start_date]
            self.logger.info(f"Filtered to dates >= {start_date}: {len(filtered_df)} records")
        
        if end_date:
            filtered_df = filtered_df[filtered_df['date'] <= end_date]
            self.logger.info(f"Filtered to dates <= {end_date}: {len(filtered_df)} records")
        
        return filtered_df