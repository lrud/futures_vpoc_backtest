"""
Data handling functionality for futures markets.
Handles loading, cleaning, and merging different data sources.
"""

import os
import pandas as pd
import numpy as np
import glob
from typing import Optional, Dict, List, Union
import datetime as dt

from src.utils.logging import get_logger
from src.config.settings import settings

class FuturesDataManager:
    """
    Handles loading and preprocessing of futures market data.
    Supports both ES and VIX data formats.
    """
    
    def __init__(self):
        """Initialize data manager."""
        self.logger = get_logger(__name__)
        self.data_cache = {}
    
    def load_futures_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load futures data from file or directory.
        
        Parameters:
        -----------
        data_path: str, optional
            Path to data file or directory containing data files
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with futures data
        """
        data_path = data_path or settings.DATA_DIR
        self.logger.info(f"Loading futures data from {data_path}")
        
        # Handle file case
        if os.path.isfile(data_path):
            df = self.load_futures_data_file(data_path)
            if df is not None:
                # Ensure contract column exists (from filename if needed)
                if 'contract' not in df.columns:
                    contract = os.path.basename(data_path).split("_")[0]
                    df['contract'] = contract
                return df
            return None
            
        # Handle directory case
        if not os.path.isdir(data_path):
            self.logger.error(f"Data path not found: {data_path}")
            return None
        
        # Find all CSV files in directory
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        
        if not csv_files:
            self.logger.error(f"No CSV files found in {data_path}")
            return None
        
        # Load and combine all files
        dfs = []
        for file in csv_files:
            try:
                # Extract contract from filename
                contract = os.path.basename(file).split("_")[0]
                
                # Load data
                df = self.load_futures_data_file(file)
                
                if df is not None:
                    # Add contract column if not present
                    if 'contract' not in df.columns:
                        df['contract'] = contract
                    
                    dfs.append(df)
                    self.logger.info(f"Loaded {len(df)} rows from {os.path.basename(file)}")
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
                
        if not dfs:
            self.logger.error("Failed to load any data files")
            return None
            
        # Combine all data
        data = pd.concat(dfs, ignore_index=True)
        
        # Cache for reuse
        self.data_cache[data_path] = data
        
        self.logger.info(f"Combined data: {len(data)} rows, contracts: {data['contract'].unique()}")
        return data
    
    def load_futures_data_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single futures data file.
        
        Parameters:
        -----------
        file_path: str
            Path to CSV file
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with futures data
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return None
            
            # Use cached data if available
            if file_path in self.data_cache:
                return self.data_cache[file_path].copy()
            
            # Read CSV file
            data = pd.read_csv(file_path)
            
            # Clean and preprocess
            data = self._preprocess_data(data)
            
            # Cache for reuse
            self.data_cache[file_path] = data
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess futures data.
        
        Parameters:
        -----------
        data: pandas.DataFrame
            Raw data
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned data
        """
        # Make a copy to avoid modifying original
        df = data.copy()
        
        try:
            # Handle common column name formats for different data sources
            # Try to standardize to lowercase names
            column_mappings = {
                # Standard capitalized names
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
                # VIX specific column names
                'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume',
                # Date/time column variations
                'DATE': 'date', 'Date': 'date', 'Time': 'timestamp', 'TIMESTAMP': 'timestamp',
                # Additional columns for merged data
                'Bar_Range': 'bar_range', 'BarRange': 'bar_range', 'BAR_RANGE': 'bar_range',
                'Bar_Return': 'bar_return', 'BarReturn': 'bar_return', 'BAR_RETURN': 'bar_return',
                'VIX': 'vix', 'Volatility': 'vix', 'VOLATILITY': 'vix',
                'Contract': 'contract', 'CONTRACT': 'contract',
                'Session': 'session', 'SESSION': 'session'
            }
            
            # Apply column renaming where columns exist
            rename_dict = {old: new for old, new in column_mappings.items() if old in df.columns}
            if rename_dict:
                df = df.rename(columns=rename_dict)
                self.logger.info(f"Renamed columns: {rename_dict}")
            
            # Convert timestamp/date column if needed
            if 'timestamp' in df.columns:
                time_col = 'timestamp'
            elif 'date' in df.columns:
                time_col = 'date'
            else:
                self.logger.error("No timestamp or date column found")
                return None
                
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
                self.logger.info(f"Converted {time_col} column to datetime")
            
            # Set timestamp as index
            df = df.set_index(time_col)
            self.logger.info(f"Set {time_col} column as index")
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close']
            optional_cols = ['bar_range', 'bar_return', 'vix', 'contract', 'session']
            
            # Check required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing required columns: {missing_cols}")
                # Try to find columns that might contain the required data
                for req_col in missing_cols:
                    potential_matches = [col for col in df.columns if req_col.lower() in col.lower()]
                    if potential_matches:
                        df[req_col] = df[potential_matches[0]]
                        self.logger.info(f"Using {potential_matches[0]} as {req_col}")
                    else:
                        self.logger.error(f"Required column {req_col} missing and no alternative found")
                        return None
            
            # Check if volume is missing - this is optional for VIX data
            if 'volume' not in df.columns:
                self.logger.warning(f"Missing volume column, adding placeholder with 0 values")
                df['volume'] = 0  # Add a placeholder volume column with zeros
                
            # Identify contract type if not present
            if 'contract' not in df.columns:
                # Try to determine from filename in index name
                df['contract'] = 'UNKNOWN'
            
            # Add session column if not present
            if 'session' not in df.columns:
                # Default to RTH (Regular Trading Hours)
                df['session'] = 'RTH'
            
            # Sort by date
            df = df.sort_index()
            
            # Handle duplicate timestamps
            if df.index.duplicated().any():
                self.logger.warning(f"Found duplicate timestamps, keeping last value")
                df = df[~df.index.duplicated(keep='last')]
            
            # Handle missing values
            if df.isna().any().any():
                self.logger.warning(f"Found missing values, filling with appropriate methods")
                # Fill missing OHLC with forward fill, volume with 0
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
                df['volume'] = df['volume'].fillna(0)
                
            # Validate data types
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Drop any rows that still have NaN after conversion
            if df.isna().any().any():
                before_count = len(df)
                df = df.dropna(subset=['open', 'high', 'low', 'close'])
                after_count = len(df)
                self.logger.warning(f"Dropped {before_count - after_count} rows with NaN values")
                
            # Calculate bar_range if missing but OHLC present
            if 'bar_range' not in df.columns and all(col in df.columns for col in ['high', 'low']):
                df['bar_range'] = df['high'] - df['low']
                self.logger.info("Calculated bar_range from high-low")
            
            # Calculate bar_return if missing but open/close present
            if 'bar_return' not in df.columns and all(col in df.columns for col in ['open', 'close']):
                df['bar_return'] = (df['close'] - df['open']) / df['open']
                self.logger.info("Calculated bar_return from (close-open)/open")
            
            # Initialize VIX column if missing
            if 'vix' not in df.columns:
                df['vix'] = np.nan
                self.logger.info("Initialized missing VIX column with NaN")
            
            # Add date column back for convenience in downstream processing
            df['date'] = df.index.date.astype(str) if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index).date.astype(str)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return data  # Return original if preprocessing fails
    
    def split_by_contract(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data by contract type.
        
        Parameters:
        -----------
        data: pandas.DataFrame
            Combined data
            
        Returns:
        --------
        Dict[str, pandas.DataFrame]
            Dictionary with data for each contract
        """
        if 'contract' not in data.columns:
            self.logger.error("No contract column in data")
            return {'UNKNOWN': data}
            
        result = {}
        
        for contract, group in data.groupby('contract'):
            result[contract] = group.copy()
            self.logger.info(f"Split {contract} data: {len(group)} rows")
            
        return result
