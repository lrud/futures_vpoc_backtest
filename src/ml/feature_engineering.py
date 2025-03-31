# src/ml/feature_engineering.py
"""
Feature engineering functionality for futures trading models.
Handles creating, transforming, and selecting features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

from src.utils.logging import get_logger
from src.core.vpoc import calculate_volume_profile, find_vpoc, find_value_area
from src.core.data import FuturesDataManager

class FeatureEngineer:
    """
    Handles data loading, feature creation, transformation, and selection for ML models.
    """
    
    def __init__(self, lookback_periods: List[int] = None, device_ids: List[int] = None):
        """
        Initialize feature engineering module.
        
        Parameters:
        -----------
        lookback_periods: List[int]
            Time periods for lookback features
        device_ids: List[int]
            List of GPU device IDs to use for parallel processing
        """
        self.logger = get_logger(__name__)
        self.data_manager = FuturesDataManager()
        
        # Use default lookback periods from settings if not provided
        if lookback_periods is None:
            from src.config.settings import settings
            self.lookback_periods = settings.DEFAULT_LOOKBACK_PERIODS
        else:
            self.lookback_periods = lookback_periods
            
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_and_prepare_data(self, data_path: str, contract_filter: Optional[str] = None, device_ids: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Complete data loading and preparation pipeline.
        
        Parameters:
        -----------
        data_path: str
            Path to data directory or file
        contract_filter: Optional[str]
            Contract type to filter (ES, VIX, or None for all)
        device_ids: List[int]
            List of GPU device IDs to use for parallel processing
            
        Returns:
        --------
        Tuple containing:
        - X_train: Training features
        - y_train: Training targets  
        - X_val: Validation features
        - y_val: Validation targets
        - feature_columns: List of feature names
        """
        self.logger.info(f"Loading data from {data_path}")
        
        # Load raw data (distribute across GPUs if specified)
        if device_ids and len(device_ids) > 1:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            # Load data normally first
            if os.path.isfile(data_path):
                data = self.data_manager.load_futures_data_file(data_path)
            else:
                data = self.data_manager.load_futures_data(data_path)
            
            # Convert to tensors and distribute
            features_df = self.prepare_features(data)
            X = torch.tensor(features_df[self.feature_columns].values, dtype=torch.float32)
            y = torch.tensor(features_df['target'].values, dtype=torch.float32)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=len(X)//len(device_ids), 
                                  shuffle=False, num_workers=len(device_ids))
            
            # Distribute batches to GPUs
            X_batches = []
            y_batches = []
            for i, (X_batch, y_batch) in enumerate(dataloader):
                device = f'cuda:{device_ids[i % len(device_ids)]}'
                X_batches.append(X_batch.to(device))
                y_batches.append(y_batch.to(device))
            
            # Split into train/val
            split_idx = int(len(X_batches) * 0.8)
            return (X_batches[:split_idx], y_batches[:split_idx], 
                   X_batches[split_idx:], y_batches[split_idx:], 
                   self.feature_columns)
        else:
            # Single GPU/CPU path
            if os.path.isfile(data_path):
                data = self.data_manager.load_futures_data_file(data_path)
            else:
                data = self.data_manager.load_futures_data(data_path)
        
        if data is None or len(data) == 0:
            raise ValueError("Failed to load data or empty dataset")
            
        # Handle contract filtering
        if contract_filter:
            self.logger.info(f"Filtering for contract: {contract_filter}")
            matching_data = data[
                (data['contract'].str.startswith(contract_filter)) | 
                (data.get('contract_root', '') == contract_filter)
            ]
            
            if len(matching_data) == 0:
                self.logger.warning(f"No data found with contract {contract_filter}, using all data")
                data['contract'] = contract_filter
                data['contract_root'] = contract_filter
            else:
                data = matching_data
                
        # Generate features
        features_df = self.prepare_features(data)
        if len(features_df) == 0:
            raise ValueError("Feature generation failed")
            
        # Create target variable
        if 'vpoc' in features_df.columns:
            features_df['target'] = np.sign(features_df['vpoc'].shift(-1) - features_df['vpoc'])
        else:
            features_df['target'] = np.sign(features_df['close'].shift(-1) - features_df['close'])
            
        features_df = features_df.dropna(subset=['target'])
        
        # Split into features and target
        X = features_df[self.feature_columns].values
        y = features_df['target'].values
        
        # Train/validation split (80/20)
        split_idx = int(len(X) * 0.8)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:], self.feature_columns

    def generate_session_features(self, session_data: pd.DataFrame) -> Dict:
        """
        Generate features for a single trading session.
        
        Parameters:
        -----------
        session_data: pd.DataFrame
            DataFrame containing OHLCV data for a single session
            
        Returns:
        --------
        Dict
            Dictionary with session features
        """
        try:
            # Handle cases where session_data has an incorrect index
            if not hasattr(session_data, 'index') or len(session_data) == 0:
                self.logger.error("Empty or invalid session data")
                return None
            
            # Extract key data points
            session_high = session_data['high'].max()
            session_low = session_data['low'].min()
            session_open = session_data['open'].iloc[0]
            session_close = session_data['close'].iloc[-1]
            
            # Handle volume (which might be missing for VIX data)
            if 'volume' in session_data.columns:
                session_volume = session_data['volume'].sum()
            else:
                session_volume = 0
                self.logger.warning("No volume data available, using 0")
            
            # Get date - try various methods to ensure we get a valid date
            if isinstance(session_data.index, pd.DatetimeIndex):
                date = session_data.index[0].date()
            elif 'date' in session_data.columns:
                if pd.api.types.is_datetime64_any_dtype(session_data['date']):
                    date = session_data['date'].iloc[0].date() 
                else:
                    date = pd.to_datetime(session_data['date'].iloc[0]).date()
            else:
                self.logger.error("Could not determine date for session")
                return None
            
            # Calculate price change percentage
            close_change_pct = (session_close - session_open) / max(session_open, 0.0001) * 100
            
            # Calculate range metrics
            price_range = max(session_high - session_low, 0.0001)
            range_pct = price_range / max(session_open, 0.0001) * 100
            
            # Handle VIX data differently - skip volume profile generation
            if session_volume == 0 and 'VIX' in str(session_data.get('contract', '')):
                # For VIX data, just use price-based features without volume profiles
                vpoc = session_close  # Use close as VPOC proxy
                val = session_low
                vah = session_high
                va_volume_pct = 100  # Entire range is value area
                
            else:
                # Create volume profile using imported functions
                volume_profile = calculate_volume_profile(session_data, price_precision=0.25)
                vpoc = find_vpoc(volume_profile)
                val, vah, va_volume_pct = find_value_area(volume_profile)
            
            # Calculate VWAP or use close price if no volume
            if session_volume > 0:
                vwap = (session_data['close'] * session_data['volume']).sum() / session_volume
            else:
                vwap = session_close  # Use close price as VWAP proxy
            
            # Return session features dictionary
            return {
                'date': date,
                'vpoc': float(vpoc),
                'total_volume': float(session_volume),
                'value_area_low': float(val),
                'value_area_high': float(vah),
                'value_area_width': float(vah - val),
                'price_range': float(price_range),
                'range_pct': float(range_pct),
                'close_change_pct': float(close_change_pct),
                'session_high': float(session_high),
                'session_low': float(session_low),
                'session_open': float(session_open),
                'session_close': float(session_close),
                'vwap': float(vwap),
                'close_to_vwap_pct': float((session_close - vwap) / max(vwap, 0.0001) * 100)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating session features: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def generate_timeseries_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time series features based on lookback periods.
        
        Parameters:
        -----------
        features_df: pd.DataFrame
            DataFrame with base session features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added time series features
        """
        # Ensure data is sorted by date
        features_df = features_df.sort_values('date')
        
        # Add features for each lookback period
        for period in self.lookback_periods:
            # Skip if not enough data
            if len(features_df) <= period:
                self.logger.warning(f"Not enough data for lookback period {period}")
                continue
                
            # Price momentum
            features_df[f'price_mom_{period}d'] = features_df['session_close'].pct_change(period) * 100
            
            # Volatility
            features_df[f'volatility_{period}d'] = features_df['close_change_pct'].rolling(period).std()
            
            # Volume trend
            features_df[f'volume_trend_{period}d'] = features_df['total_volume'].pct_change(period) * 100
            
            # VPOC migration
            features_df[f'vpoc_change_{period}d'] = features_df['vpoc'].diff(period)
            features_df[f'vpoc_pct_change_{period}d'] = features_df['vpoc'].pct_change(period) * 100
            
            # Range evolution
            features_df[f'range_change_{period}d'] = features_df['range_pct'].pct_change(period) * 100
            
            # Value area features
            features_df[f'va_width_{period}d_ma'] = (
                features_df['value_area_high'] - features_df['value_area_low']
            ).rolling(period).mean()
            
            # VA expansion/contraction
            current_width = features_df['value_area_high'] - features_df['value_area_low']
            features_df[f'va_width_change_{period}d'] = (
                current_width / features_df[f'va_width_{period}d_ma'].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)
            
            # VPOC standard deviation
            features_df[f'vpoc_std_{period}d'] = features_df['vpoc'].rolling(period).std()
            
            # Normalized distance features
            features_df[f'vpoc_zscore_{period}d'] = (
                features_df['vpoc'] - features_df['vpoc'].rolling(period).mean()
            ) / features_df['vpoc'].rolling(period).std().replace(0, np.nan)
            
        # Clean up data
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Store feature columns for future reference (excluding the index name if it's 'date')
        self.feature_columns = [col for col in features_df.columns
                               if col != features_df.index.name and pd.api.types.is_numeric_dtype(features_df[col])]

        return features_df # Return DataFrame with DatetimeIndex and numeric columns
    
    def select_features(self, X: np.ndarray, y: np.ndarray, max_features: int = 15) -> Tuple[np.ndarray, List[int]]:
        """
        Select most important features using statistical tests.
        
        Parameters:
        -----------
        X: np.ndarray
            Feature matrix
        y: np.ndarray
            Target vector
        max_features: int
            Maximum number of features to select
            
        Returns:
        --------
        Tuple[np.ndarray, List[int]]
            Selected features and their indices
        """
        if X.shape[1] <= max_features:
            self.logger.info(f"Using all {X.shape[1]} features (below max_features threshold)")
            return X, list(range(X.shape[1]))
            
        try:
            # Select K best features
            selector = SelectKBest(f_regression, k=max_features)
            X_selected = selector.fit_transform(X, y)
            
            # Get feature indices
            selected_indices = selector.get_support(indices=True)
            
            # Log selected feature info
            if hasattr(self, 'feature_columns') and len(self.feature_columns) == X.shape[1]:
                selected_features = [self.feature_columns[i] for i in selected_indices]
                self.logger.info(f"Selected top {max_features} features: {selected_features}")
            
            return X_selected, selected_indices
            
        except Exception as e:
            self.logger.error(f"Feature selection error: {e}")
            return X, list(range(X.shape[1]))
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        X: np.ndarray
            Feature matrix
        fit: bool
            Whether to fit the scaler or just transform
            
        Returns:
        --------
        np.ndarray
            Scaled features
        """
        try:
            if fit:
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        except Exception as e:
            self.logger.error(f"Feature scaling error: {e}")
            return X

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Complete pipeline to prepare features from raw data.
        
        Parameters:
        -----------
        raw_data: pd.DataFrame
            Raw OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            Processed features DataFrame
        """
        # Group by date to get sessions
        features_list = []
        skipped_dates = []
        
        # Make sure date is present either as index or column
        if not isinstance(raw_data.index, pd.DatetimeIndex) and 'date' not in raw_data.columns:
            self.logger.error("Data has neither DatetimeIndex nor 'date' column")
            # Try to restore date column from index if it's missing
            if isinstance(raw_data.index, pd.Index) and raw_data.index.name == 'date':
                raw_data = raw_data.reset_index()
                self.logger.info("Restored date column from index")
                
        # Check if the data is already indexed by date, if not, try to use 'date' column
        if not isinstance(raw_data.index, pd.DatetimeIndex):
            if 'date' in raw_data.columns:
                # Try to convert date column to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(raw_data['date']):
                    try:
                        raw_data['date'] = pd.to_datetime(raw_data['date'])
                    except Exception as e:
                        self.logger.error(f"Could not convert 'date' column to datetime: {e}")
                        return pd.DataFrame()
                        
                # Group by date
                self.logger.info("Grouping by date column")
                # Use groupby with date component to handle different time values
                for date_obj, group in raw_data.groupby(raw_data['date'].dt.date):
                    try:
                        # Generate session features
                        session_features = self.generate_session_features(group)
                        if session_features:
                            features_list.append(session_features)
                        else:
                            self.logger.warning(f"No features generated for session {date_obj}")
                    except Exception as e:
                        self.logger.error(f"Error processing session {date_obj}: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        skipped_dates.append(date_obj)
            else:
                self.logger.error("Data has no DatetimeIndex and no 'date' column")
                return pd.DataFrame()
        else:
            # Original code path for DatetimeIndex
            self.logger.info("Grouping by DatetimeIndex")
            for date, group in raw_data.groupby(raw_data.index.date):
                try:
                    # Generate session features
                    session_features = self.generate_session_features(group)
                    if session_features:
                        features_list.append(session_features)
                    else:
                        self.logger.warning(f"No features generated for session {date}")
                except Exception as e:
                    self.logger.error(f"Error processing session {date}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    skipped_dates.append(date)
        
        if not features_list:
            self.logger.error("No valid features could be generated")
            return pd.DataFrame()
            
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # --- Convert 'date' to datetime and set as index ---
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df.set_index('date', inplace=True)
        self.logger.info("Set 'date' column as DatetimeIndex.")
        
        # Add time series features
        features_df = self.generate_timeseries_features(features_df)
        
        # Clean data
        original_count = len(features_df)
        features_df.dropna(inplace=True)
        dropped_count = original_count - len(features_df)
        
        # Validate we have features
        if len(features_df) == 0 or len(self.feature_columns) == 0:
            self.logger.error("No valid features generated - empty DataFrame or no feature columns")
            return pd.DataFrame()
            
        # Log feature matrix details
        self.logger.info(
            f"Generated features for {len(features_df)} sessions (dropped {dropped_count} with NaN values). "
            f"Feature matrix shape: ({len(features_df)}, {len(self.feature_columns)})"
        )
        
        # Ensure we only return columns that exist in the DataFrame
        valid_columns = [col for col in self.feature_columns if col in features_df.columns]
        if len(valid_columns) != len(self.feature_columns):
             self.logger.warning(
                 f"Feature mismatch: expected {len(self.feature_columns)} features, "
                 f"found {len(valid_columns)} in DataFrame. Using valid columns."
             )
             self.feature_columns = valid_columns # Update internal list

        if not valid_columns:
             self.logger.error(
                 "No valid numeric feature columns found after processing. Available columns: "
                 f"{features_df.columns.tolist()}. Index: {features_df.index.name}"
             )
             raise ValueError("No valid numeric feature columns generated - check data and feature generation logic")

        # Validate feature matrix has non-zero dimensions
        if len(features_df) == 0 or len(valid_columns) == 0:
             self.logger.error(
                 "Empty feature matrix generated - no sessions or features available. "
                 f"Sessions: {len(features_df)}, Features: {len(valid_columns)}"
             )
             raise ValueError("Empty feature matrix - check data and feature generation parameters")

        # Validate features are not all NaN/zero
        if not features_df.empty:
             feature_sample = features_df[valid_columns].iloc[0]
             if feature_sample.isnull().all() or (feature_sample == 0).all():
                 self.logger.error(
                     "All features are null or zero - check feature generation logic. "
                     f"Sample row: {feature_sample.to_dict()}"
                 )
                 # Don't raise error here, let it proceed, might be valid for some cases
                 # raise ValueError("Invalid features generated - all null or zero values")

        # Log final feature matrix details
        self.logger.info(
             f"Final feature matrix shape: ({len(features_df)}, {len(valid_columns)}) with DatetimeIndex."
        )
        self.logger.debug(
             f"Feature columns: {valid_columns}\n"
             # f"Sample row:\n{feature_sample.to_dict()}" # Avoid logging potentially large row
        )
        # Return only the valid numeric columns, index is preserved
        return features_df[valid_columns]
