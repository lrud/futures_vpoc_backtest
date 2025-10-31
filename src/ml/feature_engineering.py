# src/ml/feature_engineering.py
"""
Feature engineering functionality for futures trading models.
Handles creating, transforming, and selecting features.
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging import get_logger
from src.core.vpoc import calculate_volume_profile, find_vpoc, find_value_area
from src.core.data import FuturesDataManager

class FeatureEngineer:
    """
    Handles data loading, feature creation, transformation, and selection for ML models.
    """
    
    def __init__(self, lookback_periods: List[int] = None, device_ids: List[int] = None, chunk_size: int = 3500):
        """
        Initialize feature engineering module.

        Parameters:
        -----------
        lookback_periods: List[int]
            Time periods for lookback features
        device_ids: List[int]
            List of GPU device IDs to use for parallel processing
        chunk_size: int
            VPOC chunk size for processing large sessions (default: 3500)
        """
        self.logger = get_logger(__name__)
        self.data_manager = FuturesDataManager()

        # Use default lookback periods from settings if not provided
        if lookback_periods is None:
            from src.config.settings import settings
            self.lookback_periods = settings.DEFAULT_LOOKBACK_PERIODS
        else:
            self.lookback_periods = lookback_periods

        # Store device IDs for multi-GPU VPOC processing
        self.device_ids = device_ids
        self.chunk_size = chunk_size

        self.scaler = StandardScaler()
        self.feature_columns = []

    def calculate_garch_features(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate GARCH-style volatility features for robust modeling.

        Parameters:
        -----------
        returns : pd.Series
            Return series

        Returns:
        --------
        Dict[str, float]
            GARCH-style volatility features
        """
        try:
            # Remove NaN values from returns
            returns_clean = returns.dropna()
            if len(returns_clean) < 10:
                self.logger.warning("Not enough data points for GARCH calculation")
                return {}

            squared_returns = returns_clean**2

            # Initial parameter estimates using method of moments
            omega = max(1e-6, np.var(squared_returns) * 0.01)  # Base variance

            # Calculate ARCH effect safely
            if len(squared_returns) > 1:
                arch_corr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0,1]
                if not np.isnan(arch_corr):
                    alpha = max(0.01, min(0.3, arch_corr**2))
                else:
                    alpha = 0.1  # Default if correlation is NaN
            else:
                alpha = 0.1

            beta = max(0.7, min(0.95, 1 - alpha))  # Persistence

            # Calculate conditional variance using proper GARCH(1,1) recursion
            conditional_var = []
            var = omega / (1 - beta)  # Unconditional variance

            for i in range(len(squared_returns)):
                if i == 0:
                    var = omega / (1 - beta)
                else:
                    var = omega + alpha * squared_returns.iloc[i-1] + beta * var
                conditional_var.append(var)

            # Create series with matching index
            conditional_var_series = pd.Series(conditional_var, index=returns_clean.index)

            # Calculate additional GARCH statistics
            long_run_var = omega / (1 - beta)
            half_life = np.log(0.5) / np.log(beta) if beta > 0 and beta < 1 else 100

            return {
                'garch_omega': omega,
                'garch_alpha': alpha,
                'garch_beta': beta,
                'garch_vol_persistence': beta,
                'garch_current_vol': np.sqrt(conditional_var_series.iloc[-1]) if len(conditional_var_series) > 0 else np.std(returns_clean),
                'garch_vol_forecast': np.sqrt(conditional_var_series.mean() if len(conditional_var_series) > 0 else np.std(returns_clean)),
                'garch_long_run_vol': np.sqrt(long_run_var),
                'garch_half_life': half_life,
                'arch_effect': alpha,
                'garch_num_obs': len(returns_clean)
            }

        except Exception as e:
            self.logger.warning(f"GARCH calculation failed: {e}")
            return {}

    def apply_robust_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply robust statistical transformations to mitigate overfitting.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data

        Returns:
        --------
        pd.DataFrame
            Data with robust transformations applied
        """
        df = data.copy()

        try:
            # 1. Log transformation for returns (handles heteroskedasticity)
            close_col = 'Close' if 'Close' in df.columns else 'session_close' if 'session_close' in df.columns else None
            if close_col:
                df['log_return'] = np.log(1 + df[close_col].pct_change())
                df['log_return_abs'] = np.abs(df['log_return'])
                self.logger.info(f"Applied log transformation to returns (using {close_col} column)")

            # 2. Winsorization (handles outliers)
            if 'returns' in df.columns:
                lower_1 = df['returns'].quantile(0.01)
                upper_99 = df['returns'].quantile(0.99)
                df['returns_winsorized'] = df['returns'].clip(lower_1, upper_99)

                lower_5 = df['returns'].quantile(0.05)
                upper_95 = df['returns'].quantile(0.95)
                df['returns_winsorized_95'] = df['returns'].clip(lower_5, upper_95)

                self.logger.info("Applied winsorization to returns")

            # 3. Robust scaling using median and MAD
            if 'log_return' in df.columns:
                median_val = df['log_return'].median()
                mad_val = (df['log_return'] - median_val).abs().median()
                df['log_return_robust_scaled'] = (df['log_return'] - median_val) / (mad_val + 1e-8)

                # Z-score robust scaling
                rolling_median = df['log_return'].rolling(20).median()
                rolling_mad = (df['log_return'] - rolling_median).abs().rolling(20).median()
                df['log_return_rolling_zscore'] = (df['log_return'] - rolling_median) / (rolling_mad + 1e-8)

                self.logger.info("Applied robust scaling methods")

        except Exception as e:
            self.logger.error(f"Robust transformations failed: {e}")

        return df

    def create_robust_target(self, data: pd.DataFrame, target_type: str = 'log') -> pd.DataFrame:
        """
        Create robust target variable for ML training.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_type : str
            Type of target transformation ('log', 'raw', 'winsorized')

        Returns:
        --------
        pd.DataFrame
            Data with robust target variable
        """
        df = data.copy()
        self.logger.info(f"🎯 Starting target creation with columns: {list(df.columns)[:10]}...")

        try:
            # Look for close price column in order of preference
            close_col = None
            if 'session_close' in df.columns:
                close_col = 'session_close'
                self.logger.info(f"✅ Found 'session_close' column for target creation")
            elif 'Close' in df.columns:
                close_col = 'Close'
                self.logger.info(f"✅ Found 'Close' column for target creation")
            elif 'close' in df.columns:
                close_col = 'close'
                self.logger.info(f"✅ Found 'close' column for target creation")

            if close_col:
                self.logger.info(f"📈 Using '{close_col}' column for target creation")
                returns = df[close_col].pct_change()
                self.logger.info(f"📊 Returns calculation completed, len={len(returns)}, non-null={returns.notna().sum()}")

                if target_type == 'log':
                    # Log transformation to handle heteroskedasticity and make distribution more normal
                    df['target'] = np.log(1 + returns.shift(-1))
                    # CRITICAL FIX: Clip extreme target values to prevent gradient explosion
                    df['target'] = df['target'].clip(-0.1, 0.1)  # Clip to ±10% returns
                    df['target_transformed'] = 'log_clipped'
                    self.logger.info("✅ Created log-transformed target variable with clipping")
                elif target_type == 'winsorized':
                    # Winsorized target to reduce outlier impact
                    lower_1 = returns.quantile(0.01)
                    upper_99 = returns.quantile(0.99)
                    df['target'] = returns.shift(-1).clip(lower_1, upper_99)
                    df['target_transformed'] = 'winsorized_1-99'
                    self.logger.info("✅ Created winsorized target variable")
                else:
                    # Raw returns
                    df['target'] = returns.shift(-1)
                    df['target_transformed'] = 'raw'
                    self.logger.info("✅ Created raw returns target variable")

                # Verify target was created
                if 'target' in df.columns:
                    target_count = df['target'].notna().sum()
                    self.logger.info(f"🎯 SUCCESS: Target variable created with {target_count} non-null values")
                    return df
                else:
                    self.logger.error("❌ FAILED: Target column not created after calculation")

            else:
                self.logger.error(f"❌ No close price column found for target creation. Available columns: {list(df.columns)}")
                # Create a dummy target to prevent complete failure
                df['target'] = 0.0
                df['target_transformed'] = 'dummy'
                self.logger.warning("⚠️ Created dummy target variable to prevent failure")
                return df

        except Exception as e:
            self.logger.error(f"❌ Target creation failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            # Create a dummy target to prevent complete failure
            df['target'] = 0.0
            df['target_transformed'] = 'dummy'
            self.logger.warning("⚠️ Created dummy target variable to prevent failure")

        return df

    def load_and_prepare_data(self, data_path: str, contract_filter: Optional[str] = None, device_ids: List[int] = None, data_fraction: float = 1.0, chunk_size: int = 3500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
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
        data_fraction: float
            Fraction of data to use (0.1 = 10%, 1.0 = 100%)
            
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

        # Load raw data
        if os.path.isfile(data_path):
            data = self.data_manager.load_futures_data_file(data_path)
        else:
            data = self.data_manager.load_futures_data(data_path)

        if data is None or len(data) == 0:
            raise ValueError("Failed to load data or empty dataset")

        # Apply data sampling if specified
        if data_fraction < 1.0:
            self.logger.info(f"Using {data_fraction*100:.1f}% of data for training")
            sample_size = int(len(data) * data_fraction)
            # Use systematic sampling to maintain temporal order
            step = len(data) // sample_size
            data = data.iloc[::step].copy()
            self.logger.info(f"Sampled {len(data)} rows from original dataset")

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

        # Handle multi-GPU processing
        if device_ids and len(device_ids) > 1:
            from torch.utils.data import DataLoader, TensorDataset

            # Convert to tensors and distribute
            features_df = self.prepare_features(data)

            # Create target variable using the enhanced robust target creation method
            features_df = self.create_robust_target(features_df, target_type='log')

            # CRITICAL FIX: Apply robust feature scaling before returning
            X = features_df[self.feature_columns].values
            y = features_df['target'].values

            self.logger.info("🔧 Applying robust feature scaling to prevent gradient explosion...")
            X_scaled = self.scale_features(X, fit=True)  # Apply our robust scaling method

            # Split into train/val
            split_idx = int(len(X_scaled) * 0.8)
            return (X_scaled[:split_idx], y[:split_idx],
                   X_scaled[split_idx:], y[split_idx:],
                   self.feature_columns)
        else:
            # Single GPU/CPU path - data loading already done above
            pass
                
        # Generate features
        features_df = self.prepare_features(data)
        if len(features_df) == 0:
            raise ValueError("Feature generation failed")
            
        # Apply robust transformations including log transforms and GARCH modeling
        self.logger.info("🔧 Applying robust statistical transformations...")
        features_df = self.apply_robust_transformations(features_df)

        # Create enhanced target variable using log-transformed returns
        self.logger.info("🎯 Creating enhanced target variable...")
        features_df = self.create_robust_target(features_df, target_type='log')

        # Add GARCH volatility features if not already present
        if not any(col.startswith('garch_') for col in features_df.columns):
            self.logger.info("📈 Adding GARCH volatility features...")
            if 'log_return' in features_df.columns:
                garch_features = self.calculate_garch_features(features_df['log_return'].dropna())
                # Add GARCH features as columns to the DataFrame
                for key, value in garch_features.items():
                    features_df[f'garch_{key}'] = value
                self.logger.info("Added GARCH volatility features")
            else:
                self.logger.warning("No 'log_return' column found for GARCH calculation")

        features_df = features_df.dropna(subset=['target'])
        
        # Split into features and target
        X = features_df[self.feature_columns].values
        y = features_df['target'].values

        # CRITICAL FIX: Apply robust feature scaling before returning
        self.logger.info("🔧 Applying robust feature scaling to prevent gradient explosion...")
        X_scaled = self.scale_features(X, fit=True)  # Apply our robust scaling method

        # Train/validation split (80/20)
        split_idx = int(len(X_scaled) * 0.8)
        return X_scaled[:split_idx], y[:split_idx], X_scaled[split_idx:], y[split_idx:], self.feature_columns

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
                # Create volume profile using imported functions with multi-GPU support
                volume_profile = calculate_volume_profile(session_data, price_precision=0.25, device_ids=self.device_ids, chunk_size=self.chunk_size)
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
        Scale features using robust method with outlier handling.

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
            # CRITICAL FIX: Apply robust scaling with outlier handling
            if fit:
                # Check for extreme outliers before scaling
                X_df = pd.DataFrame(X)
                # CRITICAL FIX: Use numpy instead of deprecated pandas mad() method
                median_vals = X_df.median()
                mad_vals = (X_df - median_vals).abs().median()
                outlier_mask = np.abs(X_df - median_vals) > 5 * mad_vals
                outlier_count = outlier_mask.sum().sum()

                if outlier_count > 0:
                    self.logger.warning(f"⚠️  Detected {outlier_count} extreme outliers (>5 MAD) - applying robust scaling")

                # Use robust scaling (median and MAD) instead of StandardScaler
                from sklearn.preprocessing import RobustScaler
                robust_scaler = RobustScaler(with_centering=True, with_scaling=True)
                X_scaled = robust_scaler.fit_transform(X)

                # Store the robust scaler for future use
                self.scaler = robust_scaler

                # Additional clipping for extreme values after scaling
                X_scaled = np.clip(X_scaled, -10, 10)  # Clip to ±10 standard deviations

                self.logger.info(f"✅ Applied robust scaling with clipping to {X_scaled.shape} features")
                return X_scaled
            else:
                # Transform using existing scaler
                X_scaled = self.scaler.transform(X)
                # Apply clipping
                X_scaled = np.clip(X_scaled, -10, 10)
                return X_scaled

        except Exception as e:
            self.logger.error(f"Robust feature scaling error: {e}")
            # Fallback to original scaling
            try:
                if fit:
                    return self.scaler.fit_transform(X)
                else:
                    return self.scaler.transform(X)
            except:
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


def prepare_features_and_labels(data, use_feature_selection=True, max_features=15, test_size=0.2):
    """
    Standalone function to prepare features and labels for ML models.
    This is a convenience function that wraps the FeatureEngineer class.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with price and volume data
    use_feature_selection : bool
        Whether to use feature selection
    max_features : int
        Maximum number of features to select
    test_size : float
        Proportion of data to use for testing

    Returns:
    --------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    feature_cols : list
        List of feature column names
    scaler : StandardScaler
        Fitted scaler object
    """
    logger = get_logger(__name__)

    # Create feature engineer instance
    engineer = FeatureEngineer()

    # Handle empty data
    if data.empty:
        logger.error("Empty dataframe provided")
        return np.array([]), np.array([]), [], None

    # Use the existing FeatureEngineer to prepare features
    try:
        features_df = engineer.prepare_features(data)

        if features_df.empty:
            logger.error("No features created from data")
            return np.array([]), np.array([]), [], None

        # Create target variable using the enhanced robust target creation method
        features_df = engineer.create_robust_target(features_df)

        # Remove rows with NaN values
        features_df = features_df.dropna()

        if features_df.empty:
            logger.error("No valid data after target creation")
            return np.array([]), np.array([]), [], None

        # Get feature columns (exclude non-numeric and target columns)
        exclude_cols = ['target', 'Date', 'date', 'Datetime', 'datetime']
        feature_cols = [col for col in features_df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        if not feature_cols:
            logger.error("No feature columns found")
            return np.array([]), np.array([]), [], None

        # Prepare feature matrix and target
        X = features_df[feature_cols].fillna(0).values
        y = features_df['target'].fillna(0).values

        # Feature selection if requested
        if use_feature_selection and len(feature_cols) > max_features:
            from sklearn.feature_selection import SelectKBest, f_regression
            selector = SelectKBest(score_func=f_regression, k=max_features)
            X_selected = selector.fit_transform(X, y)

            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            feature_cols = [feature_cols[i] for i in selected_indices]
            X = X_selected

            logger.info(f"Selected {len(feature_cols)} features using f_regression")

        # Scale features
        scaler = engineer.scaler
        X_scaled = scaler.fit_transform(X)

        logger.info(f"Prepared features: X shape={X_scaled.shape}, y shape={y.shape}")
        logger.info(f"Feature columns: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

        return X_scaled, y, feature_cols, scaler

    except Exception as e:
        logger.error(f"Error in prepare_features_and_labels: {e}")
        return np.array([]), np.array([]), [], None
