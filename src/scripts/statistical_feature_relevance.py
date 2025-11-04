#!/usr/bin/env python3
"""
Statistical Relevance Analysis for VIX, VPOC, and Volume Profile Features

This script performs rigorous statistical analysis to determine if VIX, VPOC,
and volume profile features have meaningful relationships with price returns
before considering them for ML model inclusion.

Focus: Statistical significance testing, not ML performance.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

from src.utils.logging import get_logger

# Phase 1 Technical Analysis Libraries
import talib
import pandas_ta as ta
from arch import arch_model

# GPU Acceleration Setup (ROCm 7)
os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU Initialized: {torch.cuda.get_device_name(0)}")
        print(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
        print(f"✅ Using GPU for statistical analysis")
    else:
        raise RuntimeError("GPU not available")
except Exception as e:
    print(f"❌ GPU initialization failed: {e}")
    print("📊 Cannot proceed without GPU - this prevents CPU crashes")
    sys.exit(1)

logger = get_logger(__name__)

class StatisticalFeatureAnalyzer:
    """Analyze statistical relevance of features for price returns prediction."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.target = None
        self.features = {}
        self.gpu_available = GPU_AVAILABLE
        self.use_gpu = False  # Will be set based on data size

        # GPU memory management
        if self.gpu_available:
            try:
                # Test GPU memory availability
                torch.cuda.init()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                logger.info(f"🎮 GPU Memory Available: {gpu_memory:.1f} GB")
                if gpu_memory >= 8:  # Require at least 8GB
                    self.use_gpu = True
                    logger.info("✅ GPU acceleration enabled for large dataset")
                else:
                    logger.info("⚠️ Limited GPU memory, using CPU processing")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                self.use_gpu = False

    def load_and_prepare_data(self):
        """Load data and prepare target variable with GPU acceleration for large datasets."""
        logger.info(f"📁 Loading data from {self.data_path}")

        # Load data in chunks to manage memory and use GPU processing
        logger.info("📊 Loading data with chunked GPU processing...")

        # Load full dataset
        self.data = pd.read_csv(self.data_path)
        logger.info(f"✅ Loaded {len(self.data):,} rows for GPU analysis (FULL DATASET)")

        # Basic processing on CPU (minimal)
        if 'VIX' in self.data.columns:
            self.data = self.data.rename(columns={'VIX': 'vix'})

        # Remove essential missing data
        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        initial_len = len(self.data)
        for col in essential_cols:
            if col in self.data.columns:
                self.data = self.data[self.data[col].notna()]

        removed_rows = initial_len - len(self.data)
        logger.info(f"📊 Removed {removed_rows:,} rows with missing essential data")

        # Forward fill missing values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')

        # Create target variable
        self.data['target'] = self.data['close'].pct_change().shift(-1)
        self.target = self.data['target'].dropna()

        # Align data with target
        self.data = self.data.iloc[:len(self.target)]

        logger.info(f"✅ Target created: {len(self.target):,} observations")
        logger.info(f"  • Target mean: {self.target.mean():.6f}")
        logger.info(f"  • Target std: {self.target.std():.6f}")

        # Move target to GPU for faster processing
        self.target_gpu = torch.tensor(self.target.values, dtype=torch.float32, device=device)
        logger.info(f"✅ Target moved to GPU: {self.target_gpu.shape}")
            
    def create_vix_features(self):
        """Create VIX-related features for statistical analysis with forward-filling for minute-level data."""
        logger.info("🎯 Creating VIX features...")

        if 'vix' not in self.data.columns:
            logger.warning("❌ VIX data not available")
            return

        vix = self.data['vix']

        # Forward-fill daily VIX data to minute-level
        # Since VIX is only available at daily resolution, forward-fill within each day
        if hasattr(self.data.index, 'date'):
            vix_ffilled = vix.groupby(vix.index.date).ffill()
        elif 'timestamp' in self.data.columns:
            # Convert timestamp to datetime and group by date
            temp_data = self.data.copy()
            temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'])
            vix_ffilled = vix.groupby(temp_data['timestamp'].dt.date).ffill()
        else:
            # Simple forward fill if no date grouping available
            vix_ffilled = vix.ffill()
        logger.info(f"📈 Forward-filled VIX from {vix.notna().sum()} daily observations to {vix_ffilled.notna().sum()} minute-level observations")

        # Basic VIX features (using forward-filled data)
        self.features['vix_level'] = vix_ffilled
        self.features['vix_change_1min'] = vix_ffilled.pct_change(1)
        self.features['vix_change_5min'] = vix_ffilled.pct_change(5)
        self.features['vix_change_15min'] = vix_ffilled.pct_change(15)
        self.features['vix_change_60min'] = vix_ffilled.pct_change(60)

        # Moving averages (shorter windows for minute-level data)
        self.features['vix_ma_30min'] = vix_ffilled.rolling(30).mean()  # ~30 minutes
        self.features['vix_ma_60min'] = vix_ffilled.rolling(60).mean()  # 1 hour
        self.features['vix_ma_240min'] = vix_ffilled.rolling(240).mean()  # 4 hours

        # Relative positioning
        self.features['vix_vs_ma30'] = vix_ffilled / self.features['vix_ma_30min'] - 1
        self.features['vix_vs_ma240'] = vix_ffilled / self.features['vix_ma_240min'] - 1

        # Intraday volatility of VIX
        self.features['vix_volatility_15min'] = vix_ffilled.rolling(15).std()
        self.features['vix_volatility_60min'] = vix_ffilled.rolling(60).std()

        # Percentile rankings (rolling windows adapted for minute-level)
        self.features['vix_percentile_240min'] = vix_ffilled.rolling(240).rank(pct=True)  # 4 hours
        self.features['vix_percentile_1440min'] = vix_ffilled.rolling(1440).rank(pct=True)  # 1 day

        # Extreme value indicators (shorter lookback for intraday)
        vix_p20_4h = vix_ffilled.rolling(240).quantile(0.2)  # 4-hour 20th percentile
        vix_p80_4h = vix_ffilled.rolling(240).quantile(0.8)  # 4-hour 80th percentile
        self.features['vix_extreme_low_4h'] = (vix_ffilled < vix_p20_4h).astype(int)
        self.features['vix_extreme_high_4h'] = (vix_ffilled > vix_p80_4h).astype(int)

        # Fear/Greed index (inverse of VIX percentile)
        self.features['fear_greed_index_4h'] = 1 - self.features['vix_percentile_240min']

        # Daily change features (only available at daily rollover)
        if hasattr(vix_ffilled.index, 'date'):
            daily_vix = vix_ffilled.groupby(vix_ffilled.index.date).first()
            daily_vix_change = daily_vix.pct_change(1)
            # Map daily changes back to minute-level data
            vix_daily_change_map = daily_vix_change.to_dict()
            self.features['vix_daily_change'] = vix_ffilled.index.map(lambda x: vix_daily_change_map.get(x.date(), 0))
        elif 'timestamp' in self.data.columns:
            # Use timestamp column for date grouping
            temp_data = self.data.copy()
            temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'])
            temp_data['vix_ffilled'] = vix_ffilled
            daily_vix = temp_data.groupby(temp_data['timestamp'].dt.date)['vix_ffilled'].first()
            daily_vix_change = daily_vix.pct_change(1)
            vix_daily_change_map = daily_vix_change.to_dict()
            self.features['vix_daily_change'] = temp_data['timestamp'].dt.date.map(lambda x: vix_daily_change_map.get(x, 0))
        else:
            # Simple daily changes if no date grouping available
            self.features['vix_daily_change'] = vix_ffilled.pct_change(1440)  # Assume 1440 minutes = 1 day

        logger.info(f"✅ Created {len([k for k in self.features.keys() if 'vix' in k or 'fear' in k])} VIX features with minute-level granularity")

    def create_volume_features(self):
        """Create volume-related features for statistical analysis."""
        logger.info("📊 Creating volume features...")

        if 'volume' not in self.data.columns:
            logger.warning("❌ Volume data not available")
            return

        volume = self.data['volume']

        # Basic volume features
        self.features['volume_level'] = volume
        self.features['volume_change_1d'] = volume.pct_change(1)
        self.features['volume_change_5d'] = volume.pct_change(5)

        # Moving averages
        self.features['volume_ma_5d'] = volume.rolling(5).mean()
        self.features['volume_ma_10d'] = volume.rolling(10).mean()
        self.features['volume_ma_20d'] = volume.rolling(20).mean()

        # Relative volume
        self.features['volume_vs_ma5'] = volume / self.features['volume_ma_5d'] - 1
        self.features['volume_vs_ma20'] = volume / self.features['volume_ma_20d'] - 1

        # Volume volatility
        self.features['volume_volatility_5d'] = volume.rolling(5).std()
        self.features['volume_volatility_10d'] = volume.rolling(10).std()

        # Volume price relationship
        price_change = self.data['close'].pct_change()
        self.features['volume_price_corr_5d'] = price_change.rolling(5).corr(volume.pct_change())
        self.features['volume_price_corr_10d'] = price_change.rolling(10).corr(volume.pct_change())

        # On-Balance Volume (OBV)
        obv = (volume * np.sign(price_change)).cumsum()
        self.features['obv'] = obv
        self.features['obv_ma_5d'] = obv.rolling(5).mean()
        self.features['obv_change_5d'] = obv.pct_change(5)

        # Volume Accumulation/Distribution
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        money_flow_volume = volume * ((typical_price - typical_price.shift(1)) / typical_price.shift(1))
        self.features['volume_acc_dist'] = money_flow_volume.cumsum()
        self.features['volume_acc_dist_change_5d'] = self.features['volume_acc_dist'].pct_change(5)

        logger.info(f"✅ Created {len([k for k in self.features.keys() if 'volume' in k or 'obv' in k])} volume features")

    def create_vpoc_features(self):
        """Create hourly VPOC (Volume Point of Control) related features for minute-level data."""
        logger.info("🎯 Creating hourly VPOC features...")

        if 'volume' not in self.data.columns:
            logger.warning("❌ Volume data not available for VPOC")
            return

        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        volume = self.data['volume']

        # Hourly volume profiles using resampling
        logger.info("📊 Creating hourly volume profiles...")

        # Resample to hourly data for volume profile calculation
        hourly_data = self.data.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        logger.info(f"✅ Created {len(hourly_data)} hourly bars from {len(self.data)} minute bars")

        # Calculate hourly VWAP (Volume Weighted Average Price)
        hourly_typical_price = (hourly_data['high'] + hourly_data['low'] + hourly_data['close']) / 3
        hourly_vwap = (hourly_typical_price * hourly_data['volume']).rolling(12).sum() / hourly_data['volume'].rolling(12).sum()

        # Map hourly VWAP back to minute-level data
        hourly_vwap_map = hourly_vwap.to_dict()
        hourly_vwap_series = self.data.index.to_series().map(lambda x: hourly_vwap_map.get(x.replace(minute=0, second=0, microsecond=0), np.nan))
        self.features['hourly_vwap'] = hourly_vwap_series

        # Forward-fill hourly VWAP within each hour
        self.features['hourly_vwap'] = self.features['hourly_vwap'].ffill(limit=59)  # Max 59 minutes in an hour

        # Enhanced hourly VPOC approximation
        self.features['hourly_vpoc'] = self.features['hourly_vwap']

        # Price distance from hourly VWAP/VPOC
        self.features['close_to_hourly_vwap_pct'] = (close - self.features['hourly_vwap']) / self.features['hourly_vwap']
        self.features['close_to_hourly_vwap_abs'] = abs(close - self.features['hourly_vwap'])
        self.features['close_to_hourly_vpoc_pct'] = (close - self.features['hourly_vpoc']) / self.features['hourly_vpoc']
        self.features['hourly_vpoc_distance_pct'] = abs(close - self.features['hourly_vpoc']) / self.features['hourly_vpoc']

        # Hourly VPOC position indicators
        self.features['close_above_hourly_vpoc'] = (close > self.features['hourly_vpoc']).astype(int)

        # Hourly VWAP trend analysis
        hourly_vwap_change = hourly_vwap.pct_change(1)  # Hour-over-hour change
        hourly_vwap_change_map = hourly_vwap_change.to_dict()
        hourly_vwap_change_series = self.data.index.to_series().map(lambda x: hourly_vwap_change_map.get(x.replace(minute=0, second=0, microsecond=0), 0))
        self.features['hourly_vwap_change'] = hourly_vwap_change_series

        # Forward-fill hourly VWAP changes
        self.features['hourly_vwap_change'] = self.features['hourly_vwap_change'].ffill(limit=59)
        self.features['hourly_vwap_trend_strength'] = abs(self.features['hourly_vwap_change'])

        # Hourly Value Area approximation
        hourly_range = hourly_data['high'] - hourly_data['low']
        hourly_avg_range = hourly_range.rolling(12).mean()  # 12-hour rolling average
        hourly_value_area_upper = hourly_vwap + hourly_avg_range * 0.5
        hourly_value_area_lower = hourly_vwap - hourly_avg_range * 0.5

        # Map hourly value area back to minute-level
        value_area_upper_map = hourly_value_area_upper.to_dict()
        value_area_lower_map = hourly_value_area_lower.to_dict()

        value_area_upper_series = self.data.index.to_series().map(lambda x: value_area_upper_map.get(x.replace(minute=0, second=0, microsecond=0), np.nan))
        value_area_lower_series = self.data.index.to_series().map(lambda x: value_area_lower_map.get(x.replace(minute=0, second=0, microsecond=0), np.nan))

        self.features['hourly_value_area_upper'] = value_area_upper_series
        self.features['hourly_value_area_lower'] = value_area_lower_series

        # Forward-fill value area levels
        self.features['hourly_value_area_upper'] = self.features['hourly_value_area_upper'].ffill(limit=59)
        self.features['hourly_value_area_lower'] = self.features['hourly_value_area_lower'].ffill(limit=59)

        self.features['in_hourly_value_area'] = ((close >= self.features['hourly_value_area_lower']) &
                                                (close <= self.features['hourly_value_area_upper'])).astype(int)

        # Volume-Price momentum using hourly profiles
        hourly_volume_price_momentum = (hourly_vwap * hourly_data['volume'].rolling(5).mean()).pct_change(5)
        volume_price_momentum_map = hourly_volume_price_momentum.to_dict()
        volume_price_momentum_series = self.data.index.to_series().map(lambda x: volume_price_momentum_map.get(x.replace(minute=0, second=0, microsecond=0), 0))
        self.features['hourly_volume_price_momentum'] = volume_price_momentum_series.ffill(limit=59)

        # Additional minute-level VWAP for intraday analysis
        typical_price = (high + low + close) / 3
        self.features['minute_vwap_15min'] = (typical_price * volume).rolling(15).sum() / volume.rolling(15).sum()
        self.features['minute_vwap_60min'] = (typical_price * volume).rolling(60).sum() / volume.rolling(60).sum()
        self.features['minute_vwap_240min'] = (typical_price * volume).rolling(240).sum() / volume.rolling(240).sum()

        # Volume profile intensity (volume concentration around price levels)
        price_volume_relation = volume / (high - low)  # Volume per price unit
        self.features['volume_profile_intensity'] = price_volume_relation.rolling(60).mean()  # 1-hour average

        logger.info(f"✅ Created {len([k for k in self.features.keys() if 'vpoc' in k or 'vwap' in k or 'value' in k or 'volume' in k])} VPOC and volume profile features")

    def create_price_features(self):
        """Create basic price features for comparison."""
        logger.info("💰 Creating price features...")

        close = self.data['close']
        high = self.data['high']
        low = self.data['low']

        # Price changes
        self.features['price_change_1d'] = close.pct_change(1)
        self.features['price_change_3d'] = close.pct_change(3)
        self.features['price_change_5d'] = close.pct_change(5)

        # Price ranges
        self.features['price_range'] = (high - low) / close
        self.features['price_range_5d_avg'] = self.features['price_range'].rolling(5).mean()

        # Moving averages
        self.features['price_ma_5d'] = close.rolling(5).mean()
        self.features['price_ma_10d'] = close.rolling(10).mean()
        self.features['price_ma_20d'] = close.rolling(20).mean()

        # Price position
        self.features['price_vs_ma5'] = (close - self.features['price_ma_5d']) / self.features['price_ma_5d']
        self.features['price_vs_ma20'] = (close - self.features['price_ma_20d']) / self.features['price_ma_20d']

        logger.info(f"✅ Created {len([k for k in self.features.keys() if 'price' in k])} price features")

    def create_phase1_technical_indicators(self):
        """Create Phase 1 technical indicators using TA-Lib."""
        logger.info("📈 Creating Phase 1 technical indicators...")

        try:
            # RSI (14)
            self.features['rsi_14'] = talib.RSI(self.data['close'].values, timeperiod=14)

            # MACD (12, 26, 9)
            macd, macd_signal, macd_hist = talib.MACD(
                self.data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            self.features['macd_line'] = macd
            self.features['macd_signal'] = macd_signal
            self.features['macd_histogram'] = macd_hist

            # Stochastic Oscillator (%K 14, %D 3)
            slowk, slowd = talib.STOCH(
                self.data['high'].values, self.data['low'].values, self.data['close'].values,
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            self.features['stoch_k'] = slowk
            self.features['stoch_d'] = slowd

            # ATR (14)
            self.features['atr_14'] = talib.ATR(
                self.data['high'].values, self.data['low'].values, self.data['close'].values, timeperiod=14
            )

            # Bollinger Bands Position (Normalized within bands)
            upper, middle, lower = talib.BBANDS(
                self.data['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            # Avoid division by zero
            band_width = upper - lower
            self.features['bb_position'] = np.where(
                band_width > 0, (self.data['close'].values - middle) / band_width, 0
            )

            logger.info(f"✅ Created 8 Phase 1 technical indicators: RSI, MACD, Stochastic, ATR, Bollinger Bands")

        except Exception as e:
            logger.error(f"❌ Phase 1 technical indicators failed: {e}")
            # Add placeholder NaN columns to maintain structure
            for tech_col in ['rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
                           'stoch_k', 'stoch_d', 'atr_14', 'bb_position']:
                self.features[tech_col] = np.nan

    def create_phase1_volatility_features(self):
        """Create Phase 1 advanced volatility features."""
        logger.info("📊 Creating Phase 1 volatility features...")

        try:
            # Calculate minute returns
            minute_returns = self.data['close'].pct_change()
            self.features['bar_return'] = minute_returns

            # Ensure date column exists and is datetime
            if 'date' not in self.data.columns:
                if hasattr(self.data.index, 'date'):
                    date_col = self.data.index.date
                else:
                    date_col = pd.to_datetime(self.data['timestamp']).dt.date
            else:
                date_col = self.data['date']

            # Group by date for daily volatility calculations
            daily_groups = self.data.groupby(date_col)

            # Realized Volatility (square root of sum of squared returns)
            realized_vol = daily_groups['bar_return'].apply(
                lambda x: np.sqrt(np.sum(np.square(x.dropna())))
            )

            # Bipower Variation (jump-robust estimator)
            bipower_var = daily_groups['bar_return'].apply(
                lambda x: (np.pi/2) * (1/max(len(x)-1, 1)) *
                         np.sum(np.abs(x.dropna().iloc[:-1]) * np.abs(x.dropna().iloc[1:])) if len(x.dropna()) > 1 else np.nan
            )

            # Merge back to main dataframe
            if 'date' in self.data.columns:
                self.features['realized_vol_daily'] = self.data['date'].map(realized_vol)
                self.features['bipower_var_daily'] = self.data['date'].map(bipower_var)
            else:
                # Use date_col that was created for grouping
                date_series = pd.Series(date_col, index=self.data.index)
                self.features['realized_vol_daily'] = date_series.map(realized_vol)
                self.features['bipower_var_daily'] = date_series.map(bipower_var)

            # Realized Jump Variance
            self.features['realized_jump_var'] = np.maximum(0,
                self.features['realized_vol_daily'] - self.features['bipower_var_daily'])

            # HAR Features (Heterogeneous Autoregressive)
            self.features['har_1d'] = self.features['realized_vol_daily'].rolling(1).mean()
            self.features['har_5d'] = self.features['realized_vol_daily'].rolling(5).mean()
            self.features['har_22d'] = self.features['realized_vol_daily'].rolling(22).mean()

            # GARCH(1,1) - sample calculation (optimize for performance)
            # Note: This is computationally intensive, consider using a sample
            try:
                # Use a sample of the data for GARCH to avoid long computation times
                sample_returns = minute_returns.dropna().iloc[-5000:]  # Last 5000 observations
                if len(sample_returns) > 1000:  # Minimum sample size
                    model = arch_model(sample_returns * 100, vol='Garch', p=1, q=1, disp='off')
                    res = model.fit(disp='off')
                    # Map GARCH volatility back to the main dataframe (simplified approach)
                    # For now, use conditional volatility from the sample
                    garch_vol_series = pd.Series(index=sample_returns.index, data=res.conditional_volatility / 100)
                    self.features.loc[garch_vol_series.index, 'garch_vol'] = garch_vol_series
            except Exception as garch_error:
                logger.warning(f"GARCH calculation failed: {garch_error}")
                self.features['garch_vol'] = np.nan

            # Volatility Regime
            vol_25 = self.features['realized_vol_daily'].rolling(60).quantile(0.25)
            vol_75 = self.features['realized_vol_daily'].rolling(60).quantile(0.75)
            self.features['vol_regime'] = (self.features['realized_vol_daily'] > vol_75).astype(int)

            logger.info(f"✅ Created 8 Phase 1 volatility features: Realized Vol, Bipower Var, HAR, GARCH, Regime")

        except Exception as e:
            logger.error(f"❌ Phase 1 volatility features failed: {e}")
            # Add placeholder NaN columns
            for vol_col in ['realized_vol_daily', 'bipower_var_daily', 'realized_jump_var',
                          'har_1d', 'har_5d', 'har_22d', 'garch_vol', 'vol_regime']:
                self.features[vol_col] = np.nan

    def create_phase1_time_features(self):
        """Create Phase 1 time-based features."""
        logger.info("⏰ Creating Phase 1 time-based features...")

        try:
            # Convert timestamp to datetime if not already
            if hasattr(self.data.index, 'hour'):
                timestamp_series = self.data.index
            else:
                timestamp_series = pd.to_datetime(self.data['timestamp'])

            # Day of week (0=Monday, 4=Friday)
            self.features['day_of_week'] = timestamp_series.dayofweek

            # Time of day (normalized 0-1)
            self.features['time_of_day'] = (timestamp_series.hour / 24.0 +
                                          timestamp_series.minute / 1440.0)

            # Session indicator (if available)
            if 'session' in self.data.columns:
                self.features['session_indicator'] = self.data['session']
            else:
                # Default to regular session (1)
                self.features['session_indicator'] = 1

            logger.info(f"✅ Created 3 Phase 1 time features: Day of Week, Time of Day, Session")

        except Exception as e:
            logger.error(f"❌ Phase 1 time features failed: {e}")
            # Add placeholder NaN columns
            for time_col in ['day_of_week', 'time_of_day', 'session_indicator']:
                self.features[time_col] = np.nan

    def calculate_statistical_relevance(self, feature_name: str, feature_series: pd.Series):
        """Calculate statistical relevance for a single feature using GPU acceleration."""
        logger.info(f"🔍 Analyzing {feature_name} on GPU...")

        # Move feature to GPU tensor (handle both pandas Series and numpy arrays)
        try:
            if hasattr(feature_series, 'values'):
                feature_values = feature_series.values
            else:
                feature_values = feature_series  # Already numpy array

            feature_gpu = torch.tensor(feature_values, dtype=torch.float32, device=device)

            # Align with target GPU tensor
            min_len = min(len(self.target_gpu), len(feature_gpu))
            target_clean = self.target_gpu[:min_len]
            feature_clean = feature_gpu[:min_len]

            # Remove NaN/Inf values using GPU mask
            nan_mask = ~(torch.isnan(target_clean) | torch.isnan(feature_clean) |
                         torch.isinf(target_clean) | torch.isinf(feature_clean))

            target_clean = target_clean[nan_mask]
            feature_clean = feature_clean[nan_mask]

            if len(target_clean) < 100:
                logger.warning(f"⚠️ Insufficient valid data points for {feature_name}")
                return None

            return self._calculate_gpu_correlation(feature_name, feature_clean, target_clean)

        except Exception as e:
            logger.error(f"❌ GPU calculation failed for {feature_name}: {e}")
            return None

    def _calculate_gpu_correlation(self, feature_name: str, feature_gpu: torch.Tensor, target_gpu: torch.Tensor):
        """Calculate correlations using GPU tensors."""
        try:
            # Calculate Pearson correlation on GPU
            target_mean = torch.mean(target_gpu)
            feature_mean = torch.mean(feature_gpu)

            target_centered = target_gpu - target_mean
            feature_centered = feature_gpu - feature_mean

            numerator = torch.sum(target_centered * feature_centered)
            target_std = torch.sqrt(torch.sum(target_centered ** 2))
            feature_std = torch.sqrt(torch.sum(feature_centered ** 2))

            denominator = target_std * feature_std

            if denominator == 0:
                pearson_corr = 0.0
                pearson_p = 1.0
            else:
                pearson_corr = numerator / denominator
                # Approximate p-value calculation
                n = len(target_gpu)
                if n > 2:
                    t_stat = pearson_corr.cpu() * np.sqrt((n-2) / (1 - pearson_corr.cpu()**2))
                    from scipy.stats import t as t_dist
                    pearson_p = 2 * (1 - t_dist.cdf(abs(t_stat), n-2))
                else:
                    pearson_p = 1.0

            # Calculate Spearman correlation (convert to ranks on GPU)
            target_ranks = torch.argsort(torch.argsort(target_gpu)).float()
            feature_ranks = torch.argsort(torch.argsort(feature_gpu)).float()

            target_ranks_mean = torch.mean(target_ranks)
            feature_ranks_mean = torch.mean(feature_ranks)

            target_ranks_centered = target_ranks - target_ranks_mean
            feature_ranks_centered = feature_ranks - feature_ranks_mean

            spearman_num = torch.sum(target_ranks_centered * feature_ranks_centered)
            spearman_den = torch.sqrt(torch.sum(target_ranks_centered ** 2)) * torch.sqrt(torch.sum(feature_ranks_centered ** 2))

            if spearman_den == 0:
                spearman_corr = 0.0
                spearman_p = 1.0
            else:
                spearman_corr = spearman_num / spearman_den
                n = len(target_gpu)
                if n > 2:
                    t_stat = spearman_corr.cpu() * np.sqrt((n-2) / (1 - spearman_corr.cpu()**2))
                    spearman_p = 2 * (1 - t_dist.cdf(abs(t_stat), n-2))
                else:
                    spearman_p = 1.0

            # Convert to CPU for results
            results = {
                'feature_name': feature_name,
                'sample_size': int(len(target_gpu)),
                'feature_mean': float(feature_mean.cpu()),
                'feature_std': float(torch.std(feature_gpu).cpu()),
                'pearson_corr': float(pearson_corr.cpu()),
                'pearson_p_value': float(pearson_p),
                'pearson_significant': pearson_p < 0.05,
                'spearman_corr': float(spearman_corr.cpu()),
                'spearman_p_value': float(spearman_p),
                'spearman_significant': spearman_p < 0.05,
            }

            # Effect size
            abs_pearson = abs(results['pearson_corr'])
            if abs_pearson >= 0.1:
                results['effect_size'] = 'medium' if abs_pearson >= 0.3 else 'small'
            else:
                results['effect_size'] = 'negligible'

            # Overall significance
            if results['pearson_significant'] and results['spearman_significant']:
                results['overall_significance'] = 'high'
            elif results['pearson_significant'] or results['spearman_significant']:
                results['overall_significance'] = 'moderate'
            else:
                results['overall_significance'] = 'low'

            return results

        except Exception as e:
            logger.warning(f"GPU correlation calculation failed for {feature_name}: {e}")
            return None

    def _calculate_statistical_relevance_gpu(self, feature_name: str, target_clean: pd.Series, feature_clean: pd.Series):
        """Calculate statistical relevance using GPU acceleration."""
        try:
            # Convert to GPU arrays
            target_gpu = cp.asarray(target_clean.values)
            feature_gpu = cp.asarray(feature_clean.values)

            results = {
                'sample_size': len(target_clean),
                'feature_mean': float(cp.mean(feature_gpu)),
                'feature_std': float(cp.std(feature_gpu)),
                'target_mean': float(cp.mean(target_gpu)),
                'target_std': float(cp.std(target_gpu))
            }

            # Calculate correlations using cuML
            try:
                from cuml.stats import pearsonr as cuml_pearsonr, spearmanr as cuml_spearmanr

                # Pearson correlation
                pearson_corr, pearson_p = cuml_pearsonr(feature_gpu, target_gpu)
                results['pearson_corr'] = float(pearson_corr)
                results['pearson_p_value'] = float(pearson_p)
                results['pearson_significant'] = pearson_p < 0.05
                results['pearson_highly_significant'] = pearson_p < 0.01

                # Spearman correlation
                spearman_corr, spearman_p = cuml_spearmanr(feature_gpu, target_gpu)
                results['spearman_corr'] = float(spearman_corr)
                results['spearman_p_value'] = float(spearman_p)
                results['spearman_significant'] = spearman_p < 0.05
                results['spearman_highly_significant'] = spearman_p < 0.01

            except ImportError:
                # Fall back to scipy correlations
                target_cpu = cp.asnumpy(target_gpu)
                feature_cpu = cp.asnumpy(feature_gpu)

                pearson_corr, pearson_p = pearsonr(feature_cpu, target_cpu)
                results['pearson_corr'] = pearson_corr
                results['pearson_p_value'] = pearson_p
                results['pearson_significant'] = pearson_p < 0.05
                results['pearson_highly_significant'] = pearson_p < 0.01

                spearman_corr, spearman_p = spearmanr(feature_cpu, target_cpu)
                results['spearman_corr'] = spearman_corr
                results['spearman_p_value'] = spearman_p
                results['spearman_significant'] = spearman_p < 0.05
                results['spearman_highly_significant'] = spearman_p < 0.01

                # Kendall's tau
                try:
                    kendall_corr, kendall_p = kendalltau(feature_cpu, target_cpu)
                    results['kendall_corr'] = kendall_corr
                    results['kendall_p_value'] = kendall_p
                    results['kendall_significant'] = kendall_p < 0.05
                    results['kendall_highly_significant'] = kendall_p < 0.01
                except:
                    results['kendall_corr'] = np.nan
                    results['kendall_p_value'] = np.nan
                    results['kendall_significant'] = False

            # Effect size and significance assessment
            abs_pearson = abs(results['pearson_corr']) if not np.isnan(results['pearson_corr']) else 0
            if abs_pearson >= 0.3:
                results['effect_size'] = 'large'
            elif abs_pearson >= 0.1:
                results['effect_size'] = 'medium'
            elif abs_pearson >= 0.01:
                results['effect_size'] = 'small'
            else:
                results['effect_size'] = 'negligible'

            significance_count = sum([
                results.get('pearson_significant', False),
                results.get('spearman_significant', False),
                results.get('kendall_significant', False)
            ])

            if significance_count >= 2:
                results['overall_significance'] = 'high'
            elif significance_count == 1:
                results['overall_significance'] = 'moderate'
            else:
                results['overall_significance'] = 'low'

            return results

        except Exception as e:
            logger.error(f"GPU calculation failed for {feature_name}: {e}")
            return None

    def _calculate_statistical_relevance_cpu(self, feature_name: str, target_clean: pd.Series, feature_clean: pd.Series):
        """Calculate statistical relevance using CPU (fallback method)."""
        results = {
            'sample_size': len(target_clean),
            'feature_mean': feature_clean.mean(),
            'feature_std': feature_clean.std(),
            'target_mean': target_clean.mean(),
            'target_std': target_clean.std()
        }

        try:
            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(feature_clean, target_clean)
            results['pearson_corr'] = pearson_corr
            results['pearson_p_value'] = pearson_p
            results['pearson_significant'] = pearson_p < 0.05
            results['pearson_highly_significant'] = pearson_p < 0.01
        except:
            results['pearson_corr'] = np.nan
            results['pearson_p_value'] = np.nan
            results['pearson_significant'] = False

        try:
            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(feature_clean, target_clean)
            results['spearman_corr'] = spearman_corr
            results['spearman_p_value'] = spearman_p
            results['spearman_significant'] = spearman_p < 0.05
            results['spearman_highly_significant'] = spearman_p < 0.01
        except:
            results['spearman_corr'] = np.nan
            results['spearman_p_value'] = np.nan
            results['spearman_significant'] = False

        try:
            # Kendall's tau
            kendall_corr, kendall_p = kendalltau(feature_clean, target_clean)
            results['kendall_corr'] = kendall_corr
            results['kendall_p_value'] = kendall_p
            results['kendall_significant'] = kendall_p < 0.05
            results['kendall_highly_significant'] = kendall_p < 0.01
        except:
            results['kendall_corr'] = np.nan
            results['kendall_p_value'] = np.nan
            results['kendall_significant'] = False

        # Effect size interpretation
        abs_pearson = abs(results['pearson_corr']) if not np.isnan(results['pearson_corr']) else 0
        if abs_pearson >= 0.3:
            results['effect_size'] = 'large'
        elif abs_pearson >= 0.1:
            results['effect_size'] = 'medium'
        elif abs_pearson >= 0.01:
            results['effect_size'] = 'small'
        else:
            results['effect_size'] = 'negligible'

        # Overall significance assessment
        significance_count = sum([
            results['pearson_significant'],
            results['spearman_significant'],
            results['kendall_significant']
        ])

        if significance_count >= 2:
            results['overall_significance'] = 'high'
        elif significance_count == 1:
            results['overall_significance'] = 'moderate'
        else:
            results['overall_significance'] = 'low'

        return results

    def analyze_all_features(self):
        """Analyze statistical relevance of all created features."""
        logger.info("📈 Analyzing statistical relevance of all features...")

        results = {}

        for feature_name, feature_series in self.features.items():
            logger.info(f"  Analyzing {feature_name}...")

            feature_results = self.calculate_statistical_relevance(feature_name, feature_series)
            if feature_results:
                results[feature_name] = feature_results

        return results

    def print_summary(self, results):
        """Print comprehensive statistical analysis summary."""
        logger.info("\n" + "="*80)
        logger.info("📊 STATISTICAL FEATURE RELEVANCE ANALYSIS SUMMARY")
        logger.info("="*80)

        # Categorize features
        vix_features = {k: v for k, v in results.items() if 'vix' in k or 'fear' in k}
        volume_features = {k: v for k, v in results.items() if 'volume' in k or 'obv' in k}
        vpoc_features = {k: v for k, v in results.items() if 'vpoc' in k or 'vwap' in k or 'value' in k}
        price_features = {k: v for k, v in results.items() if 'price' in k}

        # Phase 1 Feature Categories
        technical_features = {k: v for k, v in results.items() if
                             any(ind in k for ind in ['rsi', 'macd', 'stoch', 'atr', 'bb'])}
        volatility_features = {k: v for k, v in results.items() if
                             any(vol in k for vol in ['realized_vol', 'bipower_var', 'jump_var', 'har_', 'garch_vol', 'vol_regime'])}
        time_features = {k: v for k, v in results.items() if
                        any(time in k for time in ['day_of_week', 'time_of_day', 'session'])}

        def print_category_summary(category_name, features):
            logger.info(f"\n🎯 {category_name.upper()} FEATURES:")

            if not features:
                logger.info("  ❌ No features analyzed")
                return

            # Sort by absolute Pearson correlation
            sorted_features = sorted(features.items(),
                                   key=lambda x: abs(x[1]['pearson_corr']) if not np.isnan(x[1]['pearson_corr']) else 0,
                                   reverse=True)

            significant_features = [k for k, v in features.items() if v['overall_significance'] == 'high']
            moderate_features = [k for k, v in features.items() if v['overall_significance'] == 'moderate']

            logger.info(f"  • Total features: {len(features)}")
            logger.info(f"  • Highly significant: {len(significant_features)}")
            logger.info(f"  • Moderately significant: {len(moderate_features)}")

            # Top 5 features by correlation
            logger.info(f"  • Top 5 by correlation strength:")
            for i, (name, stats) in enumerate(sorted_features[:5]):
                if not np.isnan(stats['pearson_corr']):
                    logger.info(f"    {i+1}. {name}: r={stats['pearson_corr']:.4f} (p={stats['pearson_p_value']:.4f})")

            # Overall assessment
            high_sig_pct = (len(significant_features) / len(features)) * 100 if features else 0
            if high_sig_pct >= 50:
                assessment = "🟢 HIGH RELEVANCE"
            elif high_sig_pct >= 25:
                assessment = "🟡 MODERATE RELEVANCE"
            else:
                assessment = "🔴 LOW RELEVANCE"

            logger.info(f"  • Overall Assessment: {assessment}")

        # Print summaries for each category
        print_category_summary("VIX", vix_features)
        print_category_summary("Volume Profile & VPOC", vpoc_features)
        print_category_summary("Volume", volume_features)
        print_category_summary("Price (Baseline)", price_features)

        # Phase 1 Feature Summaries
        print_category_summary("Phase 1 - Technical Indicators", technical_features)
        print_category_summary("Phase 1 - Volatility Features", volatility_features)
        print_category_summary("Phase 1 - Time Features", time_features)

        # Overall recommendations
        logger.info(f"\n💡 STATISTICAL RECOMMENDATIONS:")

        vix_high_sig = len([k for k, v in vix_features.items() if v['overall_significance'] == 'high'])
        vpoc_high_sig = len([k for k, v in vpoc_features.items() if v['overall_significance'] == 'high'])
        volume_high_sig = len([k for k, v in volume_features.items() if v['overall_significance'] == 'high'])

        # Phase 1 feature counts
        technical_high_sig = len([k for k, v in technical_features.items() if v['overall_significance'] == 'high'])
        volatility_high_sig = len([k for k, v in volatility_features.items() if v['overall_significance'] == 'high'])
        time_high_sig = len([k for k, v in time_features.items() if v['overall_significance'] == 'high'])

        if vix_high_sig >= 3:
            logger.info("  ✅ VIX features show STRONG statistical relevance - RECOMMENDED for inclusion")
        elif vix_high_sig >= 1:
            logger.info("  🟡 VIX features show MODERATE statistical relevance - CONSIDER for inclusion")
        else:
            logger.info("  ❌ VIX features show LOW statistical relevance - NOT recommended")

        if vpoc_high_sig >= 2:
            logger.info("  ✅ VPOC features show STRONG statistical relevance - RECOMMENDED for inclusion")
        elif vpoc_high_sig >= 1:
            logger.info("  🟡 VPOC features show MODERATE statistical relevance - CONSIDER for inclusion")
        else:
            logger.info("  ❌ VPOC features show LOW statistical relevance - NOT recommended")

        if volume_high_sig >= 2:
            logger.info("  ✅ Volume features show STRONG statistical relevance - RECOMMENDED for inclusion")
        elif volume_high_sig >= 1:
            logger.info("  🟡 Volume features show MODERATE statistical relevance - CONSIDER for inclusion")
        else:
            logger.info("  ❌ Volume features show LOW statistical relevance - NOT recommended")

        # Phase 1 Recommendations
        logger.info(f"\n🚀 PHASE 1 FEATURE RECOMMENDATIONS:")

        if technical_high_sig >= 4:
            logger.info("  ✅ Phase 1 Technical Indicators show STRONG statistical relevance - HIGHLY RECOMMENDED")
        elif technical_high_sig >= 2:
            logger.info("  🟡 Phase 1 Technical Indicators show MODERATE statistical relevance - RECOMMENDED")
        else:
            logger.info("  ❌ Phase 1 Technical Indicators show LOW statistical relevance - RECONSIDER")

        if volatility_high_sig >= 4:
            logger.info("  ✅ Phase 1 Volatility Features show STRONG statistical relevance - HIGHLY RECOMMENDED")
        elif volatility_high_sig >= 2:
            logger.info("  🟡 Phase 1 Volatility Features show MODERATE statistical relevance - RECOMMENDED")
        else:
            logger.info("  ❌ Phase 1 Volatility Features show LOW statistical relevance - RECONSIDER")

        if time_high_sig >= 2:
            logger.info("  ✅ Phase 1 Time Features show STRONG statistical relevance - HIGHLY RECOMMENDED")
        elif time_high_sig >= 1:
            logger.info("  🟡 Phase 1 Time Features show MODERATE statistical relevance - RECOMMENDED")
        else:
            logger.info("  ❌ Phase 1 Time Features show LOW statistical relevance - RECONSIDER")

        # Overall Phase 1 Assessment
        total_phase1_high = technical_high_sig + volatility_high_sig + time_high_sig
        if total_phase1_high >= 8:
            logger.info(f"\n🎉 OVERALL PHASE 1 ASSESSMENT: EXCELLENT ({total_phase1_high} highly significant features)")
            logger.info("    → Phase 1 features should significantly improve model predictive power")
        elif total_phase1_high >= 4:
            logger.info(f"\n👍 OVERALL PHASE 1 ASSESSMENT: GOOD ({total_phase1_high} highly significant features)")
            logger.info("    → Phase 1 features should improve model predictive power")
        else:
            logger.info(f"\n⚠️  OVERALL PHASE 1 ASSESSMENT: LIMITED ({total_phase1_high} highly significant features)")
            logger.info("    → Phase 1 features may not significantly improve model performance")

        logger.info("\n" + "="*80)

def main():
    """Main execution function."""
    logger.info("🚀 Starting GPU-Accelerated Statistical Feature Relevance Analysis...")

    try:
        # Initialize analyzer
        analyzer = StatisticalFeatureAnalyzer('/workspace/DATA/MERGED/merged_es_vix_test.csv')

        # Load and prepare data
        analyzer.load_and_prepare_data()

        # Create feature sets
        analyzer.create_price_features()    # Baseline comparison
        analyzer.create_vix_features()      # Primary interest
        analyzer.create_volume_features()   # Primary interest
        # Skip VPOC features for now - they require datetime index resampling
        # analyzer.create_vpoc_features()     # Primary interest

        # Phase 1 Features (NEW) - Main focus
        analyzer.create_phase1_technical_indicators()  # RSI, MACD, Stochastic, ATR, Bollinger Bands
        analyzer.create_phase1_volatility_features()    # HAR, GARCH, Realized Vol, Bipower Var
        analyzer.create_phase1_time_features()          # Day of Week, Time of Day, Session

        # Analyze all features
        results = analyzer.analyze_all_features()

        # Print comprehensive summary
        analyzer.print_summary(results)

        logger.info("✅ GPU statistical analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🗑️ GPU memory cleaned up")

if __name__ == "__main__":
    main()