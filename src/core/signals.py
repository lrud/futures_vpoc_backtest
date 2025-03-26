"""
VPOC signal generation module for futures trading strategy.
Focuses on volume profile analysis and signal generation.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from src.utils.logging import get_logger


class VolumeProfileAnalyzer:
    """
    Analyzes volume profiles to identify VPOC and key price levels.
    """
    
    def __init__(self, price_precision=0.25):
        """Initialize with configuration parameters."""
        self.logger = get_logger(__name__)
        self.price_precision = price_precision
        
    def calculate_volume_profile(self, session_df):
        """Calculate volume profile for a trading session."""
        # Calculate price range
        min_price = min(session_df['low'].min(), session_df['open'].min(), session_df['close'].min())
        max_price = max(session_df['high'].max(), session_df['open'].max(), session_df['close'].max())

        # Round to nearest tick
        min_price = np.floor(min_price / self.price_precision) * self.price_precision
        max_price = np.ceil(max_price / self.price_precision) * self.price_precision

        # Create price bins
        price_bins = np.arange(min_price, max_price + self.price_precision, self.price_precision)

        # Initialize volume profile
        volume_profile = pd.DataFrame({
            'price_level': price_bins,
            'volume': 0.0
        })

        # Distribute volume across price levels
        for _, row in session_df.iterrows():
            # Find range for this bar
            bar_min = min(row['low'], row['open'], row['close'])
            bar_max = max(row['high'], row['open'], row['close'])

            # Get levels within this bar's range
            mask = (volume_profile['price_level'] >= bar_min) & (volume_profile['price_level'] <= bar_max)
            levels_in_range = volume_profile.loc[mask]
            
            if not levels_in_range.empty:
                # Weight by proximity to OHLC
                weights = np.ones(len(levels_in_range))
                price_points = levels_in_range['price_level'].values
                
                # Add weights based on proximity to key prices
                for price in [row['open'], row['high'], row['low'], row['close']]:
                    proximity_weight = 1.0 / (1.0 + np.abs(price_points - price))
                    weights += proximity_weight
                    
                # Normalize and distribute volume
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                weighted_volume = weights * row['volume']
                
                # Update volume profile
                for i, idx in enumerate(levels_in_range.index):
                    volume_profile.loc[idx, 'volume'] += weighted_volume[i]

        # Add smoothed volume
        volume_profile['volume_smooth'] = volume_profile['volume'].rolling(window=3, center=True).mean()
        volume_profile['volume_smooth'] = volume_profile['volume_smooth'].fillna(volume_profile['volume'])
        
        return volume_profile
        
    def find_vpoc(self, volume_profile):
        """Find Volume Point of Control - the price level with highest volume."""
        vol_col = 'volume_smooth' if 'volume_smooth' in volume_profile.columns else 'volume'
        vpoc_idx = volume_profile[vol_col].argmax()
        return volume_profile.iloc[vpoc_idx]['price_level']
        
    def find_value_area(self, volume_profile, pct=0.70):
        """Find Value Area - price range containing specified % of volume."""
        vol_col = 'volume_smooth' if 'volume_smooth' in volume_profile.columns else 'volume'
        
        # Start with VPOC
        vpoc_idx = volume_profile[vol_col].argmax()
        included_indices = [vpoc_idx]
        current_volume = volume_profile.iloc[vpoc_idx][vol_col]
        total_volume = volume_profile[vol_col].sum()
        target_volume = total_volume * pct
        
        # Walk outward from VPOC
        upper_idx = vpoc_idx
        lower_idx = vpoc_idx
        
        while current_volume < target_volume and (upper_idx < len(volume_profile) - 1 or lower_idx > 0):
            # Check volumes above and below current range
            upper_vol = volume_profile.iloc[upper_idx + 1][vol_col] if upper_idx < len(volume_profile) - 1 else 0
            lower_vol = volume_profile.iloc[lower_idx - 1][vol_col] if lower_idx > 0 else 0
            
            # Add the level with more volume
            if upper_vol >= lower_vol and upper_idx < len(volume_profile) - 1:
                upper_idx += 1
                included_indices.append(upper_idx)
                current_volume += upper_vol
            elif lower_idx > 0:
                lower_idx -= 1
                included_indices.append(lower_idx)
                current_volume += lower_vol
            else:
                break
                
        # Calculate value area bounds
        value_area_levels = volume_profile.iloc[included_indices]['price_level'].values
        val = min(value_area_levels)
        vah = max(value_area_levels)
        va_volume_pct = (current_volume / total_volume) * 100
        
        return val, vah, va_volume_pct


class SignalGenerator:
    """
    Generates trading signals based on VPOC analysis.
    """
    
    def __init__(self, settings=None):
        """Initialize signal generator with settings."""
        self.logger = get_logger(__name__)
        
        # Initialize with settings or defaults
        if settings is None:
            try:
                from src.config.settings import settings
            except ImportError:
                settings = None
                
        # Output directory
        self.output_dir = getattr(settings, 'STRATEGY_DIR', 
                                 os.path.join(os.getcwd(), 'STRATEGY'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create volume profile analyzer
        price_precision = getattr(settings, 'PRICE_PRECISION', 0.25)
        self.analyzer = VolumeProfileAnalyzer(price_precision)
        
    def analyze_session(self, df, date, contract=None):
        """
        Analyze a single trading session.
        
        Args:
            df: DataFrame with price data
            date: Session date
            contract: Contract identifier
            
        Returns:
            Dict with analysis results
        """
        self.logger.info(f"Analyzing session: {date} {contract or ''}")
        
        # Filter data for this session
        if 'date' in df.columns:
            session_df = df[df['date'] == date]
        elif 'session_date' in df.columns:
            session_df = df[df['session_date'] == date]
        else:
            self.logger.error("No date column found in DataFrame")
            return None
            
        # Apply contract filter if provided
        if contract and 'contract' in session_df.columns:
            session_df = session_df[session_df['contract'] == contract]
            
        if len(session_df) == 0:
            self.logger.warning(f"No data found for session {date}")
            return None
            
        # Calculate volume profile
        volume_profile = self.analyzer.calculate_volume_profile(session_df)
        
        # Find VPOC and value area
        vpoc = self.analyzer.find_vpoc(volume_profile)
        val, vah, va_volume_pct = self.analyzer.find_value_area(volume_profile)
        
        # Create result dictionary
        result = {
            'date': date,
            'contract': contract,
            'vpoc': vpoc,
            'value_area_low': val,
            'value_area_high': vah,
            'value_area_width': vah - val,
            'total_volume': session_df['volume'].sum(),
            'session_high': session_df['high'].max(),
            'session_low': session_df['low'].min(),
            'session_open': session_df['open'].iloc[0],
            'session_close': session_df['close'].iloc[-1]
        }
        
        return result
        
    def generate_vpoc_signals(self, df, trend_validation=None):
        """
        Generate trading signals based on VPOC analysis.
        
        Args:
            df: DataFrame with VPOC data
            trend_validation: Optional trend validation data
            
        Returns:
            DataFrame with trading signals
        """
        self.logger.info("Generating VPOC trading signals")
        
        # Initialize signals list
        signals = []
        
        # Generate signals based on VPOC data
        for i in range(10, len(df)):
            current_row = df.iloc[i]
            current_date = current_row['date']
            current_vpoc = current_row['vpoc']
            current_val = current_row['value_area_low']
            current_vah = current_row['value_area_high']
            
            # Skip if missing data
            if pd.isna(current_vpoc) or pd.isna(current_val) or pd.isna(current_vah):
                continue
                
            # Apply signal generation rules (simplified here)
            # Long Signal
            long_signal = {
                'date': current_date,
                'signal': 'BUY',
                'price': current_val,
                'stop_loss': current_val - (current_vah - current_val) * 0.1,
                'target': current_vah,
                'position_size': 1.0,
                'confidence': 70,
                'reason': "VPOC Buy Signal"
            }
            
            # Short Signal
            short_signal = {
                'date': current_date,
                'signal': 'SELL',
                'price': current_vah,
                'stop_loss': current_vah + (current_vah - current_val) * 0.1,
                'target': current_val,
                'position_size': 1.0,
                'confidence': 70,
                'reason': "VPOC Sell Signal"
            }
            
            signals.extend([long_signal, short_signal])
            
        # Convert to DataFrame and save
        signals_df = pd.DataFrame(signals)
        
        if not signals_df.empty:
            # Save signals if we have any
            signals_file = os.path.join(self.output_dir, 'trading_signals.csv')
            signals_df.to_csv(signals_file, index=False)
            self.logger.info(f"Saved {len(signals_df)} trading signals to {signals_file}")
            
        return signals_df