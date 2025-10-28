"""
Signal generation functionality for trading strategies.
Handles VPOC-based and ML-enhanced signal creation.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.utils.logging import get_logger
from src.config.settings import settings
from src.core.vpoc import VolumeProfileAnalyzer
from src.analysis.math_utils import validate_trend, calculate_bayesian_probabilities

class SignalGenerator:
    """
    Generates trading signals based on various strategies.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize signal generator.
        
        Parameters:
        -----------
        output_dir: Optional[str]
            Directory to save signal output
        """
        self.logger = get_logger(__name__)
        self.output_dir = output_dir or settings.STRATEGY_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.vpoc_analyzer = VolumeProfileAnalyzer(price_precision=settings.PRICE_PRECISION)
        self.logger.info(f"Initialized SignalGenerator with output to {self.output_dir}")
    
    def generate_vpoc_signals(self, df: pd.DataFrame, 
                             trend_validation: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate trading signals based on VPOC analysis.
        
        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame with OHLCV data and VPOC analysis results
        trend_validation: Optional[Dict]
            Optional trend validation results
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals
        """
        self.logger.info(f"Generating VPOC signals from {len(df)} sessions")
        
        signals = []
        
        for i in range(1, len(df)):
            # Get current and previous session data
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Basic signal criteria - VPOC movement
            vpoc_up = current['vpoc'] > prev['vpoc']
            vpoc_down = current['vpoc'] < prev['vpoc']
            
            # Skip if no clear direction
            if not (vpoc_up or vpoc_down):
                continue
                
            # Apply trend filter if provided
            if trend_validation:
                if vpoc_up and trend_validation['direction'] != 'up':
                    continue
                if vpoc_down and trend_validation['direction'] != 'down':
                    continue
            
            # Extract current session info
            current_date = current['date']
            current_vpoc = current['vpoc']
            current_val = current['value_area_low']
            current_vah = current['value_area_high']
            
            # Calculate confidence score
            confidence = 70  # Default confidence
            if trend_validation:
                confidence = trend_validation.get('confidence', 70)
            
            # Long Signal
            long_signal = {
                'date': current_date,
                'signal': 'BUY',
                'price': current_val,
                'stop_loss': current_val - (current_vah - current_val) * 0.1,
                'target': current_vah,
                'position_size': 1.0,
                'confidence': confidence,
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
                'confidence': confidence,
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
    
    def generate_enhanced_signals(self, df: pd.DataFrame, 
                                 vpoc_data: pd.DataFrame,
                                 linear_regression_params: Dict,
                                 lookback_days: int = 10,
                                 min_bayesian_up_prob: float = 0.52) -> pd.DataFrame:
        """
        Generate enhanced trading signals with mathematical validation.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Main DataFrame with OHLCV data
        vpoc_data: pd.DataFrame
            DataFrame with VPOC analysis results
        linear_regression_params: Dict
            Linear regression parameters from math analysis
        lookback_days: int
            Number of days to look back for pattern validation
        min_bayesian_up_prob: float
            Minimum Bayesian probability for upward moves
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with enhanced trading signals
        """
        self.logger.info(f"Generating enhanced signals with {lookback_days} day lookback")
        
        # Unpack linear regression parameters
        trend_slope = linear_regression_params.get('trend_slope', 0)
        trend_r_squared = linear_regression_params.get('r_squared', 0)
        
        enhanced_signals = []
        
        # Get Bayesian probability
        if 'vpoc' in vpoc_data.columns:
            vpocs = vpoc_data['vpoc'].tolist()
            bayesian_prob = calculate_bayesian_probabilities(vpocs)
            bayesian_prob_up = bayesian_prob['probability_up']
        else:
            bayesian_prob_up = 0.5305  # Default value from math analysis
        
        # Process each day with enhanced validation
        for i in range(lookback_days, len(vpoc_data)):
            current_row = vpoc_data.iloc[i]
            current_date = current_row['date']
            
            # Get volatility metrics for the session
            volatility_metrics = {
                'volatility_10': current_row.get('volatility_10', 71.97),
                'volatility_20': current_row.get('volatility_20', 57.73),
                'volatility_50': current_row.get('volatility_50', 76.86)
            }
            
            # Calculate signal confidence score
            signal_confidence = self._calculate_signal_confidence(
                linear_regression_params, bayesian_prob_up, volatility_metrics)
            
            # Dynamic signal conditions based on math analysis
            signal_conditions = (
                trend_slope > 0 and                    # Trend is positive
                trend_r_squared > 0.6 and              # Strong R-squared from math
                (
                    (volatility_metrics['volatility_10'] < volatility_metrics['volatility_50']) or  # Lower short-term volatility
                    (bayesian_prob_up > min_bayesian_up_prob)                                       # Strong Bayesian probability
                ) and
                signal_confidence > 60                  # Overall confidence above 60%
            )
            
            if signal_conditions:
                # Extract price levels
                current_vpoc = current_row['vpoc']
                current_val = current_row['value_area_low']
                current_vah = current_row['value_area_high']
                
                # Calculate position size based on volatility
                position_size = self._calculate_position_size(volatility_metrics)
                
                # Calculate adaptive stop loss
                long_stop = self._calculate_stop_loss(current_val, volatility_metrics)
                short_stop = self._calculate_stop_loss(current_vah, volatility_metrics)
                
                # Long Signal Generation
                long_signal = {
                    'date': current_date,
                    'signal': 'BUY',
                    'price': current_val,
                    'stop_loss': long_stop,
                    'target': current_vah,
                    'position_size': position_size,
                    'signal_type': 'Enhanced-VPOC',
                    'confidence': signal_confidence,
                    'reason': f"Trend:{trend_slope:.2f}, R²:{trend_r_squared:.2f}, Conf:{signal_confidence:.0f}%"
                }
                
                # Short Signal Generation
                short_signal = {
                    'date': current_date,
                    'signal': 'SELL',
                    'price': current_vah,
                    'stop_loss': short_stop,
                    'target': current_val,
                    'position_size': position_size,
                    'signal_type': 'Enhanced-VPOC',
                    'confidence': signal_confidence,
                    'reason': f"Trend:{trend_slope:.2f}, R²:{trend_r_squared:.2f}, Conf:{signal_confidence:.0f}%"
                }
                
                enhanced_signals.extend([long_signal, short_signal])
                self.logger.info(f"Generated signals for {current_date} - Confidence: {signal_confidence:.0f}%")
            else:
                self.logger.debug(f"No signals for {current_date} - conditions not met")
        
        # Convert to DataFrame
        enhanced_signals_df = pd.DataFrame(enhanced_signals)
        if len(enhanced_signals_df) > 0:
            enhanced_signals_df = enhanced_signals_df.sort_values('date')
            
            # Save results
            signals_file = os.path.join(self.output_dir, 'enhanced_trading_signals.csv')
            enhanced_signals_df.to_csv(signals_file, index=False)
            self.logger.info(f"Saved {len(enhanced_signals_df)} enhanced signals to {signals_file}")
        
        return enhanced_signals_df
    
    def generate_ml_signals(self, features_df: pd.DataFrame, model, 
                           prediction_threshold: float = 0.5,
                           confidence_threshold: float = 70) -> pd.DataFrame:
        """
        Generate machine learning-based trading signals.
        
        Parameters:
        -----------
        features_df: pd.DataFrame
            DataFrame with features
        model: torch.nn.Module
            Trained ML model
        prediction_threshold: float
            Threshold for converting predictions to signals
        confidence_threshold: float
            Minimum confidence required for signals
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with ML-based signals
        """
        import torch
        from sklearn.preprocessing import StandardScaler
        
        self.logger.info(f"Generating ML signals from {len(features_df)} sessions")
        
        # Extract feature columns from the actual DataFrame
        # Use actual column names instead of placeholder names from model
        non_feature_cols = {'date', 'signal_type', 'signal', 'confidence', 'contract', 'session'}
        feature_cols = [col for col in features_df.columns if col not in non_feature_cols]

        # Validate we have the right number of features
        if hasattr(model, 'feature_columns'):
            expected_features = len(model.feature_columns)
        else:
            # Infer expected features from model input dimension
            expected_features = features_df.shape[1] - len(non_feature_cols)

        if len(feature_cols) != expected_features:
            self.logger.warning(f"Feature count mismatch: expected {expected_features}, got {len(feature_cols)}")

        # Select and scale features
        X = features_df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Generate predictions - Ensure tensor is on the same device as the model
        target_device = next(model.parameters()).device # Get device from model parameters
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(target_device) # Move tensor to target device
        model.eval()
        with torch.no_grad():
            # Explicitly cast to float32 before converting to numpy
            predictions = model(X_tensor).squeeze().to(torch.float32).cpu().numpy()
        
        # Add predictions and confidence scores
        features_df['ml_prediction'] = predictions
        features_df['confidence'] = np.abs(predictions) * 100
        
        # Generate signals
        features_df['signal_type'] = np.where(
            predictions > prediction_threshold, 'BUY',
            np.where(predictions < -prediction_threshold, 'SELL', 'NEUTRAL')
        )
        
        # Filter signals by confidence
        signals_df = features_df[
            (features_df['signal_type'] != 'NEUTRAL') & 
            (features_df['confidence'] >= confidence_threshold)
        ].copy()
        
        # Set signal values for compatibility with backtest
        signals_df['signal'] = signals_df['signal_type']
        
        # This code works for both VIX and ES data
        signals_df['price'] = signals_df['vpoc']  # Use VPOC as entry price regardless of contract
        signals_df['stop_loss'] = np.where(
            signals_df['signal'] == 'BUY',
            signals_df['value_area_low'] * 0.99,  # Long stop below VA low
            signals_df['value_area_high'] * 1.01   # Short stop above VA high
        )
        
        # Add target based on risk:reward (2:1)
        signals_df['target'] = np.where(
            signals_df['signal'] == 'BUY',
            signals_df['price'] + 2 * (signals_df['price'] - signals_df['stop_loss']),  # Long target
            signals_df['price'] - 2 * (signals_df['stop_loss'] - signals_df['price'])   # Short target
        )
        
        # Position sizing and metadata
        signals_df['position_size'] = 1.0
        signals_df['reason'] = 'ML_' + signals_df['signal_type']
        
        # Save signal data for debugging
        if not signals_df.empty:
            signal_debug_file = os.path.join(self.output_dir, 'ml_signal_debug.csv')
            features_df.to_csv(signal_debug_file, index=False)
            self.logger.info(f"Detailed signal debug saved to: {signal_debug_file}")
            
            # Save final signals
            signals_file = os.path.join(self.output_dir, 'ml_trading_signals.csv')
            signals_df.to_csv(signals_file, index=False)
            self.logger.info(f"Saved {len(signals_df)} ML signals to {signals_file}")
        
        # Return signals without 'date' as it's added back in backtest_integration.py
        return signals_df[['signal', 'price', 'stop_loss', 'target', 
                         'position_size', 'confidence', 'reason']]
    
    def _calculate_position_size(self, volatility_metrics: Dict[str, float]) -> float:
        """
        Calculate position size based on volatility metrics.
        
        Parameters:
        -----------
        volatility_metrics: Dict[str, float]
            Dict with volatility measures
            
        Returns:
        --------
        float
            Position size multiplier (0.5-1.5)
        """
        # If volatility is very high, reduce position size
        vol_10d = volatility_metrics.get('volatility_10', 70)
        vol_50d = volatility_metrics.get('volatility_50', 70)
        
        # Calculate ratio of short-term to long-term volatility
        vol_ratio = vol_10d / max(vol_50d, 0.1) if vol_50d > 0 else 1.0
        
        # Inverse relationship: higher vol_ratio = lower position size
        if vol_ratio > 1.5:
            # Very high short-term volatility compared to long-term
            return 0.5
        elif vol_ratio > 1.2:
            # Moderately high short-term volatility
            return 0.75
        elif vol_ratio < 0.8:
            # Low short-term volatility - opportunity for larger position
            return 1.25
        elif vol_ratio < 0.5:
            # Very low short-term volatility
            return 1.5
        else:
            # Normal volatility conditions
            return 1.0
    
    def _calculate_stop_loss(self, price: float, volatility_metrics: Dict[str, float]) -> float:
        """
        Calculate adaptive stop loss based on volatility.
        
        Parameters:
        -----------
        price: float
            Entry price
        volatility_metrics: Dict[str, float]
            Dict with volatility measures
            
        Returns:
        --------
        float
            Stop loss price
        """
        # Use shorter-term volatility for stop calculation
        vol_10d = volatility_metrics.get('volatility_10', 70)
        
        # Convert percentage volatility to price units (approximation)
        vol_price = price * (vol_10d / 100)
        
        # Set stop at 0.5-1.5x the 10-day volatility away from price
        if vol_10d > 100:
            # Very high volatility - wider stop (1.5x)
            stop_distance = vol_price * 1.5
        elif vol_10d > 70:
            # High volatility - normal stop (1x)
            stop_distance = vol_price
        else:
            # Low volatility - tighter stop (0.5x)
            stop_distance = vol_price * 0.5
        
        return price - stop_distance  # For long positions (subtract for short)
    
    def _calculate_signal_confidence(self, 
                                   linear_regression_params: Dict,
                                   bayesian_prob: float,
                                   volatility_metrics: Dict[str, float]) -> float:
        """
        Calculate overall signal confidence score.
        
        Parameters:
        -----------
        linear_regression_params: Dict
            Linear regression parameters
        bayesian_prob: float
            Bayesian probability of upward move
        volatility_metrics: Dict[str, float]
            Dict with volatility measures
            
        Returns:
        --------
        float
            Confidence score (0-100)
        """
        # Extract metrics
        trend_r_squared = linear_regression_params.get('r_squared', 0)
        
        # Statistical significance component (0-40 points)
        stat_confidence = trend_r_squared * 40
        
        # Bayesian probability component (0-30 points)
        # Scale from [0.5, 1.0] to [0, 30]
        bayes_confidence = (bayesian_prob - 0.5) * 60 if bayesian_prob > 0.5 else 0
        
        # Volatility component (0-30 points)
        # Lower vol_ratio = higher confidence
        vol_10d = volatility_metrics.get('volatility_10', 70)
        vol_50d = volatility_metrics.get('volatility_50', 70)
        vol_ratio = vol_10d / max(vol_50d, 0.1)
        
        if vol_ratio < 0.8:
            vol_confidence = 30  # Low vol_ratio = high confidence
        elif vol_ratio < 1.0:
            vol_confidence = 20
        elif vol_ratio < 1.2:
            vol_confidence = 10
        else:
            vol_confidence = 0  # High vol_ratio = low confidence
        
        # Calculate total confidence
        total_confidence = stat_confidence + bayes_confidence + vol_confidence
        
        # Cap at 100
        return min(total_confidence, 100)
