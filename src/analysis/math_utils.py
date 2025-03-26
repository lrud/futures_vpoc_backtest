"""
Mathematical analysis utilities for VPOC strategy.
Provides statistical validation and probability estimation.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from datetime import datetime

from src.utils.logging import get_logger


class VPOCMathAnalyzer:
    """
    Mathematical analysis for Volume Point of Control (VPOC) data.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the math analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.logger = get_logger(__name__)
        
        # Get output directory from settings or use default
        if output_dir is None:
            try:
                from src.config.settings import settings
                self.output_dir = getattr(settings, 'MATH_OUTPUT_DIR', 
                                         os.path.join(os.getcwd(), 'MATH_ANALYSIS'))
            except (ImportError, AttributeError):
                self.output_dir = os.path.join(os.getcwd(), 'MATH_ANALYSIS')
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"Initialized VPOCMathAnalyzer with output_dir: {self.output_dir}")
    
    def validate_vpoc_trend(self, vpocs: List[float], dates: List[datetime], 
                          lookback: int = 20) -> Dict[str, any]:
        """
        Validate VPOC trend using statistical methods.
        
        Args:
            vpocs: List of VPOC prices
            dates: List of corresponding dates
            lookback: Number of days to analyze for trend
            
        Returns:
            Dictionary with trend validation results
        """
        self.logger.info(f"Validating VPOC trend with lookback={lookback}")
        
        # Check for sufficient data points
        if len(vpocs) < 5:
            self.logger.warning("Insufficient data for trend validation (minimum 5 points required)")
            return {
                'valid_trend': False,
                'p_value': None,
                'direction': None,
                'consecutive_count': 0,
                'confidence': 0,
                'slope': None,
                'r_squared': None
            }
        
        # Use as many data points as available, up to lookback limit
        recent_vpocs = vpocs[-min(lookback, len(vpocs)):]
        recent_dates = dates[-min(lookback, len(vpocs)):]
        
        # Calculate price differences and determine trend direction
        diffs = [recent_vpocs[i] - recent_vpocs[i-1] for i in range(1, len(recent_vpocs))]
        pos_moves = sum(1 for d in diffs if d > 0)
        neg_moves = sum(1 for d in diffs if d < 0)
        zero_moves = sum(1 for d in diffs if d == 0)
        
        # Determine trend direction based on majority of moves
        direction = 'up' if pos_moves > neg_moves else 'down' if neg_moves > pos_moves else 'neutral'
        
        # Find consecutive moves (streaks)
        current_up_streak = 0
        current_down_streak = 0
        max_up_streak = 0
        max_down_streak = 0
        
        for diff in diffs:
            if diff > 0:
                current_up_streak += 1
                current_down_streak = 0
                max_up_streak = max(max_up_streak, current_up_streak)
            elif diff < 0:
                current_down_streak += 1
                current_up_streak = 0
                max_down_streak = max(max_down_streak, current_down_streak)
            else:  # diff == 0 (No change)
                # Consider no change as continuing the current streak
                pass
        
        # Linear regression for trend strength
        x = np.arange(len(recent_vpocs))
        y = np.array(recent_vpocs)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Run test for randomness
        median = np.median(recent_vpocs)
        runs = [1 if v > median else 0 for v in recent_vpocs]
        runs_count = 1  # Initialize run count to 1
        
        for i in range(1, len(runs)):
            if runs[i] != runs[i-1]:  # Check for change in run sequence
                runs_count += 1
        
        n1 = sum(runs)
        n2 = len(runs) - n1
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                         ((n1 + n2)**2 * (n1 + n2 - 1)))
        
        run_test_p = 2 * (1 - stats.norm.cdf(abs((runs_count - expected_runs) / std_runs))) if std_runs > 0 else 1.0
        
        # Relaxed criteria for trend validation
        is_valid_trend = (
            (p_value < 0.1) and          # More lenient p-value threshold
            (r_squared > 0.2) and        # Lower R-squared requirement
            (                            # Reduced consecutive moves requirement
                (direction == 'up' and max_up_streak >= 2) or
                (direction == 'down' and max_down_streak >= 2)
            )
        )
        
        # Alternative validation for strong consecutive moves
        if not is_valid_trend and ((max_up_streak >= 3) or (max_down_streak >= 3)):
            is_valid_trend = True
        
        # If trend is not valid but we have directional bias, still set direction
        if not is_valid_trend and direction == 'neutral':
            direction = 'up' if slope > 0 else 'down'
        
        # Calculate confidence score
        if is_valid_trend:
            trend_strength = min(r_squared * 100, 100)
            statistical_significance = (1 - min(p_value, 0.5) * 2) * 100
            streak_factor = (max(max_up_streak, max_down_streak) / 5) * 100
            confidence = (trend_strength + statistical_significance + streak_factor) / 3
        else:
            # Assign some confidence even without full validation
            consecutive_factor = max(max_up_streak, max_down_streak) * 10
            confidence = min(consecutive_factor, 50)
        
        # Create result dictionary
        result = {
            'valid_trend': is_valid_trend,
            'p_value': p_value,
            'run_test_p': run_test_p,
            'direction': direction,
            'consecutive_up': max_up_streak,
            'consecutive_down': max_down_streak,
            'confidence': confidence,
            'slope': slope,
            'r_squared': r_squared
        }
        
        self.logger.info(f"Trend validation result: {direction} trend, " +
                       f"confidence={confidence:.1f}%, valid={is_valid_trend}")
        
        return result
    
    def momentum_analysis(self, vpoc_data: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Calculate rolling window momentum for VPOC movements.
        
        Args:
            vpoc_data: DataFrame with VPOC data
            window: Size of rolling window
            
        Returns:
            DataFrame with momentum metrics
        """
        self.logger.info(f"Performing momentum analysis with window={window}")
        
        if vpoc_data.empty:
            self.logger.warning("Empty DataFrame provided for momentum analysis")
            return pd.DataFrame()
        
        if 'vpoc' not in vpoc_data.columns:
            self.logger.error("Required column 'vpoc' not found in DataFrame")
            return pd.DataFrame()
        
        momentums = []
        
        for i in range(len(vpoc_data) - window):
            try:
                subset = vpoc_data.iloc[i:i+window]
                x = np.arange(len(subset))
                y = subset['vpoc'].values
                
                # Data validation
                if len(set(y)) < 2:
                    continue
                    
                slope, _, r_value, p_value, _ = stats.linregress(x, y)
                
                # Normalize momentum
                std_dev = np.std(y)
                normalized_momentum = slope / std_dev if std_dev != 0 else 0
                
                # Store result
                momentums.append({
                    'date': subset.index[-1] if isinstance(subset.index, pd.DatetimeIndex) else subset.iloc[-1]['date'],
                    'window_momentum': normalized_momentum,
                    'window_confidence': r_value**2,
                    'window_significance': 1 - p_value
                })
            except Exception as e:
                self.logger.error(f"Error in momentum calculation: {str(e)}")
                continue
        
        return pd.DataFrame(momentums)
    
    def bayesian_probability_estimation(self, vpoc_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Bayesian probability of VPOC movement direction.
        
        Args:
            vpoc_data: DataFrame with VPOC data
            
        Returns:
            Dictionary with probability estimates
        """
        self.logger.info("Calculating Bayesian probability estimates")
        
        if vpoc_data.empty or 'vpoc' not in vpoc_data.columns:
            self.logger.warning("Invalid data for Bayesian analysis")
            return {'probability_up': 0.5, 'probability_down': 0.5}
        
        try:
            # Calculate price changes with exponential weighting for recency
            price_changes = vpoc_data['vpoc'].diff().dropna()
            
            if len(price_changes) == 0:
                return {'probability_up': 0.5, 'probability_down': 0.5}
            
            # Create exponential weights (more recent = higher weight)
            weights = np.exp(np.linspace(-1, 0, len(price_changes)))
            
            # Weight recent moves more heavily
            positive_mask = price_changes > 0
            negative_mask = price_changes < 0
            
            positive_moves = price_changes[positive_mask]
            negative_moves = price_changes[negative_mask]
            
            # Align weights with the masks
            positive_weights = weights[-len(positive_moves):] if len(positive_moves) > 0 else []
            negative_weights = weights[-len(negative_moves):] if len(negative_moves) > 0 else []
            
            # Calculate weighted probabilities
            weighted_positive = np.sum(np.abs(positive_moves) * positive_weights) if len(positive_moves) > 0 else 0
            weighted_negative = np.sum(np.abs(negative_moves) * negative_weights) if len(negative_moves) > 0 else 0
            
            total_weighted = weighted_positive + weighted_negative
            
            prob_up = weighted_positive / total_weighted if total_weighted > 0 else 0.5
            
            return {
                'probability_up': prob_up,
                'probability_down': 1 - prob_up
            }
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian analysis: {str(e)}")
            return {
                'probability_up': 0.5,
                'probability_down': 0.5
            }
    
    def save_analysis(self, analysis_results: Dict[str, any], filename: str = None) -> str:
        """
        Save analysis results to a CSV file.
        
        Args:
            analysis_results: Dictionary of analysis results
            filename: Custom filename for output
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'vpoc_analysis_{timestamp}.csv'
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # Flatten dictionary for CSV output
            flat_results = {}
            for key, value in analysis_results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_results[f'{key}_{subkey}'] = subvalue
                elif isinstance(value, pd.DataFrame):
                    # Save DataFrame separately
                    df_path = output_path.replace('.csv', f'_{key}.csv')
                    value.to_csv(df_path, index=False)
                    self.logger.info(f"Saved {key} data to {df_path}")
                else:
                    flat_results[key] = value
            
            # Save flattened results to CSV
            pd.DataFrame([flat_results]).to_csv(output_path, index=False)
            self.logger.info(f"Analysis saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {str(e)}")
            return ""