import numpy as np
import pandas as pd
import scipy.stats as stats
import os
from typing import Dict, Any, List
from datetime import datetime

# Configuration
BASE_DIR = '/home/lr/Documents/FUTURUES_PROJECT/futures_vpoc_backtest/'
MATH_OUTPUT_DIR = os.path.join(BASE_DIR, 'MATH_ANALYSIS')

# Ensure output directory exists
os.makedirs(MATH_OUTPUT_DIR, exist_ok=True)

class VPOCMathAnalysis:
    """
    Comprehensive mathematical analysis for Volume Point of Control (VPOC) data
    """
    
    def __init__(self, vpoc_data: pd.DataFrame):
        """
        Initialize analysis with VPOC data
        
        Parameters:
        -----------
        vpoc_data : pd.DataFrame
            DataFrame containing VPOC data with 'date' and 'vpoc' columns
        """
        self.validate_data(vpoc_data)
        self.vpoc_data = vpoc_data.sort_values('date')
        self.vpoc_data.set_index('date', inplace=True)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data structure and content
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data to validate
        
        Returns:
        --------
        bool : True if validation passes
        """
        if data.empty:
            raise ValueError("Empty VPOC data provided")
        
        required_columns = ['date', 'vpoc']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return True
    
    def linear_regression_analysis(self) -> Dict[str, Any]:
        """
        Perform linear regression analysis on VPOC data
        
        Returns:
        --------
        Dict containing regression insights
        """
        try:
            x = np.arange(len(self.vpoc_data))
            y = self.vpoc_data['vpoc'].values
            
            # Add data validation
            if len(set(y)) < 2:
                raise ValueError("Insufficient unique values for regression")
                
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return {
                'trend_slope': slope,
                'trend_intercept': intercept,
                'correlation_coefficient': r_value,
                'p_value': p_value,
                'r_squared': r_value**2,
                'standard_error': std_err
            }
        except Exception as e:
            print(f"Error in linear regression analysis: {e}")
            return {
                'trend_slope': 0,
                'trend_intercept': 0,
                'correlation_coefficient': 0,
                'p_value': 1,
                'r_squared': 0,
                'standard_error': 0
            }
    
    def momentum_analysis(self, window: int = 10) -> pd.DataFrame:
        """
        Calculate rolling window momentum with improved normalization
        
        Parameters:
        -----------
        window : int, optional
            Size of the rolling window for momentum calculation
        
        Returns:
        --------
        DataFrame with momentum metrics
        """
        momentums = []
        
        for i in range(len(self.vpoc_data) - window):
            try:
                subset = self.vpoc_data.iloc[i:i+window]
                x = np.arange(len(subset))
                y = subset['vpoc'].values
                
                # Data validation
                if len(set(y)) < 2:
                    continue
                    
                slope, _, r_value, p_value, _ = stats.linregress(x, y)
                
                # Normalize momentum
                std_dev = np.std(y)
                normalized_momentum = slope / std_dev if std_dev != 0 else 0
                
                momentums.append({
                    'date': subset.index[-1],
                    'window_momentum': normalized_momentum,
                    'window_confidence': r_value**2,
                    'window_significance': 1 - p_value
                })
            except Exception as e:
                print(f"Error calculating momentum for window ending {subset.index[-1]}: {e}")
                continue
        
        return pd.DataFrame(momentums)
    
    def volatility_analysis(self, windows: List[int] = [10, 20, 50]) -> Dict[str, float]:
        """
        Calculate volatility across different windows with simplified output
        
        Parameters:
        -----------
        windows : List[int], optional
            List of window sizes for volatility calculation
        
        Returns:
        --------
        Dictionary of volatility metrics
        """
        try:
            volatility_metrics = {}
            
            for window in windows:
                rolling_std = self.vpoc_data['vpoc'].rolling(window=window).std()
                volatility_metrics[f'volatility_{window}'] = rolling_std.iloc[-1]
            
            return volatility_metrics
            
        except Exception as e:
            print(f"Error in volatility analysis: {e}")
            return {f'volatility_{window}': 0 for window in windows}
    
    def bayesian_probability_estimation(self) -> Dict[str, float]:
        """
        Calculate Bayesian probability with exponential weighting for recency bias
        
        Returns:
        --------
        Dictionary of movement probabilities
        """
        try:
            # Calculate price changes with exponential weighting
            price_changes = self.vpoc_data['vpoc'].diff()
            weights = np.exp(np.linspace(-1, 0, len(price_changes)))
            
            # Weight recent moves more heavily
            positive_moves = price_changes[price_changes > 0] * weights[price_changes > 0]
            negative_moves = price_changes[price_changes < 0] * weights[price_changes < 0]
            
            # Calculate weighted probabilities
            total_weighted_moves = np.sum(np.abs(positive_moves)) + np.sum(np.abs(negative_moves))
            prob_up = np.sum(np.abs(positive_moves)) / total_weighted_moves if total_weighted_moves > 0 else 0.5
            
            return {
                'probability_up': prob_up,
                'probability_down': 1 - prob_up
            }
            
        except Exception as e:
            print(f"Error in Bayesian analysis: {e}")
            return {
                'probability_up': 0.5,
                'probability_down': 0.5
            }
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive mathematical analysis
        
        Returns:
        --------
        Dictionary of comprehensive analysis results
        """
        try:
            analysis_results = {
                'linear_regression': self.linear_regression_analysis(),
                'momentum': self.momentum_analysis(),
                'volatility': self.volatility_analysis(),
                'bayesian_probabilities': self.bayesian_probability_estimation()
            }
            
            return analysis_results
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return {}
    
    def save_analysis(self, analysis_results: Dict[str, Any], filename: str = None):
        """
        Save analysis results to CSV files
        
        Parameters:
        -----------
        analysis_results : Dict
            Comprehensive analysis results
        filename : str, optional
            Custom filename for output
        """
        try:
            if not filename:
                filename = f'vpoc_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            
            output_path = os.path.join(MATH_OUTPUT_DIR, filename)
            
            # Flatten dictionary for CSV output
            flat_results = {}
            for key, value in analysis_results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_results[f'{key}_{subkey}'] = subvalue
                elif isinstance(value, pd.DataFrame):
                    value.to_csv(output_path.replace('.csv', f'_{key}.csv'), index=False)
                else:
                    flat_results[key] = value
            
            # Save flattened results
            pd.DataFrame([flat_results]).to_csv(output_path, index=False)
            print(f"Analysis saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving analysis results: {e}")

def main():
    """
    Main function to demonstrate VPOC mathematical analysis
    """
    try:
        # Load VPOC data
        vpoc_data_path = os.path.join(BASE_DIR, 'RESULTS', 'RTH_vpoc_data.csv')
        vpoc_data = pd.read_csv(vpoc_data_path, parse_dates=['date'])
        
        # Initialize and run analysis
        math_analyzer = VPOCMathAnalysis(vpoc_data)
        analysis_results = math_analyzer.comprehensive_analysis()
        
        # Save analysis results
        math_analyzer.save_analysis(analysis_results)
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()