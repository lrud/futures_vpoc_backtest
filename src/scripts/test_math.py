#!/usr/bin/env python
"""
Test script for the mathematical analysis utilities.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.data import FuturesDataManager
from src.core.signals import SignalGenerator
from src.analysis.math_utils import VPOCMathAnalyzer

def main():
    """Test mathematical analysis functionality"""
    print("Testing mathematical analysis utilities...")
    
    # Load data
    data_manager = FuturesDataManager()
    futures_data = data_manager.load_futures_data()
    
    if futures_data is None or futures_data.empty:
        print("No data available for testing")
        return
        
    # Filter to RTH sessions
    rth_data = data_manager.filter_by_session(futures_data, 'RTH')
    
    # Create signal generator
    signal_gen = SignalGenerator()
    
    # Generate VPOC data for last 30 sessions
    print("\nGenerating VPOC data for trend analysis...")
    dates = sorted(rth_data['date'].unique())[-30:]
    
    vpoc_data = []
    for date in dates:
        result = signal_gen.analyze_session(rth_data, date)
        if result:
            vpoc_data.append(result)
    
    # Create DataFrame
    vpoc_df = pd.DataFrame(vpoc_data)
    print(f"Created VPOC data for {len(vpoc_df)} sessions")
    
    # Create math analyzer
    math_analyzer = VPOCMathAnalyzer()
    
    # Test trend validation
    print("\nValidating VPOC trend...")
    trend_validation = math_analyzer.validate_vpoc_trend(
        vpoc_df['vpoc'].tolist(),
        vpoc_df['date'].tolist()
    )
    
    print(f"Trend direction: {trend_validation['direction']}")
    print(f"Trend validity: {trend_validation['valid_trend']}")
    print(f"Confidence: {trend_validation['confidence']:.2f}%")
    print(f"R-squared: {trend_validation['r_squared']:.3f}")
    print(f"Slope: {trend_validation['slope']:.3f}")
    
    # Test momentum analysis
    print("\nPerforming momentum analysis...")
    momentum_results = math_analyzer.momentum_analysis(vpoc_df)
    
    if not momentum_results.empty:
        print(f"Generated momentum data for {len(momentum_results)} windows")
        print(f"Average momentum: {momentum_results['window_momentum'].mean():.3f}")
        print(f"Average confidence: {momentum_results['window_confidence'].mean():.3f}")
    else:
        print("No momentum results generated")
    
    # Test Bayesian probability estimation
    print("\nCalculating Bayesian probabilities...")
    bayesian_results = math_analyzer.bayesian_probability_estimation(vpoc_df)
    
    print(f"Probability up: {bayesian_results['probability_up']:.4f}")
    print(f"Probability down: {bayesian_results['probability_down']:.4f}")
    
    # Save comprehensive analysis
    print("\nSaving comprehensive analysis...")
    analysis_results = {
        'trend_validation': trend_validation,
        'momentum': momentum_results,
        'bayesian': bayesian_results
    }
    
    output_path = math_analyzer.save_analysis(analysis_results)
    print(f"Analysis saved to: {output_path}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()