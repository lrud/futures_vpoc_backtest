#!/usr/bin/env python
"""
Test script for the refactored signal generation.
"""

import os
import sys
import pandas as pd
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.data import FuturesDataManager
from src.core.signals import VolumeProfileAnalyzer, SignalGenerator
from src.config.settings import settings

def main():
    """Test signal generation functionality"""
    print("Testing signal generation...")
    
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
    
    # Test analyzing a single session
    # Get a sample date from the data
    sample_date = rth_data['date'].iloc[100]
    print(f"\nAnalyzing session for {sample_date}")
    
    # Analyze session
    session_analysis = signal_gen.analyze_session(rth_data, sample_date)
    if session_analysis:
        print(f"VPOC: {session_analysis['vpoc']}")
        print(f"Value Area: {session_analysis['value_area_low']} - {session_analysis['value_area_high']}")
    
    # Test analyzing multiple sessions
    print("\nAnalyzing 10 sessions...")
    
    # Group by date
    dates = rth_data['date'].unique()[:10]  # First 10 dates
    
    vpoc_data = []
    for date in dates:
        result = signal_gen.analyze_session(rth_data, date)
        if result:
            vpoc_data.append(result)
    
    # Convert to DataFrame
    vpoc_df = pd.DataFrame(vpoc_data)
    print(f"Analyzed {len(vpoc_df)} sessions")
    
    # Generate signals
    print("\nGenerating trading signals...")
    signals = signal_gen.generate_vpoc_signals(vpoc_df)
    
    print(f"Generated {len(signals)} trading signals")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()