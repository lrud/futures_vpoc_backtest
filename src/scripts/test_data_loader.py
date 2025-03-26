#!/usr/bin/env python
"""
Test script for the refactored FuturesDataManager class.
"""

import os
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.data import FuturesDataManager
from src.config.settings import settings

def main():
    """Test data loading functionality"""
    print("Testing FuturesDataManager...")
    
    # Initialize the data manager
    data_manager = FuturesDataManager()
    
    # Load futures data
    futures_data = data_manager.load_futures_data()
    
    # Test filtering methods if data was loaded
    if futures_data is not None and not futures_data.empty:
        print("\nTesting filtering methods...")
        
        # Filter by session
        rth_data = data_manager.filter_by_session(futures_data, 'RTH')
        print(f"RTH data: {len(rth_data)} records")
        
        # Try to filter by a contract (adjust prefix as needed)
        # Get the first contract prefix from the data
        if not futures_data.empty and 'contract' in futures_data.columns:
            first_contract = futures_data['contract'].iloc[0]
            prefix = first_contract[:2]  # e.g., "ES" from "ES_03_22"
            contract_data = data_manager.filter_by_contract(futures_data, prefix)
            print(f"{prefix} contract data: {len(contract_data)} records")
        
        # Test date range filtering
        # Get min and max dates from the data
        if 'date' in futures_data.columns:
            min_date = futures_data['date'].min()
            max_date = futures_data['date'].max()
            mid_point = min_date + (max_date - min_date) // 2
            
            # Filter to second half of the data
            date_filtered = data_manager.filter_by_date_range(
                futures_data, start_date=mid_point
            )
            print(f"Data after {mid_point}: {len(date_filtered)} records")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()