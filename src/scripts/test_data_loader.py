#!/usr/bin/env python
"""
Test script for the refactored FuturesDataManager class.
"""

import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.data import FuturesDataManager
from src.config.settings import settings

def main():
    """Test data loading functionality"""
    print("Testing FuturesDataManager...")

    data_manager = FuturesDataManager()
    futures_data = data_manager.load_futures_data()

    if futures_data is not None and not futures_data.empty:
        print("\nTesting filtering methods...")

        rth_data = data_manager.filter_by_session(futures_data, 'RTH')
        print(f"RTH data: {len(rth_data)} records")

        if not futures_data.empty and 'contract' in futures_data.columns:
            first_contract = futures_data['contract'].iloc[0]
            prefix = first_contract[:2]
            contract_data = data_manager.filter_by_contract(futures_data, prefix)
            print(f"{prefix} contract data: {len(contract_data)} records")

        if 'date' in futures_data.columns:
            min_date = futures_data['date'].min()
            max_date = futures_data['date'].max()
            mid_point = min_date + (max_date - min_date) // 2

            date_filtered = data_manager.filter_by_date_range(
                futures_data, start_date=mid_point
            )
            print(f"Data after {mid_point}: {len(date_filtered)} records")

    print("\nTest completed.")

def test_vix_data_loading():
    """Test loading standalone VIX data from VIX_History.csv."""
    data_manager = FuturesDataManager()
    vix_data = data_manager.load_vix_data()

    assert not vix_data.empty, "VIX data should not be empty"
    assert {'date', 'vix_close', 'vix_1d_change'}.issubset(vix_data.columns), \
           "VIX data missing required columns"
    assert vix_data['date'].is_monotonic_increasing, "VIX dates should be sorted ascending"
    assert vix_data['vix_close'].min() > 0, "VIX close prices should be positive"

if __name__ == "__main__":
    main()