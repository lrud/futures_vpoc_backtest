#!/usr/bin/env python3
"""
Test script to verify GARCH calculation fix.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.ml.feature_engineering import FeatureEngineer
from src.utils.logging import get_logger

def test_garch_calculation():
    """Test the fixed GARCH calculation."""
    print("=== Testing GARCH Calculation Fix ===")

    # Create sample data
    np.random.seed(42)
    n_obs = 100

    # Generate synthetic return data
    returns = pd.Series(np.random.normal(0, 0.02, n_obs),
                       index=pd.date_range('2023-01-01', periods=n_obs))

    # Add some volatility clustering
    for i in range(20, 40):
        returns.iloc[i] *= 2  # Higher volatility period
    for i in range(60, 80):
        returns.iloc[i] *= 0.5  # Lower volatility period

    print(f"📊 Generated {len(returns)} synthetic returns")
    print(f"✅ Mean return: {returns.mean():.6f}")
    print(f"✅ Std deviation: {returns.std():.6f}")

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Test GARCH calculation
    print("\n🔬 Testing GARCH calculation...")
    garch_features = feature_engineer.calculate_garch_features(returns)

    if garch_features:
        print("✅ GARCH calculation successful!")
        print("\n📈 GARCH Parameters:")
        for key, value in garch_features.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        return True
    else:
        print("❌ GARCH calculation failed")
        return False

def test_log_transformation():
    """Test log transformation implementation."""
    print("\n=== Testing Log Transformation ===")

    # Create sample price data
    np.random.seed(42)
    n_obs = 100
    prices = 100 + np.cumsum(np.random.normal(0, 1, n_obs))

    # Create DataFrame
    df = pd.DataFrame({
        'session_close': prices,
        'date': pd.date_range('2023-01-01', periods=n_obs)
    })

    print(f"📊 Generated {len(df)} price observations")
    print(f"✅ Price range: {df['session_close'].min():.2f} - {df['session_close'].max():.2f}")

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Test log transformation
    df_transformed = feature_engineer.apply_robust_transformations(df)

    if 'log_return' in df_transformed.columns:
        print("✅ Log transformation successful!")
        print(f"✅ Log return range: {df_transformed['log_return'].min():.6f} - {df_transformed['log_return'].max():.6f}")
        print(f"✅ Log return mean: {df_transformed['log_return'].mean():.6f}")
        print(f"✅ Log return std: {df_transformed['log_return'].std():.6f}")
        return True
    else:
        print("❌ Log transformation failed")
        return False

def test_vpoc_integration():
    """Test VPOC integration with feature engineering."""
    print("\n=== Testing VPOC Integration ===")

    # Try to load some actual data
    data_path = '/workspace/DATA/MERGED/merged_es_vix_test.csv'

    if os.path.exists(data_path):
        print(f"📁 Loading data from {data_path}")
        data = pd.read_csv(data_path, parse_dates=['date'])

        if len(data) > 0:
            print(f"✅ Loaded {len(data)} records")
            print(f"📅 Date range: {data['date'].min()} to {data['date'].max()}")

            # Initialize feature engineer
            feature_engineer = FeatureEngineer()

            # Test feature generation
            print("🔧 Testing feature generation...")
            try:
                features_df = feature_engineer.prepare_features(data.head(200))  # Test with subset
                print(f"✅ Generated features: {features_df.shape}")

                # Test robust transformations
                features_df = feature_engineer.apply_robust_transformations(features_df)
                print(f"✅ Applied robust transformations: {features_df.shape}")

                # Test GARCH features
                if 'log_return' in features_df.columns:
                    garch_features = feature_engineer.calculate_garch_features(features_df['log_return'])
                    if garch_features:
                        print("✅ GARCH integration successful!")
                        print(f"✅ Generated {len(garch_features)} GARCH features")
                        return True
                    else:
                        print("❌ GARCH integration failed")
                        return False
                else:
                    print("❌ No log_return column found")
                    return False

            except Exception as e:
                print(f"❌ Feature generation failed: {e}")
                return False
        else:
            print("❌ No data found")
            return False
    else:
        print(f"❌ Data file not found: {data_path}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting GARCH and Log Transformation Tests\n")

    results = []

    # Test 1: GARCH calculation
    results.append(test_garch_calculation())

    # Test 2: Log transformation
    results.append(test_log_transformation())

    # Test 3: VPOC integration
    results.append(test_vpoc_integration())

    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ GARCH calculation is working correctly")
        print("✅ Log transformation is working correctly")
        print("✅ VPOC integration is working correctly")
        print("\n🚀 Ready to train model with all features!")
    else:
        print(f"❌ SOME TESTS FAILED! ({passed}/{total})")
        print("🔧 Need to fix issues before training")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)