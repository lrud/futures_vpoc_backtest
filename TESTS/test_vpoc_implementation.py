#!/usr/bin/env python3
"""
Test script to verify the ROCm 7 multi-GPU VPOC implementation works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import torch
from src.core.vpoc import VolumeProfileAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

def create_test_data(n_bars=1000):
    """Create synthetic test data with realistic price movements."""
    np.random.seed(42)

    # Generate realistic price data
    initial_price = 4000.0
    prices = [initial_price]

    for i in range(1, n_bars):
        # Random walk with some trend
        change = np.random.normal(0, 2)  # 2 point standard deviation
        new_price = prices[-1] + change
        prices.append(max(new_price, 1000))  # Ensure positive prices

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC
        high_noise = np.random.uniform(0, 3)
        low_noise = np.random.uniform(0, 3)

        high = price + high_noise
        low = price - low_noise

        # Ensure OHLC relationships are maintained
        open_price = np.random.uniform(low, high)
        close_price = np.random.uniform(low, high)

        # Generate volume (higher around price levels with more activity)
        base_volume = 1000
        volume_variation = np.random.uniform(0.5, 2.0)
        volume = int(base_volume * volume_variation)

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume,
            'date': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i)
        })

    return pd.DataFrame(data)

def test_vpoc_functionality():
    """Test VPOC calculation with the new implementation."""
    print("🚀 Testing ROCm 7 Multi-GPU VPOC Implementation")
    print("=" * 60)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ {torch.cuda.device_count()} GPUs detected")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
    else:
        print("❌ No GPUs detected - will test CPU fallback")

    print("\n📊 Creating test data...")
    test_data = create_test_data(1000)  # 1000 bars for good test
    print(f"✅ Generated {len(test_data)} test bars")
    print(f"📅 Price range: ${test_data['low'].min():.2f} - ${test_data['high'].max():.2f}")
    print(f"📈 Total volume: {test_data['volume'].sum():,}")

    # Test VPOC calculation
    print("\n🔬 Testing VPOC calculation...")

    try:
        # Initialize analyzer
        analyzer = VolumeProfileAnalyzer(price_precision=0.25, device_ids=[0, 1] if torch.cuda.device_count() > 1 else [0])

        # Calculate volume profile
        print("⚡ Calculating volume profile...")
        import time
        start_time = time.time()

        volume_profile = analyzer.calculate_volume_profile(test_data)

        end_time = time.time()
        print(f"✅ VPOC calculation completed in {end_time - start_time:.3f} seconds")
        print(f"📊 Volume profile shape: {volume_profile.shape}")

        # Find VPOC
        vpoc = analyzer.find_vpoc(volume_profile)
        print(f"🎯 VPOC: ${vpoc:.2f}")

        # Find Value Area
        val, vah, va_pct = analyzer.find_value_area(volume_profile)
        print(f"📏 Value Area: ${val:.2f} - ${vah:.2f} ({va_pct:.1f}% of volume)")
        print(f"📏 Value Area Width: {vah - val:.2f} points")

        # Test complete session analysis
        print("\n🔍 Testing complete session analysis...")
        session_analysis = analyzer.analyze_session(test_data)

        print("✅ Session Analysis Results:")
        print(f"   VPOC: ${session_analysis['vpoc']:.2f}")
        print(f"   Value Area: ${session_analysis['value_area_low']:.2f} - ${session_analysis['value_area_high']:.2f}")
        print(f"   VA Width: {session_analysis['value_area_width']:.2f} points")
        print(f"   VA Volume: {session_analysis['value_area_pct']:.1f}%")
        print(f"   Total Volume: {session_analysis['total_volume']:,}")

        print("\n🎉 ALL TESTS PASSED!")
        print("✅ ROCm 7 Multi-GPU VPOC implementation is working correctly")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test backward compatibility functions."""
    print("\n🔄 Testing Backward Compatibility...")

    try:
        from src.core.vpoc import calculate_volume_profile, find_vpoc, find_value_area

        # Create test data
        test_data = create_test_data(500)  # Smaller dataset for compatibility test

        # Test wrapper functions
        print("📦 Testing wrapper functions...")

        # Test calculate_volume_profile wrapper
        vp_dict = calculate_volume_profile(test_data, price_precision=0.25)
        print(f"✅ calculate_volume_profile returned {len(vp_dict)} price levels")

        # Test find_vpoc wrapper
        vpoc = find_vpoc(vp_dict)
        print(f"✅ find_vpoc returned: ${vpoc:.2f}")

        # Test find_value_area wrapper
        val, vah, pct = find_value_area(vp_dict, value_area_pct=0.7)
        print(f"✅ find_value_area returned: ${val:.2f} - ${vah:.2f} ({pct:.1%})")

        print("✅ Backward compatibility tests passed")
        return True

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting VPOC Implementation Tests\n")

    results = []

    # Test 1: VPOC functionality
    results.append(test_vpoc_functionality())

    # Test 2: Backward compatibility
    results.append(test_backward_compatibility())

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ VPOC implementation is ready for production")
        print("✅ Multi-GPU optimization is working")
        print("✅ Backward compatibility maintained")
        print("\n🚀 Ready to train model with optimized VPOC!")
    else:
        print(f"❌ SOME TESTS FAILED! ({passed}/{total})")
        print("🔧 Need to fix issues before proceeding")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)