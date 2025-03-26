#!/usr/bin/env python
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent  # This gets to futures_vpoc_backtest
sys.path.append(str(project_root))

# Print debugging information
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root added to path: {project_root}")

try:
    import pandas as pd
    print(f"Successfully imported pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Error importing pandas: {e}")
    print("Please make sure pandas is installed: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
    print(f"Successfully imported numpy version: {np.__version__}")
except ImportError as e:
    print(f"Error importing numpy: {e}")
    print("Please make sure numpy is installed: pip install numpy")
    sys.exit(1)

# Try to import from the src directory
try:
    from src.core.vpoc import VolumeProfileAnalyzer
    from src.utils.logging import get_logger
    print("Successfully imported from src modules")
except ImportError as e:
    print(f"Error importing from src: {e}")
    print("Make sure the src directory is properly set up")
    sys.exit(1)

# For comparing with the original implementation
notebooks_path = os.path.join(project_root, 'NOTEBOOKS')
sys.path.append(notebooks_path)
print(f"Added NOTEBOOKS path: {notebooks_path}")

try:
    from VPOC import calculate_volume_profile as original_calculate_profile
    from VPOC import find_vpoc as original_find_vpoc
    print("Successfully imported original VPOC functions")
except ImportError as e:
    print(f"Error importing original VPOC functions: {e}")
    print("This will prevent comparison with original implementation")

def test_vpoc_functionality():
    """Test the new VPOC implementation against the original."""
    logger = get_logger(__name__)
    logger.info("Testing VPOC functionality")
    
    # Load some sample data
    try:
        from src.config.settings import settings
        print(f"Loaded settings with DATA_DIR: {settings.DATA_DIR}")
        
        # Try to find a sample data file
        notebooks_dir = Path(notebooks_path)
        data_dir = settings.DATA_DIR
        
        # Try to import DATA_LOADER to get some data
        try:
            from DATA_LOADER import load_futures_data
            print("Successfully imported load_futures_data from NOTEBOOKS")
            
            # Load a small sample of data
            print(f"Attempting to load data from {data_dir}")
            df = load_futures_data(data_dir)
            if df is None or df.empty:
                logger.error("Failed to load data for testing")
                print("No data loaded, check your data directory")
                return
            
            print(f"Successfully loaded data with {len(df)} rows")
            
            # Filter to a single session for testing
            session_data = df[df['session'] == 'RTH'].copy()
            if len(session_data) > 0:
                single_day = session_data['date'].unique()[0]
                test_session = session_data[session_data['date'] == single_day]
                
                # Test new implementation
                logger.info(f"Testing session from {single_day} with {len(test_session)} bars")
                print(f"Testing session from {single_day} with {len(test_session)} bars")
                
                # Create analyzer and run calculations
                analyzer = VolumeProfileAnalyzer()
                
                # Calculate volume profile
                new_profile = analyzer.calculate_volume_profile(test_session)
                logger.info(f"New profile calculated with {len(new_profile)} price levels")
                print(f"New profile calculated with {len(new_profile)} price levels")
                
                # Find VPOC
                new_vpoc = analyzer.find_vpoc(new_profile)
                logger.info(f"New VPOC: {new_vpoc}")
                print(f"New VPOC: {new_vpoc}")
                
                # Find Value Area
                new_val, new_vah, new_va_pct = analyzer.find_value_area(new_profile)
                logger.info(f"New Value Area: {new_val} to {new_vah} ({new_va_pct:.2f}% of volume)")
                print(f"New Value Area: {new_val} to {new_vah} ({new_va_pct:.2f}% of volume)")
                
                # Compare with original implementation
                logger.info("Comparing with original implementation")
                print("\nComparing with original implementation:")
                
                try:
                    original_profile = original_calculate_profile(test_session)
                    original_vpoc = original_find_vpoc(original_profile)
                    
                    logger.info(f"Original VPOC: {original_vpoc}")
                    print(f"Original VPOC: {original_vpoc}")
                    print(f"Difference in VPOC: {abs(new_vpoc - original_vpoc)}")
                    
                    if abs(new_vpoc - original_vpoc) < 0.01:
                        logger.info("✅ VPOC calculation matches original implementation")
                        print("✅ VPOC calculation matches original implementation")
                    else:
                        logger.warning("❌ VPOC calculation differs from original implementation")
                        print("❌ VPOC calculation differs from original implementation")
                except Exception as e:
                    print(f"Error comparing with original implementation: {e}")
                
                # Run the full session analysis
                print("\nRunning full session analysis:")
                session_analysis = analyzer.analyze_session(test_session)
                logger.info(f"Full session analysis completed with {len(session_analysis)} metrics")
                print(f"Full session analysis completed with {len(session_analysis)} metrics")
                
                return session_analysis
            else:
                logger.error("No RTH session data found for testing")
                print("No RTH session data found for testing")
        except ImportError as e:
            logger.error(f"Error importing load_futures_data: {e}")
            print(f"Error importing load_futures_data: {e}")
    except Exception as e:
        logger.error(f"Error in VPOC testing: {e}")
        print(f"Error in VPOC testing: {e}")
        import traceback
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        print(traceback_str)

if __name__ == "__main__":
    print("\n===== Testing VPOC Implementation =====\n")
    results = test_vpoc_functionality()
    if results:
        print("\nSession Analysis Results:")
        for key, value in results.items():
            if key != 'volume_profile':  # Skip printing the full profile
                print(f"{key}: {value}")
    else:
        print("\nNo results returned from testing.")