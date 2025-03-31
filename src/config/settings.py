"""
Global settings and configuration for the futures VPOC project.
Provides a centralized place for all configuration values.
"""

import os
from pathlib import Path

class Settings:
    def __init__(self, config_file=None):
        # Default settings - use the existing project structure
        self.BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Paths
        self.DATA_DIR = self.BASE_DIR / 'DATA'
        self.CLEANED_DATA_DIR = self.DATA_DIR / 'CLEANED'
        self.RESULTS_DIR = self.BASE_DIR / 'RESULTS'
        self.STRATEGY_DIR = self.BASE_DIR / 'STRATEGY'
        self.BACKTEST_DIR = self.BASE_DIR / 'BACKTEST'
        self.TRAINING_DIR = self.BASE_DIR / 'TRAINING'
        
        # Core settings
        self.SESSION_TYPE = 'RTH'
        self.PRICE_PRECISION = 0.25
        
        # Backtest Parameters
        self.INITIAL_CAPITAL = 100000
        self.COMMISSION_PER_TRADE = 10
        self.SLIPPAGE = 0.25
        self.RISK_PER_TRADE = 0.01
        self.RISK_FREE_RATE = 0.02
        self.MARGIN_REQUIREMENT = 0.1
        self.OVERNIGHT_MARGIN = 0.15
        self.MAX_POSITION_SIZE = 10
        self.MIN_CAPITAL_BUFFER = 0.2
        
        # ML Parameters
        self.DEFAULT_LOOKBACK_PERIODS = [5, 10, 20, 50]
        self.DEFAULT_PREDICTION_THRESHOLD = 0.5
        self.DEFAULT_CONFIDENCE_THRESHOLD = 70
        
        # Override with config file if provided
        if config_file and os.path.exists(config_file):
            self._load_config(config_file)
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _load_config(self, config_file):
        """Load configuration from file."""
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Update settings from config
            for key, value in config.items():
                if hasattr(self, key.upper()):
                    setattr(self, key.upper(), value)
                    
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.DATA_DIR, 
            self.CLEANED_DATA_DIR, 
            self.RESULTS_DIR,
            self.STRATEGY_DIR,
            self.BACKTEST_DIR,
            self.TRAINING_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Create global settings instance
settings = Settings()

# Export variables to make them importable at the module level
BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
CLEANED_DATA_DIR = settings.CLEANED_DATA_DIR
RESULTS_DIR = settings.RESULTS_DIR
STRATEGY_DIR = settings.STRATEGY_DIR
BACKTEST_DIR = settings.BACKTEST_DIR
TRAINING_DIR = settings.TRAINING_DIR
SESSION_TYPE = settings.SESSION_TYPE
PRICE_PRECISION = settings.PRICE_PRECISION
INITIAL_CAPITAL = settings.INITIAL_CAPITAL
COMMISSION_PER_TRADE = settings.COMMISSION_PER_TRADE
SLIPPAGE = settings.SLIPPAGE
RISK_PER_TRADE = settings.RISK_PER_TRADE
RISK_FREE_RATE = settings.RISK_FREE_RATE
MARGIN_REQUIREMENT = settings.MARGIN_REQUIREMENT
OVERNIGHT_MARGIN = settings.OVERNIGHT_MARGIN
MAX_POSITION_SIZE = settings.MAX_POSITION_SIZE
MIN_CAPITAL_BUFFER = settings.MIN_CAPITAL_BUFFER
LOOKBACK_PERIODS = settings.DEFAULT_LOOKBACK_PERIODS
DEFAULT_PREDICTION_THRESHOLD = settings.DEFAULT_PREDICTION_THRESHOLD
DEFAULT_CONFIDENCE_THRESHOLD = settings.DEFAULT_CONFIDENCE_THRESHOLD