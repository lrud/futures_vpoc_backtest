"""
Logging utilities for the project.
Provides consistent and configurable logging across modules.
"""

import logging
import os
import sys
from datetime import datetime

# Default log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Default log level
DEFAULT_LOG_LEVEL = 'INFO'

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create timestamp for log files
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Global logger dictionary to avoid duplicate initialization
_LOGGERS = {}

def setup_logging(level=None, log_file=None, log_format=None):
    """
    Setup root logger configuration.
    
    Parameters:
    -----------
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str
        Path to log file (None = auto-generate)
    log_format : str
        Log message format
    """
    # Set defaults
    level = level or DEFAULT_LOG_LEVEL
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    log_format = log_format or DEFAULT_LOG_FORMAT
    
    # Generate log file path if not provided
    if log_file is None:
        log_file = os.path.join(LOG_DIR, f'futures_vpoc_{TIMESTAMP}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set level for external libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Get root logger and log initialization
    logger = logging.getLogger()
    logger.info(f"Logging initialized at level {level}, writing to {log_file}")
    
    return logger

def get_logger(name):
    """
    Get a logger with the given name.
    Creates a new logger if not exists, otherwise returns existing one.
    
    Parameters:
    -----------
    name : str
        Logger name (typically __name__ of the calling module)
        
    Returns:
    --------
    logging.Logger
        Logger instance
    """
    global _LOGGERS
    
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    # Create new logger
    logger = logging.getLogger(name)
    
    # If root logger is not configured, set up a basic configuration
    if not logger.hasHandlers() and not logging.getLogger().hasHandlers():
        setup_logging()
    
    # Store in dictionary
    _LOGGERS[name] = logger
    
    return logger