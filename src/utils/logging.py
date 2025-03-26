import logging
import os
from pathlib import Path

def get_logger(name, level=logging.INFO):
    """Create a logger with consistent formatting."""
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # Try to create file handler if possible
        try:
            from src.config.settings import settings
            logs_dir = settings.BASE_DIR / "logs"
            os.makedirs(logs_dir, exist_ok=True)
            log_file = logs_dir / f"{name.replace('.', '_')}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
        except (ImportError, AttributeError):
            # If settings is not available, just use console logging
            pass
    
    return logger