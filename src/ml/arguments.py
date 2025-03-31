"""
Common argument parsers for ML command-line tools.
"""

import os
import argparse
from src.config.settings import settings

def add_common_args(parser):
    """Add common arguments to an ArgumentParser."""
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=settings.DATA_DIR,
        help="Path to data directory"
    )
    
    return parser

def add_model_args(parser):
    """Add model-related arguments to parser."""
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.path.join(settings.TRAINING_DIR, "model_final.pt"),
        help="Path to ML model checkpoint"
    )
    return parser

def get_base_parser(description):
    """Create a parser with base/common arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser = add_common_args(parser)
    return parser

def get_common_parser(description):
    """Create a parser with common arguments."""
    return get_base_parser(description)
