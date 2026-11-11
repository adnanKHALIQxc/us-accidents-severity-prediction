# src/utils/logger.py

import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """Creates a configured logger for the project."""
    logger = logging.getLogger(name)
    
    # Only configure if it doesn't have handlers yet (prevents duplicate logs)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Format: Time - Name - Level - Message
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Print to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger