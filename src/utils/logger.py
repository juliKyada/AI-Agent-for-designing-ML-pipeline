"""
Logging utilities for MetaFlow
"""
import sys
from pathlib import Path
from loguru import logger
from src.utils.config import get_config


def setup_logger():
    """Configure logger based on config settings"""
    config = get_config()
    
    # Remove default handler
    logger.remove()
    
    # Get logging configuration
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', 
                           "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")
    log_to_file = config.get('logging.log_to_file', True)
    log_file = config.get('logging.log_file', 'logs/metaflow.log')
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # Add file handler if enabled
    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )
    
    return logger


def get_logger():
    """Get configured logger instance"""
    return logger
