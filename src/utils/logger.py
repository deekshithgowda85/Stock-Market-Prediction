"""Logging configuration for the Stock Forecasting System."""
import logging
import sys
from typing import Optional

from src.config.settings import settings


def setup_logger(
    name: str,
    log_level: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console handler.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level
    level = log_level or settings.log_level
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name not in logging.Logger.manager.loggerDict:
        return setup_logger(name)
    return logging.getLogger(name)

