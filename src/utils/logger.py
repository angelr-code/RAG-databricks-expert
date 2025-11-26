import os 
import sys

from loguru import logger as loguru_logger

def setup_logging(log_level: str = None):
    """
    Setup logging for the application. If running within a Prefect flow, use Prefect's logger.
    Otherwise, use Loguru for logging.
    
    Args:
        log_level (str, optional): The logging level to set. Defaults to None, which uses the LOG_LEVEL
                                   environment variable or "DEBUG" if not set.
    
    Returns:
        Logger: Configured logger instance.
    """
    log_level = log_level or os.getenv("LOG_LEVEL", "DEBUG").upper()
    
    loguru_logger.remove()
    loguru_logger.add(
            sys.stdout,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>",
        )
    loguru_logger.debug(f"Logging initialized at {log_level} level (Loguru).")
    return loguru_logger