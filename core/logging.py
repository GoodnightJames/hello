"""
Structured JSON logging for the Trading Engine.

Every function logs its action (INFO) and errors (ERROR).
Log files are written as JSON lines to the logs/ directory,
one file per day: trading_YYYY-MM-DD.json
"""

import json
import logging
import os
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        # Include extra fields if provided via the `extra` kwarg
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name, log_dir="logs", level=None):
    """
    Get a configured logger that writes structured JSON to files.

    Args:
        name: Logger name (typically module name).
        log_dir: Directory for log files.
        level: Logging level. Defaults to INFO.

    Returns:
        Configured logging.Logger instance.
    """
    if level is None:
        level = logging.INFO

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # File handler — one file per day
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"trading_{today}.json")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s")
    )
    logger.addHandler(console_handler)

    return logger
