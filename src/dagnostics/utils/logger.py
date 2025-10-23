import logging
import logging.config
import sys
from pathlib import Path

import yaml


def setup_logging():
    """Load logging configuration from logging.yaml and ensure logs directory exists."""
    logging_config_path = Path("config/logging.yaml")
    if logging_config_path.exists():
        with open(logging_config_path, "r") as f:
            config = yaml.safe_load(f)

            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)

            logging.config.dictConfig(config)
    else:
        # Configure logging to use stderr by default to avoid polluting stdout in CLI usage
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )
        logging.warning(
            "Logging configuration file not found. Using default logging settings."
        )
