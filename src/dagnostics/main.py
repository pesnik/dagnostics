import logging.config
from pathlib import Path

import yaml

from dagnostics.cli.main import cli
from dagnostics.core.config import load_config
from dagnostics.monitoring.collector import start_monitoring
from dagnostics.reporting.generator import setup_reporting


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
        logging.basicConfig(level=logging.INFO)
        logging.warning(
            "Logging configuration file not found. Using default logging settings."
        )


def main():
    """Main entry point for the DAGnostics application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting DAGnostics...")

    config = load_config()
    logger.info("Configuration loaded successfully.")

    start_monitoring(config)
    logger.info("Monitoring started.")

    setup_reporting(config)
    logger.info("Reporting setup complete.")

    cli()


if __name__ == "__main__":
    main()
