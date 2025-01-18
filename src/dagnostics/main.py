import logging

from dagnostics.core.config import load_config
from dagnostics.monitoring.collector import start_monitoring
from dagnostics.reporting.generator import setup_reporting
from dagnostics.utils.logger import setup_logging


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


if __name__ == "__main__":
    main()
