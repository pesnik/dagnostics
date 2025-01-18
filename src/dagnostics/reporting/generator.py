import logging

from dagnostics.core.config import AppConfig

logger = logging.getLogger(__name__)


def setup_reporting(config: AppConfig):
    """Set up the reporting system."""
    logger.info("Setting up reporting system...")
    logger.info(f"Report format: {config.reporting.format}")
    logger.info(f"Output directory: {config.reporting.output_dir}")

    logger.info("Reporting system setup complete.")
