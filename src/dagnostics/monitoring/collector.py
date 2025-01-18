import logging

from dagnostics.core.config import AppConfig

logger = logging.getLogger(__name__)


def start_monitoring(config: AppConfig):
    """Start monitoring DAG failures."""
    logger.info("Starting monitoring system...")
    logger.info(f"Monitoring interval: {config.monitoring.interval}")
    logger.info(f"Log path: {config.monitoring.log_path}")

    logger.info("Monitoring system started.")
