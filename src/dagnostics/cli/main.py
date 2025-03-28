import logging

import typer

from dagnostics.core.config import load_config
from dagnostics.monitoring.collector import start_monitoring
from dagnostics.reporting.generator import setup_reporting
from dagnostics.utils.logger import setup_logging

setup_logging()

logger = logging.getLogger(__name__)
logger.info("Logger initialized for CLI module.")

app = typer.Typer()


@app.command()
def start():
    """Start the DAGnostics monitoring and reporting system."""
    config = load_config()
    logger.info("Starting DAGnostics...")
    typer.echo("Starting DAGnostics...")
    start_monitoring(config)
    setup_reporting(config)
    logger.info("DAGnostics started successfully.")


@app.command()
def analyze(dag_name: str):
    """Analyze a specific DAG."""
    # config = load_config()
    logger.info(f"Analyzing DAG: {dag_name}")
    typer.echo(f"Analyzing DAG: {dag_name}")

    logger.info(f"Analysis completed for DAG: {dag_name}")


@app.command()
def report(daily: bool = False):
    """Generate a report."""
    # config = load_config()
    if daily:
        logger.info("Generating daily report...")
        typer.echo("Generating daily report...")
    else:
        logger.info("Generating report...")
        typer.echo("Generating report...")

    logger.info("Report generation completed.")


def cli():
    """Entry point for the CLI."""
    app()
