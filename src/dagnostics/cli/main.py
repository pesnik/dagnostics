import typer

from dagnostics.core.config import load_config
from dagnostics.monitoring.collector import start_monitoring
from dagnostics.reporting.generator import setup_reporting

app = typer.Typer()


@app.command()
def start():
    """Start the DAGnostics monitoring and reporting system."""
    config = load_config()
    typer.echo("Starting DAGnostics...")
    start_monitoring(config)
    setup_reporting(config)


@app.command()
def analyze(dag_name: str):
    """Analyze a specific DAG."""
    # config = load_config()
    typer.echo(f"Analyzing DAG: {dag_name}")


@app.command()
def report(daily: bool = False):
    """Generate a report."""
    # config = load_config()
    if daily:
        typer.echo("Generating daily report...")
    else:
        typer.echo("Generating report...")


def cli():
    """Entry point for the CLI."""
    app()
