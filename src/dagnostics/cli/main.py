import typer

from dagnostics.cli.commands import (
    analyze,
    get_error,
    notify_failures,
    report,
    restart,
    start,
    status,
    stop,
    web,
)
from dagnostics.cli.feedback_commands import feedback_app

# Lazy import training commands to avoid heavy dependencies
try:
    from dagnostics.cli.training_commands import training_app

    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False
    training_app = None
from dagnostics.utils.logger import setup_logging

setup_logging()

app = typer.Typer(help="DAGnostics - Intelligent ETL Monitoring System CLI")

# Add feedback subcommand
app.add_typer(feedback_app, name="feedback")

# Add training subcommand only if dependencies are available
if HAS_TRAINING and training_app:
    app.add_typer(training_app, name="training")

app.command()(start)
app.command()(stop)
app.command()(status)
app.command()(restart)
app.command()(analyze)
app.command()(get_error)
app.command()(notify_failures)
app.command()(report)
app.command()(web)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
