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
)
from dagnostics.cli.feedback_commands import feedback_app
from dagnostics.cli.training_commands import training_app
from dagnostics.utils.logger import setup_logging

setup_logging()

app = typer.Typer(help="DAGnostics - Intelligent ETL Monitoring System CLI")

# Add training subcommand
app.add_typer(training_app, name="training")
# Add feedback subcommand
app.add_typer(feedback_app, name="feedback")

app.command()(start)
app.command()(stop)
app.command()(status)
app.command()(restart)
app.command()(analyze)
app.command()(get_error)
app.command()(notify_failures)
app.command()(report)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
