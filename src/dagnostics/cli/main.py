import typer

from dagnostics.cli.commands import analyze, notify_failures, report, start
from dagnostics.utils.logger import setup_logging

setup_logging()

app = typer.Typer(help="DAGnostics - Intelligent ETL Monitoring System CLI")


app.command()(start)
app.command()(analyze)
app.command()(notify_failures)
app.command()(report)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
