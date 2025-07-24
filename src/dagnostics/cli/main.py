import typer

from dagnostics.utils.logger import setup_logging

from .commands import analyze, report, start

setup_logging()

app = typer.Typer()

app.command()(start)
app.command()(analyze)
app.command()(report)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
