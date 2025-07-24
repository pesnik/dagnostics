from invoke.tasks import task


@task
def format(c):
    """Format code."""
    c.run("uv run black .")
    c.run("uv run isort .")


@task
def lint(c):
    """Run all linters."""
    c.run("uv run flake8 .")
    c.run("uv run mypy .")
