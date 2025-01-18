from invoke import task


@task
def format(c):
    """Format code."""
    c.run("black .")
    c.run("isort .")


@task
def lint(c):
    """Run all linters."""
    c.run("flake8 .")
    c.run("mypy .")
