from invoke import task


@task
def setup(c):
    """Setup development environment."""
    c.run("poetry install")
    c.run("pre-commit install")


@task
def clean(c):
    """Clean build artifacts."""
    patterns = ["build/", "dist/", "*.egg-info", "__pycache__", "*.pyc"]
    for pattern in patterns:
        c.run(f"find . -name '{pattern}' -exec rm -rf {{}} +")


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
    # c.run("pre-commit run --all-files")
