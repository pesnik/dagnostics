from invoke.tasks import task


@task
def setup(c):
    """Setup development environment."""
    c.run("uv sync --extra dev")
    c.run("uv run pre-commit install")


@task
def clean(c):
    """Clean build artifacts."""
    patterns = [
        "build/",
        "dist/",
        "*.egg-info",
        "__pycache__",
        "*.pyc",
        ".mypy_cache/",
        ".pytest_cache/",
    ]
    for pattern in patterns:
        c.run(f"find . -name '{pattern}' -exec rm -rf {{}} +", warn=True)


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
    c.run("uv run pre-commit run --all-files")


@task
def test(c):
    """Run tests."""
    c.run("uv run pytest")


@task
def test_cov(c):
    """Run tests with coverage."""
    c.run("uv run pytest --cov=dagnostics --cov-report=html --cov-report=term")


@task
def sync(c):
    """Sync dependencies."""
    c.run("uv sync --extra dev")
