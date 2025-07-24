# justfile for dagnostics project
# Install just: https://github.com/casey/just

# Default recipe to display help
default:
    @just --list

# Setup development environment
setup:
    @echo "Setting up development environment..."
    uv sync --extra dev
    uv run pre-commit install
    @echo "Development environment ready!"

# Clean build artifacts and cache files
clean:
    @echo "Cleaning build artifacts..."
    find . -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -type f -delete 2>/dev/null || true
    find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "htmlcov" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".coverage" -type f -delete 2>/dev/null || true
    @echo "Cleanup complete!"

# Format code with black and isort
format:
    @echo "Formatting code..."
    uv run black .
    uv run isort .
    @echo "Code formatted!"

# Run all linters
lint:
    @echo "Running linters..."
    uv run flake8 .
    uv run mypy .
    uv run pre-commit run --all-files
    @echo "Linting complete!"

# Run tests
test:
    @echo "Running tests..."
    uv run pytest
    @echo "Tests complete!"

# Run tests with coverage
test-cov:
    @echo "Running tests with coverage..."
    uv run pytest --cov=dagnostics --cov-report=html --cov-report=term
    @echo "Coverage report generated!"

# Build the package
build:
    @echo "Building package..."
    uv build
    @echo "Package built!"

# Publish to TestPyPI
publish-test:
    @echo "Publishing to TestPyPI..."
    uv publish --repository testpypi
    @echo "Published to TestPyPI!"

# Publish to PyPI
publish:
    @echo "Publishing to PyPI..."
    uv publish
    @echo "Published to PyPI!"

# Sync dependencies
sync:
    @echo "Syncing dependencies..."
    uv sync --extra dev
    @echo "Dependencies synced!"

# Run the main application
run:
    @echo "Starting dagnostics..."
    uv run start

# Run the CLI application
cli:
    @echo "Starting dagnostics CLI..."
    uv run dagnostics

# Complete development workflow
dev: setup format lint test
    @echo "Development workflow complete!"

# CI workflow
ci: format lint test
    @echo "CI workflow complete!"

# Quick format and lint check
check:
    @echo "Running quick checks..."
    uv run black --check .
    uv run isort --check-only .
    uv run flake8 .
    @echo "Quick checks complete!"

# Watch files and run tests on change (requires entr)
watch:
    find . -name "*.py" | entr -c just test
