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

# Analyze a specific task failure
# Usage: just analyze-task <dag_id> <task_id> <run_id> <try_number>
analyze-task dag_id task_id run_id try_number:
    @echo "Analyzing task failure..."
    uv run dagnostics analyze {{dag_id}} {{task_id}} {{run_id}} {{try_number}}
    @echo "Analysis complete!"

# Analyze a specific task failure with verbose output
# Usage: just analyze-task-verbose <dag_id> <task_id> <run_id> <try_number>
analyze-task-verbose dag_id task_id run_id try_number:
    @echo "Analyzing task failure with verbose output..."
    uv run dagnostics analyze {{dag_id}} {{task_id}} {{run_id}} {{try_number}} --verbose
    @echo "Analysis complete!"

# Analyze a specific task failure with custom LLM provider
# Usage: just analyze-task-llm <dag_id> <task_id> <run_id> <try_number> <llm_provider>
analyze-task-llm dag_id task_id run_id try_number llm_provider:
    @echo "Analyzing task failure with {{llm_provider}}..."
    uv run dagnostics analyze {{dag_id}} {{task_id}} {{run_id}} {{try_number}} --llm {{llm_provider}}
    @echo "Analysis complete!"

# Notify about recent task failures (dry run)
# Usage: just notify-failures-dry-run [since_minutes]
notify-failures-dry-run since_minutes="60":
    @echo "Checking for failed tasks in last {{since_minutes}} minutes (dry run)..."
    uv run dagnostics notify-failures --since-minutes {{since_minutes}} --dry-run
    @echo "Dry run complete!"

# Notify about recent task failures (send actual SMS)
# Usage: just notify-failures [since_minutes]
notify-failures since_minutes="60":
    @echo "Checking for failed tasks in last {{since_minutes}} minutes..."
    uv run dagnostics notify-failures --since-minutes {{since_minutes}}
    @echo "Notifications sent!"

# Notify about recent task failures with real-time baseline analysis
# Usage: just notify-failures-realtime [since_minutes]
notify-failures-realtime since_minutes="60":
    @echo "Checking for failed tasks with real-time baseline analysis..."
    uv run dagnostics notify-failures --since-minutes {{since_minutes}} --no-use-stored-baseline
    @echo "Notifications sent!"

# Notify about recent task failures with custom LLM provider
# Usage: just notify-failures-llm <since_minutes> <llm_provider>
notify-failures-llm since_minutes llm_provider:
    @echo "Checking for failed tasks with {{llm_provider}}..."
    uv run dagnostics notify-failures --since-minutes {{since_minutes}} --llm {{llm_provider}}
    @echo "Notifications sent!"

# Start continuous monitoring
# Usage: just monitor [interval_minutes]
monitor interval_minutes="5":
    @echo "Starting continuous monitoring ({{interval_minutes}}m intervals)..."
    uv run dagnostics start --interval {{interval_minutes}}
    @echo "Monitoring started!"

# Start continuous monitoring as daemon
# Usage: just monitor-daemon [interval_minutes]
monitor-daemon interval_minutes="5":
    @echo "Starting continuous monitoring as daemon..."
    uv run dagnostics start --interval {{interval_minutes}} --daemon
    @echo "Daemon monitoring started!"

# Generate daily report
# Usage: just report-daily [format]
report-daily format="html":
    @echo "Generating daily report in {{format}} format..."
    uv run dagnostics report --daily --format {{format}}
    @echo "Daily report generated!"

# Generate summary report
# Usage: just report-summary [format]
report-summary format="html":
    @echo "Generating summary report in {{format}} format..."
    uv run dagnostics report --format {{format}}
    @echo "Summary report generated!"

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
