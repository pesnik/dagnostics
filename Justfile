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
    @echo "Checking for failed tasks with real-time baseline analysis (from config)..."
    uv run dagnostics notify-failures --since-minutes {{since_minutes}}
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

# =====================================
# Fine-tuning and Training Commands
# =====================================

# Install fine-tuning dependencies
setup-training:
    @echo "Installing fine-tuning dependencies..."
    uv sync --extra finetuning
    @echo "Training environment ready!"

# Install all dependencies including training
setup-full:
    @echo "Installing all dependencies..."
    uv sync --extra all
    @echo "Full environment ready!"

# Check training environment status
training-status:
    @echo "Checking training environment..."
    uv run dagnostics training status

# Prepare fine-tuning datasets from human-reviewed data
# Usage: just prepare-training-data [dataset_path]
prepare-training-data dataset_path="data/training_dataset_2025-08-17T11-15-10.json":
    @echo "Preparing training datasets from {{dataset_path}}..."
    uv run dagnostics training prepare-data {{dataset_path}}
    @echo "Training datasets prepared!"

# Alternative: run the Python script directly
prepare-data-script dataset_path="data/training_dataset_2025-08-17T11-15-10.json":
    @echo "Running data preparation script..."
    uv run python scripts/prepare_training_data.py {{dataset_path}}
    @echo "Data preparation complete!"

# Fine-tune local model with default settings
train-local-default:
    @echo "Starting local fine-tuning with default settings..."
    uv run dagnostics training train-local --epochs 3 --batch-size 2
    @echo "Local fine-tuning complete!"

# Fine-tune local model with custom settings
# Usage: just train-local <model_name> <epochs> <batch_size>
train-local model_name="microsoft/DialoGPT-small" epochs="3" batch_size="2":
    @echo "Fine-tuning {{model_name}} for {{epochs}} epochs..."
    uv run dagnostics training train-local \
        --model-name "{{model_name}}" \
        --epochs {{epochs}} \
        --batch-size {{batch_size}} \
        --use-quantization true \
        --export-for-ollama true
    @echo "Local fine-tuning complete!"

# Fine-tune with OpenAI API
# Usage: just train-openai [model] [suffix]
train-openai model="gpt-3.5-turbo" suffix="dagnostics-error-extractor":
    @echo "Starting OpenAI fine-tuning with {{model}}..."
    uv run dagnostics training train-openai \
        --model "{{model}}" \
        --suffix "{{suffix}}" \
        --wait true
    @echo "OpenAI fine-tuning complete!"

# Prepare data for Anthropic fine-tuning
# Usage: just train-anthropic [model]
train-anthropic model="claude-3-haiku-20240307":
    @echo "Preparing Anthropic data for {{model}}..."
    uv run dagnostics training train-anthropic --model "{{model}}"
    @echo "Anthropic data preparation complete!"

# Submit remote training job
# Usage: just train-remote [server_url] [model_name] [epochs]
train-remote server_url="http://localhost:8001" model_name="microsoft/DialoGPT-small" epochs="3":
    @echo "Submitting remote training job to {{server_url}}..."
    uv run dagnostics training remote-train \
        --server-url "{{server_url}}" \
        --model-name "{{model_name}}" \
        --epochs {{epochs}} \
        --wait true
    @echo "Remote training job submitted!"

# Check remote training job status
# Usage: just remote-status <job_id> [server_url]
remote-status job_id server_url="http://localhost:8001":
    @echo "Checking status of job {{job_id}}..."
    uv run dagnostics training remote-status {{job_id}} --server-url "{{server_url}}"

# Download remote training result
# Usage: just remote-download <job_id> [output_dir] [server_url]
remote-download job_id output_dir="models/fine_tuned" server_url="http://localhost:8001":
    @echo "Downloading model from job {{job_id}}..."
    uv run dagnostics training remote-download {{job_id}} \
        --output-dir "{{output_dir}}" \
        --server-url "{{server_url}}"
    @echo "Model downloaded!"

# Evaluate a fine-tuned model
# Usage: just evaluate-model <model_path> [test_dataset] [model_type]
evaluate-model model_path test_dataset="data/fine_tuning/validation_dataset.jsonl" model_type="local":
    @echo "Evaluating model: {{model_path}}..."
    uv run dagnostics training evaluate "{{model_path}}" \
        --test-dataset "{{test_dataset}}" \
        --model-type "{{model_type}}"
    @echo "Model evaluation complete!"

# Deploy model to Ollama
# Usage: just deploy-ollama <model_path> [model_name]
deploy-ollama model_path model_name="dagnostics-error-extractor":
    @echo "Deploying {{model_path}} to Ollama as {{model_name}}..."
    uv run dagnostics training deploy-ollama "{{model_path}}" \
        --model-name "{{model_name}}" \
        --auto-build true
    @echo "Model deployed to Ollama!"

# Show feedback statistics
feedback-stats:
    @echo "Showing feedback statistics..."
    uv run dagnostics training feedback-stats

# Export feedback for training
# Usage: just export-feedback [min_rating]
export-feedback min_rating="3":
    @echo "Exporting feedback with minimum rating {{min_rating}}..."
    uv run dagnostics training export-feedback --min-rating {{min_rating}}
    @echo "Feedback exported!"

# Complete fine-tuning workflow (prepare → train → evaluate)
# Usage: just train-workflow [model_name] [epochs]
train-workflow model_name="microsoft/DialoGPT-small" epochs="3":
    @echo "Running complete fine-tuning workflow..."
    @just prepare-training-data
    @just train-local "{{model_name}}" "{{epochs}}" "2"
    @echo "Fine-tuning workflow complete!"

# Quick training setup and status check
train-setup:
    @echo "Setting up training environment..."
    @just setup-training
    @just training-status
    @echo "Training setup complete!"

# Setup remote training infrastructure
# Usage: just setup-remote [mode]
setup-remote mode="":
    @echo "Setting up remote training infrastructure..."
    @if [ "{{mode}}" = "" ]; then \
        uv run dagnostics training setup-remote; \
    else \
        uv run dagnostics training setup-remote --mode {{mode}}; \
    fi
    @echo "Remote training setup complete!"

# Start training server (for remote training)
# Usage: just start-training-server [host] [port]
start-training-server host="0.0.0.0" port="8001":
    @echo "Starting training server on {{host}}:{{port}}..."
    uv run dagnostics training start-server --host {{host}} --port {{port}}

# Setup and start local training server
train-server-local:
    @echo "Setting up and starting local training server..."
    @just setup-training
    @just start-training-server localhost 8001

# =====================================
# Web Dashboard Commands
# =====================================

# Start web dashboard with default settings
web:
    @echo "Starting web dashboard..."
    uv run dagnostics web
    @echo "Web dashboard started!"

# Start web dashboard with custom host and port
# Usage: just web-custom <host> <port>
web-custom host="0.0.0.0" port="8000":
    @echo "Starting web dashboard on {{host}}:{{port}}..."
    uv run dagnostics web --host {{host}} --port {{port}}
    @echo "Web dashboard started!"

# Start web dashboard in development mode
web-dev:
    @echo "Starting web dashboard in development mode..."
    uv run dagnostics web --reload --log-level debug
    @echo "Development web dashboard started!"

# =====================================
# Daemon and Monitoring Commands
# =====================================

# Start daemon service
daemon-start:
    @echo "Starting daemon service..."
    uv run dagnostics daemon start
    @echo "Daemon started!"

# Stop daemon service
daemon-stop:
    @echo "Stopping daemon service..."
    uv run dagnostics daemon stop
    @echo "Daemon stopped!"

# Check daemon status
daemon-status:
    @echo "Checking daemon status..."
    uv run dagnostics daemon status

# Restart daemon service
daemon-restart:
    @echo "Restarting daemon service..."
    @just daemon-stop
    @just daemon-start
    @echo "Daemon restarted!"

# =====================================
# Development Workflows
# =====================================

# Complete development setup with training
dev-full: setup-full format lint test
    @echo "Full development workflow complete!"

# Training development workflow
dev-training: setup-training train-setup
    @echo "Training development workflow complete!"

# Production deployment workflow
deploy-prod: clean format lint test build
    @echo "Production deployment workflow complete!"
