#!/bin/bash

set -e

echo "🚀 Setting up DAGnostics..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# Install dependencies
echo "Installing dependencies..."
uv sync

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{clusters,processed,raw/{logfiles,metafiles}}
mkdir -p logs reports

# Copy example configs if they don't exist
if [ ! -f config/config.yaml ]; then
    echo "Creating config file..."
    cp config/config.yaml.example config/config.yaml
    echo "⚠️  Please edit config/config.yaml with your settings"
fi

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
uv run pre-commit install

# Initialize database
echo "Initializing database..."
uv run python -c "
from dagnostics.core.database import DatabaseManager
from dagnostics.core.config import load_config

config = load_config()
db_manager = DatabaseManager(config['database']['url'])
print('Database initialized successfully')
"

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config/config.yaml with your Airflow and LLM settings"
echo "2. Start Ollama and pull the model: ollama pull mistral"
echo "3. Run: uv run dagnostics --help"
echo ""
