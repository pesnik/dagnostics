#!/bin/bash
# Setup script for training environment on a separate machine

set -e

echo "🧠 Setting up DAGnostics Training Environment"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the dagnostics root directory"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$(echo $PYTHON_VERSION | cut -d. -f1)" -lt 3 ] || [ "$(echo $PYTHON_VERSION | cut -d. -f2)" -lt 10 ]; then
    echo "❌ Error: Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Install UV if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    pip install uv
fi

echo "✅ UV package manager ready"

# Setup training dependencies
echo "📚 Installing training dependencies..."

# Copy training requirements
cp pyproject.toml.training pyproject.toml

# Create virtual environment and install dependencies
uv venv --python python3.10
source .venv/bin/activate

# Install with training dependencies
uv pip install -e .

echo "✅ Training dependencies installed"

# Check GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    print(f'   CUDA version: {torch.version.cuda}')
else:
    print('💻 No GPU detected - training will use CPU (slower)')
"

# Test model loading
echo "🧪 Testing model loading..."
python3 -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    print('✅ Model loading test passed')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
    exit(1)
"

# Create directories
echo "📁 Creating training directories..."
mkdir -p data/training data/raw data/processed models/fine_tuned logs

# Test CLI commands
echo "🧪 Testing training commands..."
python3 -c "
try:
    from dagnostics.training.dataset_generator import DatasetGenerator
    from dagnostics.training.fine_tuner import SLMFineTuner
    print('✅ Training modules import successfully')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo ""
echo "🎉 Training environment setup complete!"
echo ""
echo "📋 Available commands:"
echo "  uv run dagnostics training generate-dataset"
echo "  uv run dagnostics training train-model"
echo "  uv run dagnostics training deploy-ollama <model-path>"
echo "  uv run dagnostics training pipeline"
echo ""
echo "📊 Check feedback data:"
echo "  uv run dagnostics training feedback-stats"
echo ""
echo "🔄 To sync data from main server:"
echo "  rsync -av user@main-server:/path/to/dagnostics/data/ ./data/"
echo ""
echo "📚 Full guide: docs/fine_tuning_guide.md"
