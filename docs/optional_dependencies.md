# Optional Dependencies Guide

DAGnostics uses optional dependencies to keep the core package lightweight while providing advanced features when needed.

## Installation Options

### Core Package (Default)
```bash
pip install dagnostics
```
Includes: Basic analysis, monitoring, API client, CLI tools

### With Fine-tuning Support
```bash
pip install 'dagnostics[finetuning]'
```
Includes: All core features + PyTorch, Transformers, PEFT, etc.

### All Features
```bash
pip install 'dagnostics[all]'
```
Includes: Everything

### Development Setup
```bash
git clone https://github.com/your-org/dagnostics
cd dagnostics
uv sync --dev  # Installs all dependencies including fine-tuning
```

## Usage Examples

### Basic Usage (Always Available)
```python
from dagnostics import DAGnostics

# Core functionality works without ML dependencies
client = DAGnostics()
result = client.analyze_task_failure("my_dag", "my_task")
```

### Fine-tuning (Requires Extra Dependencies)
```python
from dagnostics.training.fine_tuner import SLMFineTuner

try:
    # Only works if 'dagnostics[finetuning]' is installed
    tuner = SLMFineTuner()
    model_path = tuner.train_model("training_data.jsonl")
except ImportError as e:
    print(f"Fine-tuning not available: {e}")
    print("Install with: pip install 'dagnostics[finetuning]'")
```

## Checking Feature Availability

```python
from dagnostics.training.fine_tuner import HAS_ML_DEPS

if HAS_ML_DEPS:
    print("✅ Fine-tuning available")
    from dagnostics.training.fine_tuner import SLMFineTuner
    tuner = SLMFineTuner()
else:
    print("❌ Fine-tuning not available")
    print("Install with: pip install 'dagnostics[finetuning]'")
```

## Why This Approach?

- **Lightweight core**: Users who don't need fine-tuning don't get 2GB+ of PyTorch
- **Clear separation**: Core analysis vs advanced ML features
- **Standard Python pattern**: Same approach used by pandas, scikit-learn, FastAPI
- **Developer-friendly**: Developers get everything, users choose what they need
