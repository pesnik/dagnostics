# Fine-Tuning Guide for DAGnostics

This guide walks you through setting up and running the automated fine-tuning pipeline for creating domain-specific small language models (SLMs) tailored to your error analysis needs.

## 🎯 Overview

The fine-tuning system allows you to:
- **Collect User Feedback**: Web interface for correcting LLM analyses
- **Generate Training Data**: Convert feedback + logs into instruction datasets
- **Fine-tune Models**: Train specialized SLMs using LoRA/QLoRA
- **Deploy Locally**: Convert models for Ollama deployment
- **Iterate Continuously**: Improve models as you collect more feedback

## 🛠 Prerequisites

### Dependencies

Add to your environment:

```bash
# Core ML dependencies
pip install torch transformers datasets peft
pip install bitsandbytes accelerate  # For quantization
pip install rich typer  # For CLI

# Optional: For GGUF conversion
# git clone https://github.com/ggerganov/llama.cpp
```

### Hardware Requirements

**Minimum:**
- 8GB RAM
- 4GB VRAM (GPU recommended but not required)

**Recommended:**
- 16GB+ RAM
- 8GB+ VRAM (RTX 3080/4070+ or equivalent)

**Cloud Options:**
- Google Colab Pro (T4/A100)
- AWS g4dn.xlarge
- RunPod/Vast.ai

## 📊 Phase 1: Data Collection & Feedback

### 1.1 Setup Feedback Collection

```bash
# Start the web interface with feedback endpoints
uv run dagnostics daemon start

# Access feedback interface at:
# http://localhost:8080/feedback/interface
```

### 1.2 Collect User Feedback

Users can review and correct LLM analyses through the web interface:

1. **Review**: View original log context and AI analysis
2. **Correct**: Edit error message, category, severity, reasoning
3. **Rate**: Give 1-5 star quality rating
4. **Submit**: Feedback stored for training

### 1.3 Monitor Feedback Quality

```bash
# View feedback statistics
uv run dagnostics training feedback-stats

# Export quality feedback (rating ≥ 3)
uv run dagnostics training export-feedback --min-rating 3
```

## 🔧 Phase 2: Dataset Generation

### 2.1 Generate Training Dataset

```bash
# Generate complete training dataset from logs + feedback
uv run dagnostics training generate-dataset

# Options:
uv run dagnostics training generate-dataset \
  --output-dir data/training \
  --min-examples 50 \
  --include-feedback true
```

This creates:
- `train_dataset.jsonl` (80% of data)
- `validation_dataset.jsonl` (20% of data)
- `dataset_info.json` (metadata)

### 2.2 Dataset Format

```json
{
  "instruction": "You are an expert data engineer analyzing Airflow ETL logs...",
  "input": "Log Context:\n[2025-08-10T16:13:23] ERROR: TPT_INFRA: TPT04183...",
  "output": "{\n  \"error_message\": \"TPT Error: TPT_INFRA: TPT04183\",\n  \"confidence\": 0.85,\n  \"category\": \"configuration_error\"\n}",
  "metadata": {"source": "user_feedback", "user_id": "engineer1"}
}
```

## 🚀 Phase 3: Model Training

### 3.1 Fine-tune Model

```bash
# Basic training with defaults
uv run dagnostics training train-model

# Advanced configuration
uv run dagnostics training train-model \
  --model-name "microsoft/DialoGPT-small" \
  --epochs 5 \
  --learning-rate 2e-4 \
  --batch-size 4 \
  --use-quantization true
```

### 3.2 Model Selection Guide

| Model | Size | Best For | Memory |
|-------|------|----------|---------|
| `microsoft/DialoGPT-small` | 117M | General chat/analysis | 2GB |
| `distilbert-base-uncased` | 66M | Classification tasks | 1GB |
| `google/flan-t5-small` | 80M | Instruction following | 1.5GB |
| `microsoft/codebert-base` | 125M | Code understanding | 2GB |

### 3.3 Training Configuration

The system uses **LoRA (Low-Rank Adaptation)** for efficient fine-tuning:

```python
lora_config = LoraConfig(
    r=16,                    # Rank of adaptation
    lora_alpha=32,          # LoRA scaling parameter
    target_modules=[...],   # Attention + MLP layers
    lora_dropout=0.1,       # Dropout for regularization
    task_type="CAUSAL_LM"   # Causal language modeling
)
```

**Benefits:**
- ✅ Fast training (minutes vs hours)
- ✅ Low memory usage (4-bit quantization)
- ✅ Preserves base model knowledge
- ✅ Easy to swap/version models

## 📦 Phase 4: Deployment

### 4.1 Deploy to Ollama

```bash
# Export and deploy model
uv run dagnostics training deploy-ollama \
  models/fine_tuned/dagnostics-slm-20250813 \
  --model-name dagnostics-slm-v1 \
  --auto-build true
```

### 4.2 Update Configuration

```yaml
# config/config.yaml
llm:
  default_provider: "ollama"
  providers:
    ollama:
      base_url: "http://localhost:11434"
      model: "dagnostics-slm-v1"  # Your fine-tuned model
      temperature: 0.1
```

### 4.3 Test Fine-tuned Model

```bash
# Test via Ollama directly
ollama run dagnostics-slm-v1

# Test via DAGnostics
uv run dagnostics analyze my_dag my_task 2025-08-13T10:00:00 1 --llm ollama
```

## 🔄 Phase 5: Continuous Improvement

### 5.1 Automated Pipeline

Run the complete pipeline when you have sufficient feedback:

```bash
# Full pipeline: dataset → training → deployment
uv run dagnostics training pipeline \
  --model-name "microsoft/DialoGPT-small" \
  --min-feedback 20 \
  --epochs 3 \
  --auto-deploy false
```

### 5.2 Model Versioning Strategy

```
models/
├── dagnostics-slm-v1.0/     # First fine-tuned model
├── dagnostics-slm-v1.1/     # Improved with more feedback
├── dagnostics-slm-v2.0/     # Major improvement/new base model
└── production/              # Current production model
```

### 5.3 Performance Monitoring

Track model performance over time:

```bash
# Feedback statistics
uv run dagnostics training feedback-stats

# Model evaluation metrics
# Check evaluation_results.json in model directory
```

## 📈 Best Practices

### Data Quality
- **Minimum 50-100 examples** for meaningful fine-tuning
- **High-quality feedback** (rating ≥ 3) performs better
- **Diverse error types** improve generalization
- **Regular updates** as your systems evolve

### Training Tips
- **Start small**: 3 epochs, low learning rate
- **Monitor validation loss**: Early stopping prevents overfitting
- **Use quantization**: Enables training on modest hardware
- **Batch size**: Start with 2-4, increase if memory allows

### Production Deployment
- **A/B testing**: Compare fine-tuned vs base model
- **Gradual rollout**: Start with non-critical DAGs
- **Monitoring**: Track error analysis accuracy
- **Rollback plan**: Keep previous model version ready

## 🐛 Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Reduce batch size and enable quantization
--batch-size 1 --use-quantization true
```

**No feedback data:**
```bash
# Check feedback collection
uv run dagnostics training feedback-stats

# Verify web interface is accessible
curl http://localhost:8080/feedback/stats
```

**Poor model performance:**
- Increase training examples (aim for 100+)
- Check data quality (high-rated feedback only)
- Try different base model
- Increase training epochs (5-10)

**Deployment issues:**
```bash
# Verify Ollama installation
ollama version

# Check model creation
ollama list | grep dagnostics
```

## 📚 Advanced Topics

### Custom Base Models

Use domain-specific base models:

```bash
# CodeBERT for code-heavy logs
--model-name "microsoft/codebert-base"

# T5 for instruction following
--model-name "google/flan-t5-small"
```

### Multi-GPU Training

For larger models/datasets:

```python
# In fine_tuner.py, modify TrainingArguments:
training_args = TrainingArguments(
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    # ... other args
)
```

### Custom Evaluation Metrics

Add domain-specific metrics:

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Custom accuracy calculation for error categorization
    # Return {"accuracy": accuracy, "f1": f1_score}
```

## 🎉 Success Metrics

You'll know the fine-tuning is working when:

- ✅ **Higher confidence scores** (0.8+ vs 0.5-0.7)
- ✅ **More accurate categorization** (fewer unknown errors)
- ✅ **Better error message extraction** (concise, actionable)
- ✅ **Domain-specific understanding** (TPT, Teradata, etc.)
- ✅ **Consistent performance** across different error types

The goal is not perfect accuracy, but **significantly better performance** on your specific error patterns compared to generic models.

---

**Next Steps:** Start collecting feedback and run your first fine-tuning experiment! 🚀
