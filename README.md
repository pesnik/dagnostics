# DAGnostics 🔍

DAGnostics is an intelligent ETL monitoring system that leverages LLMs to analyze, categorize, and report DAG failures in data pipelines. It provides automated parsing of DAG errors and is designed to generate comprehensive statistics for better observability.

## 🌟 Features (v0.5.0)

### 🔍 **Core Analysis Engine**
- **Intelligent Error Analysis**: Automated DAG error log parsing and categorization using multiple LLM providers (Ollama, OpenAI, Anthropic, Gemini)
- **Smart Baseline System**: Advanced error pattern recognition using Drain3 log clustering with baseline creation from successful task runs
- **Few-Shot Learning**: Configurable prompts with curated examples for improved error extraction accuracy
- **Multi-Provider LLM Support**: Seamless switching between local (Ollama) and cloud LLM providers
- **Anomaly Detection**: Identify new error patterns by comparing against successful task baselines

### 🌐 **Web Dashboard & Real-time Monitoring**
- **Interactive Web Dashboard**: Modern React-like dashboard with real-time monitoring capabilities
- **WebSocket Integration**: Live updates for analysis completion, new failures, and status changes
- **RESTful API**: Comprehensive API endpoints for analysis, dashboard, monitoring, and training
- **Real-time Statistics**: Live error trends, categories breakdown, and failure timelines
- **Mobile-Responsive**: Optimized for desktop and mobile monitoring

### 🎯 **Training Dataset Management & Fine-tuning**
- **Training Interface**: Web-based interface for creating and managing ML training datasets
- **Human Feedback Loop**: Review and correct LLM predictions to improve model accuracy
- **Dataset Export**: Export training data in JSON, CSV, and JSONL formats
- **Live Data Integration**: Pull failed tasks directly from Airflow for dataset creation
- **Model Fine-tuning**: Built-in support for fine-tuning small language models with LoRA/QLoRA
- **Multi-Provider Training**: Local models (HuggingFace), OpenAI API, and Anthropic preparation
- **Model Evaluation**: Comprehensive accuracy metrics and production readiness assessment
- **Ollama Integration**: Export fine-tuned models for local deployment

### 🛠 **Integration & Operations**
- **Airflow Integration**: Direct integration with Airflow API and database for real-time log collection
- **CLI Interface**: Enhanced command-line tools including web dashboard launcher
- **Smart Alerting**: SMS/Email notifications with concise error summaries
- **Daemon Service**: Background monitoring service for continuous error detection
- **Configurable Prompts**: Customize LLM prompts without code deployment via configuration files

**Planned / Future Enhancements:**

- Advanced ML model training and deployment
- Integration with existing ETL monitoring systems
- Enhanced analytics and predictive capabilities

---

## System Architecture
![System Architecture](docs/system_architecture.svg)
---

## 🛠 Tech Stack

### Core Framework
- **Python 3.10+** with modern async/await patterns
- **uv** for lightning-fast dependency management
- **Pydantic** for type-safe configuration and data validation
- **SQLAlchemy** for database operations

### LLM & AI Components
- **Ollama** for local LLM deployment (privacy-focused, cost-effective)
- **OpenAI API** (GPT-3.5, GPT-4) for cloud-based analysis
- **Anthropic Claude** for advanced reasoning capabilities
- **Google Gemini** for multimodal analysis
- **Drain3** for intelligent log clustering and pattern recognition

### Web & API
- **FastAPI** for high-performance REST API endpoints with WebSocket support
- **Uvicorn** for ASGI web server with real-time capabilities
- **WebSockets** for live dashboard updates and real-time monitoring
- **Typer** for intuitive CLI interface with enhanced web commands
- **Modern HTML/CSS/JS** for responsive web dashboard

### Data Processing
- **Pandas** for log data analysis
- **PyYAML** for configuration management
- **Requests** for HTTP integrations (Airflow API, SMS gateways)

---

## 📋 Prerequisites

- Python 3.10 or higher
- **uv** installed on your system (`pip install uv`)
- Ollama installed and running locally (for default LLM usage)
- Access to your ETL system's logs

---

## 🚀 Quick Start

### Option 1: Using uv (Recommended)

1.  Navigate to the project and install dependencies:

```bash
cd dagnostics
# Basic installation (web dashboard only)
uv sync

# With LLM providers for full analysis
uv sync --extra llm

# With web dashboard (minimal)
uv sync --extra web

# With development dependencies
uv sync --extra dev

# With fine-tuning dependencies (heavy ML libraries)
uv sync --extra finetuning

# With all optional dependencies
uv sync --extra all
```

### Option 2: Using pip

```bash
# Basic installation (web dashboard only)
pip install dagnostics

# With LLM providers for full analysis
pip install dagnostics[llm]

# With web dashboard (minimal)
pip install dagnostics[web]

# With development dependencies
pip install dagnostics[dev]

# With fine-tuning dependencies (heavy ML libraries)
pip install dagnostics[finetuning]

# With all optional dependencies
pip install dagnostics[all]
```

### 💡 **Dependency Recommendations:**

- **🚀 Web Dashboard Only:** `pip install dagnostics[web]` - Minimal dependencies, fast installation
- **🧠 Full Analysis:** `pip install dagnostics[llm]` - Includes LLM providers for error analysis
- **🔬 Model Training:** `pip install dagnostics[finetuning]` - ML libraries for local model fine-tuning
- **👨‍💻 Development:** `pip install dagnostics[dev]` - Testing and linting tools

### Setup Steps

1.  Choose your installation method above, then continue with setup:

2.  Set up pre-commit hooks (if using uv for development):

```bash
uv run pre-commit install
```

3.  Set up Ollama with your preferred model:

```bash
ollama pull mistral
```

4.  Configure your environment:

```bash
cp config/config.yaml.example config/config.yaml
# Edit config.yaml with your Airflow credentials and LLM provider settings
```

5.  Test the system with built-in few-shot learning:

```bash
# Analyze a specific task failure (replace with actual values)
uv run dagnostics analyze my_dag my_task 2025-08-13T10:00:00 1 --llm ollama

# Start background monitoring daemon
uv run dagnostics daemon start

# Check daemon status
uv run dagnostics daemon status
```

---

## 📁 Project Structure

```
dagnostics/
├── data/
│   ├── clusters/              # Drain3 cluster persistence & baselines
│   ├── raw/                   # Raw log files
│   ├── processed/             # Processed analysis results
│   └── training_data.jsonl    # Generated training datasets
├── src/dagnostics/
│   ├── api/                   # FastAPI REST API
│   ├── cli/                   # Command-line interface
│   ├── core/                  # Models, config, database
│   ├── daemon/                # Background monitoring service
│   ├── llm/                   # LLM providers & configurable prompts
│   ├── clustering/            # Drain3 log clustering & baselines
│   ├── heuristics/            # Pattern filtering engines
│   ├── monitoring/            # Airflow integration & collectors
│   ├── reporting/             # Report generation (stub)
│   ├── web/                   # Web dashboard UI
│   └── utils/                 # Helpers, logging, SMS
├── config/
│   ├── config.yaml            # Main configuration
│   ├── drain3.ini            # Drain3 clustering settings
│   ├── filter_patterns.yaml  # Heuristic filtering patterns
│   └── logging.yaml          # Logging configuration
├── tests/                     # Test suites
├── scripts/                   # Development & deployment scripts
└── docs/                     # Documentation
```

---

## 🔧 Configuration

DAGnostics is highly configurable through `config/config.yaml`. Key configuration areas include:

### Core Configuration Sections

- **Airflow**: Connection settings, database URL, authentication
- **LLM Providers**: Configure multiple LLM providers (Ollama, OpenAI, Anthropic, Gemini)
- **Prompts**: Customize prompts and add few-shot examples for better accuracy
- **Monitoring**: Baseline settings, check intervals, log processing limits
- **Drain3**: Log clustering parameters for pattern recognition
- **Alerts**: SMS/Email notification settings
- **Database**: DAGnostics internal database configuration

### Configurable Prompts System

DAGnostics now supports configurable prompts with few-shot learning:

```yaml
prompts:
  # Few-shot examples for better error extraction
  few_shot_examples:
    error_extraction:
      - log_context: |
          [2025-08-13 10:15:25] ERROR: psycopg2.OperationalError: FATAL: database "analytics_db" does not exist
        extracted_response: |
          {
            "error_message": "psycopg2.OperationalError: FATAL: database \"analytics_db\" does not exist",
            "confidence": 0.95,
            "category": "configuration_error",
            "severity": "high",
            "reasoning": "Database connection error due to missing database"
          }

  # Custom prompt templates (override defaults)
  templates:
    error_extraction: |
      You are an expert ETL engineer analyzing Airflow task failure logs...
```

---

## 🤖 Model Fine-tuning & Training

DAGnostics provides comprehensive fine-tuning capabilities to improve error extraction accuracy using your production data.

### 🎯 Quick Start: Fine-tuning Workflow

```bash
# 1. Check training environment status
dagnostics training status

# 2. Prepare datasets from human-reviewed data
dagnostics training prepare-data data/your_training_dataset.json

# 3. Choose your training method:

# Option A: Local fine-tuning (requires GPU/training deps)
dagnostics training train-local --epochs 3 --batch-size 2

# Option B: OpenAI API fine-tuning
export OPENAI_API_KEY="your-key-here"
dagnostics training train-openai --model gpt-3.5-turbo

# Option C: Remote training server
dagnostics training remote-train --server-url http://training-server:8001

# 4. Evaluate your model
dagnostics training evaluate <model_path> --test-dataset data/fine_tuning/validation_dataset.jsonl

# 5. Deploy to Ollama for local inference
dagnostics training deploy-ollama <model_path> --model-name my-error-extractor
```

### 📊 Training Data Requirements

DAGnostics fine-tuning works best with:

- **Minimum 50+ examples** (more is better)
- **Human-reviewed error extractions** for quality
- **Diverse error patterns** from your production environment
- **Balanced category distribution** across error types

### 🔧 Training Options

#### Local Model Fine-tuning

**Requirements:**
- GPU with 8GB+ VRAM (recommended)
- Training dependencies: `pip install dagnostics[finetuning]`

**Features:**
- **LoRA/QLoRA**: Memory-efficient fine-tuning
- **Quantization**: 4-bit training for resource efficiency
- **Custom Models**: Support for any HuggingFace model
- **Ollama Export**: Direct deployment to local inference

```bash
# Install training dependencies
pip install dagnostics[finetuning]

# Fine-tune with custom settings
dagnostics training train-local \
  --model-name "microsoft/DialoGPT-small" \
  --epochs 5 \
  --learning-rate 2e-4 \
  --batch-size 4 \
  --model-output-name "my-error-extractor" \
  --use-quantization true
```

#### OpenAI API Fine-tuning

**Requirements:**
- OpenAI API key with fine-tuning access
- Credits for training costs

**Features:**
- **Cloud Training**: No local GPU required
- **Production Ready**: High-quality models
- **Automatic Scaling**: Handles large datasets
- **API Integration**: Seamless deployment

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Start fine-tuning
dagnostics training train-openai \
  --model "gpt-3.5-turbo" \
  --suffix "my-error-extractor" \
  --wait true
```

#### Remote Training Server

**Requirements:**
- Remote training server with GPU
- Network access to training machine

**Features:**
- **Distributed Training**: Offload compute to dedicated machines
- **Job Management**: Monitor training progress remotely
- **Model Download**: Retrieve trained models automatically

```bash
# Submit training job
dagnostics training remote-train \
  --model-name "microsoft/DialoGPT-small" \
  --epochs 3 \
  --server-url "http://gpu-server:8001" \
  --wait true

# Check job status
dagnostics training remote-status <job_id>

# Download completed model
dagnostics training remote-download <job_id>
```

### 📈 Model Evaluation

Evaluate your fine-tuned models with comprehensive metrics:

```bash
# Evaluate local model
dagnostics training evaluate models/my-model \
  --test-dataset data/fine_tuning/validation_dataset.jsonl \
  --model-type local

# Evaluate OpenAI fine-tuned model
dagnostics training evaluate "ft:gpt-3.5-turbo:my-org:model:abc123" \
  --model-type openai

# View detailed evaluation report
cat evaluations/evaluation_20250817_143022.md
```

**Evaluation Metrics:**
- **Accuracy**: Percentage of correctly extracted errors
- **Exact Match Rate**: Perfect string matches with human labels
- **Similarity Score**: Token-based similarity for partial matches
- **Category Analysis**: Performance breakdown by error type

### 🚀 Deployment Options

#### Deploy to Ollama (Local)

```bash
# Export and deploy fine-tuned model
dagnostics training deploy-ollama models/my-model \
  --model-name "dagnostics-error-extractor" \
  --auto-build true

# Test deployed model
ollama run dagnostics-error-extractor "Analyze this error log..."

# Update DAGnostics config to use fine-tuned model
# config/config.yaml:
llm:
  default_provider: "ollama"
  providers:
    ollama:
      base_url: "http://localhost:11434"
      model: "dagnostics-error-extractor"
```

#### Use OpenAI Fine-tuned Model

```bash
# Update config to use fine-tuned OpenAI model
# config/config.yaml:
llm:
  default_provider: "openai"
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "ft:gpt-3.5-turbo:my-org:model:abc123"
```

### 🔄 Continuous Improvement Workflow

1. **Collect Production Data**: Use web interface to review and correct LLM predictions
2. **Export Training Data**: `dagnostics training export-feedback --min-rating 3`
3. **Retrain Models**: Periodically fine-tune with new human feedback
4. **A/B Testing**: Compare fine-tuned vs base model performance
5. **Production Deployment**: Replace base models with fine-tuned versions

### 📚 Training Commands Reference

| Command | Description |
|---------|-------------|
| `training status` | Show training environment and dataset status |
| `training prepare-data` | Convert human-reviewed data to training format |
| `training train-local` | Fine-tune local model with LoRA/QLoRA |
| `training train-openai` | Fine-tune using OpenAI API |
| `training train-anthropic` | Prepare data for Anthropic (when available) |
| `training evaluate` | Evaluate model accuracy on test data |
| `training deploy-ollama` | Export model for Ollama deployment |
| `training remote-train` | Submit job to remote training server |
| `training remote-status` | Check remote training job status |
| `training feedback-stats` | Show human feedback statistics |
| `training export-feedback` | Export feedback for training |
      {few_shot_examples}

      Now analyze this log:
      {log_context}
```

### LLM Provider Configuration

```yaml
llm:
  default_provider: "ollama"  # ollama, openai, anthropic, gemini
  providers:
    ollama:
      base_url: "http://localhost:11434"
      model: "mistral"
      temperature: 0.1
    gemini:
      api_key: "YOUR_API_KEY"
      model: "gemini-2.5-flash"
      temperature: 0.0
```

### Customizing Prompts and Examples

#### Adding Your Own Few-Shot Examples

Edit `config/config.yaml` to add domain-specific examples:

```yaml
prompts:
  few_shot_examples:
    error_extraction:
      - log_context: |
          [2025-08-13 15:30:25] ERROR: Your custom error pattern here
          [2025-08-13 15:30:25] ERROR: Additional context
        extracted_response: |
          {
            "error_message": "Extracted error message",
            "confidence": 0.90,
            "category": "configuration_error",
            "severity": "high",
            "reasoning": "Why this is the root cause"
          }
```

#### Creating Custom Prompt Templates

Override any default prompt by adding to `config.yaml`:

```yaml
prompts:
  templates:
    sms_error_extraction: |
      Custom SMS prompt template here.
      Extract concise error for: {dag_id}.{task_id}
      Log: {log_context}
```

#### Best Practices for Prompt Customization

1. **Include Diverse Examples**: Cover different error types, severity levels, and log formats
2. **Be Specific**: Include actual log snippets and exact expected outputs
3. **Test Iteratively**: Use the CLI to test prompt changes before deployment
4. **Keep Examples Current**: Update examples as your systems evolve
5. **Limit Example Count**: 3-5 examples per prompt type for optimal performance

---

## 🧠 How It Works

### Smart Baseline System

DAGnostics uses an intelligent baseline approach for error detection:

1. **Baseline Creation**: For each DAG task, DAGnostics analyzes successful runs to create a "normal behavior" baseline using Drain3 log clustering
2. **Anomaly Detection**: When tasks fail, logs are compared against baselines to identify truly anomalous patterns vs. known issues
3. **Adaptive Learning**: Baselines are automatically refreshed based on configurable intervals to adapt to evolving systems

### Few-Shot Learning for Error Extraction

The system includes curated examples covering common Airflow error patterns:

- **Database Connection Errors**: PostgreSQL, MySQL connection failures
- **Data Quality Issues**: Empty files, schema mismatches, validation failures
- **Dependency Failures**: Upstream task failures, service unavailability
- **Timeout Errors**: Query timeouts, connection timeouts, deadlocks
- **Permission Errors**: S3 access denied, database permission issues
- **Resource Errors**: Memory limits, disk space, connection pools

These examples help LLMs provide more accurate error categorization and confidence scores.

### Multi-Provider LLM Support

- **Local Models** (Ollama): Privacy-focused, no external API calls, cost-effective
- **Cloud Models** (OpenAI, Anthropic, Gemini): Higher accuracy, latest models, requires API keys
- **Provider-Specific Optimizations**: Customized prompts and parameters per provider
- **Fallback Mechanisms**: Heuristic error extraction when LLM fails

---

## 📊 Usage

### Command-Line Interface (CLI)

DAGnostics provides a CLI for managing the monitoring and reporting system. Use the following commands:

#### Start the Web Dashboard

```bash
# Launch the interactive web dashboard
uv run dagnostics web

# Custom host and port
uv run dagnostics web --host 0.0.0.0 --port 8080

# Enable auto-reload for development
uv run dagnostics web --reload --log-level debug
```

The web dashboard provides:
- **Real-time monitoring** with live error updates
- **Interactive analysis** with manual task analysis
- **Training dataset management** for ML model improvement
- **Error trends and analytics** with visual charts
- **WebSocket integration** for instant notifications

#### Start Monitoring Daemon

```bash
uv run dagnostics start
```

_Note: The monitoring daemon is not yet implemented. This command is a placeholder._

#### Analyze a Specific Task Failure

```bash
uv run dagnostics analyze <dag-id> <task-id> <run-id> <try-number>
```

- Options:
  - `--llm`/`-l`: LLM provider (`ollama`, `openai`, `anthropic`, `gemini`)
  - `--format`/`-f`: Output format (`json`, `yaml`, `text`)
  - `--verbose`/`-v`: Verbose output
  - `--baseline`: Use baseline comparison for anomaly detection

### Monitor DAGs (Daemon Service)

```bash
# Start the monitoring daemon
uv run dagnostics daemon start

# Stop the daemon
uv run dagnostics daemon stop

# Check daemon status
uv run dagnostics daemon status
```

### Baseline Management

```bash
# Create baseline for a specific DAG task
uv run dagnostics baseline create <dag-id> <task-id>

# List existing baselines
uv run dagnostics baseline list

# Refresh stale baselines
uv run dagnostics baseline refresh
```

#### Generate a Report (Not Yet Implemented)

```bash
uv run dagnostics report
uv run dagnostics report --daily
```

_Note: Report generation and export are not yet implemented. These commands are placeholders._

### Python API

```python
# LLM Engine Usage
from dagnostics.llm.engine import LLMEngine, OllamaProvider
from dagnostics.core.config import load_config
from dagnostics.core.models import LogEntry

# Load configuration with custom prompts
config = load_config()

# Initialize LLM engine with config
provider = OllamaProvider()
engine = LLMEngine(provider, config=config)

# Analyze log entries (few-shot learning applied automatically)
log_entries = [LogEntry(...)]
analysis = engine.extract_error_message(log_entries)
print(f"Error: {analysis.error_message}")
print(f"Category: {analysis.category}")
print(f"Confidence: {analysis.confidence}")

# Baseline Management
from dagnostics.clustering.log_clusterer import LogClusterer

clusterer = LogClusterer(config)
baseline_clusters = clusterer.build_baseline_clusters(successful_logs, dag_id, task_id)
anomalous_logs = clusterer.identify_anomalous_patterns(failed_logs, dag_id, task_id)
```

### REST API & WebSocket Features

DAGnostics v0.5.0 includes a comprehensive REST API with real-time WebSocket capabilities:

#### API Endpoints

```bash
# Start the API server
uv run dagnostics web --host 0.0.0.0 --port 8000

# API Documentation available at:
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

**Key API Routes:**
- **Analysis**: `/api/v1/analysis/analyze` - Analyze task failures
- **Dashboard**: `/api/v1/dashboard/stats` - Get dashboard statistics
- **Monitor**: `/api/v1/monitor/status` - Monitor service status
- **Training**: `/api/v1/training/candidates` - Manage training datasets

#### WebSocket Real-time Updates

```javascript
// Connect to WebSocket for live updates
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);

    switch(update.type) {
        case 'analysis_complete':
            console.log('Analysis completed:', update.data);
            break;
        case 'new_failure':
            console.log('New failure detected:', update.data);
            break;
        case 'status_change':
            console.log('Status changed:', update.data);
            break;
    }
};
```

#### Training Dataset API

```bash
# Get training candidates
curl http://localhost:8000/api/v1/training/candidates

# Submit human feedback
curl -X POST http://localhost:8000/api/v1/training/candidates/{id}/feedback \
  -H "Content-Type: application/json" \
  -d '{"action": "approve", "reviewer_name": "analyst"}'

# Export dataset
curl -X POST http://localhost:8000/api/v1/training/export \
  -H "Content-Type: application/json" \
  -d '{"format": "jsonl", "include_rejected": false}'
```

---

## 🛠 Development Tasks

The `tasks/` folder contains utility scripts for common development tasks, such as setting up the environment, linting, formatting, and running tests. These tasks are powered by [Invoke](http://www.pyinvoke.org/).

### Available Tasks

Run the following commands from the root of the project:

| Command                   | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `invoke dev.setup`        | Set up the development environment.              |
| `invoke dev.clean`        | Clean build artifacts and temporary files.       |
| `invoke dev.format`       | Format the code using `black` and `isort`.       |
| `invoke dev.lint`         | Lint the code using `flake8` and `mypy`.         |
| `invoke dev.test`         | Run all tests with `pytest`.                     |

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=dagnostics

# Run specific test file
uv run pytest tests/llm/test_parser.py
```

---

## 📝 Development

1.  Create a new branch:

```bash
git checkout -b feature/amazing-feature
```

2.  Make your changes and ensure tests pass:

```bash
./scripts/test.sh
```

3.  Format and lint your code:

```bash
./scripts/lint.sh
```

4.  Commit your changes:

```bash
git commit -m "Add amazing feature"
```

---

## 🌐 Web Dashboard

A modern web dashboard UI is included in `src/dagnostics/web/`. It provides:

- Monitor status and statistics (requires backend API)
- Error trends and categories (requires backend API)
- Task analysis form (requires backend API)

_Note: The backend API endpoints for the dashboard may be incomplete or stubbed. Some dashboard features may not display real data yet._

---

## 🚧 Limitations / Roadmap

### ✅ Implemented Features

- ✅ **LLM Integration**: Multi-provider support (Ollama, OpenAI, Anthropic, Gemini) with provider-specific optimizations
- ✅ **Smart Baselines**: Drain3-based log clustering with anomaly detection
- ✅ **Configurable Prompts**: Few-shot learning system with customizable templates
- ✅ **Daemon Service**: Background monitoring with configurable intervals
- ✅ **CLI Interface**: Comprehensive command-line tools for analysis and management
- ✅ **Alerting**: SMS/Email notifications with concise error summaries
- ✅ **Python API**: Core analysis and baseline management APIs

### 🚧 In Progress / Roadmap

- **Report generation and export:** HTML, JSON, PDF report formats (stub implementation)
- **Advanced Analytics**: Trend analysis, error correlation, predictive insights
- **Web Dashboard Backend**: Complete REST API endpoints for dashboard functionality
- **Integration Plugins**: Connectors for popular monitoring tools (Datadog, Grafana, etc.)
- **Advanced Filtering**: ML-based log filtering and noise reduction
- **Auto-scaling Training**: Distributed training across multiple GPUs/machines
- **Model Registry**: Version control and management for fine-tuned models

See [CONTRIBUTING.md](docs/contributing.md) for how to help!

---

## 🤝 Contributing

See [CONTRIBUTING.md](https://www.google.com/search?q=docs/contributing.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by the daily L1 support rotation practice and the need for intelligent error analysis
- Built with modern Python ecosystem: **uv**, FastAPI, Typer, Pydantic
- **LLM Integration**: Ollama (local), OpenAI, Anthropic Claude, Google Gemini
- **Log Analysis**: Drain3 for intelligent log clustering and pattern recognition
- **Few-Shot Learning**: Curated examples for improved error extraction
- Special thanks to the open-source community and enterprise ETL teams who inspired this project

---

## 📞 Support

For questions and support, please open an issue in the GitHub repository.
