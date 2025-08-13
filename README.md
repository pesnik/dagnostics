# DAGnostics 🔍

DAGnostics is an intelligent ETL monitoring system that leverages LLMs to analyze, categorize, and report DAG failures in data pipelines. It provides automated parsing of DAG errors and is designed to generate comprehensive statistics for better observability.

## 🌟 Features (Current Implementation)

- **Intelligent Error Analysis**: Automated DAG error log parsing and categorization using multiple LLM providers (Ollama, OpenAI, Anthropic, Gemini)
- **Smart Baseline System**: Advanced error pattern recognition using Drain3 log clustering with baseline creation from successful task runs
- **Few-Shot Learning**: Configurable prompts with curated examples for improved error extraction accuracy
- **Multi-Provider LLM Support**: Seamless switching between local (Ollama) and cloud LLM providers
- **Airflow Integration**: Direct integration with Airflow API and database for real-time log collection
- **Configurable Prompts**: Customize LLM prompts without code deployment via configuration files
- **Anomaly Detection**: Identify new error patterns by comparing against successful task baselines
- **Web Dashboard UI**: Modern dashboard for monitoring (backend API may be incomplete)
- **CLI Interface**: Comprehensive command-line tools for analysis and monitoring
- **Smart Alerting**: SMS/Email notifications with concise error summaries
- **Daemon Service**: Background monitoring service for continuous error detection

**Planned / Not Yet Implemented:**

- Report generation and export (HTML, JSON, etc.)
- Full integration with existing ETL monitoring systems
- Advanced analytics and trend analysis

---

## System Architecture
![System Architecture](docs/system_architecture.png)
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
- **FastAPI** for high-performance REST API endpoints
- **Typer** for intuitive CLI interface
- **Jinja2** for web dashboard templating

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

1.  Navigate to the project and install dependencies:

```bash
cd dagnostics
uv sync
```

2.  Set up pre-commit hooks:

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

#### Start the System (Stub)

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
- **Fine-tuning Support**: Custom model fine-tuning for domain-specific error patterns
- **Integration Plugins**: Connectors for popular monitoring tools (Datadog, Grafana, etc.)
- **Advanced Filtering**: ML-based log filtering and noise reduction

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
