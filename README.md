# DAGnostics 🔍

DAGnostics is an intelligent ETL monitoring system that leverages LLMs to analyze, categorize, and report DAG failures in data pipelines. It provides automated parsing of DAG errors and generates comprehensive statistics for better observability.

## 🌟 Features

  - Automated DAG error log parsing and categorization
  - Daily statistics and trend analysis
  - Error pattern recognition using LLMs
  - Customizable reporting formats
  - Integration with existing ETL monitoring systems

-----

## 🛠 Tech Stack

  - Python 3.10+
  - **uv** for dependency management
  - Ollama for local LLM deployment
  - LangChain for LLM orchestration
  - FastAPI for API endpoints
  - Typer for CLI interface

-----

## 📋 Prerequisites

  - Python 3.10 or higher
  - **uv** installed on your system (`pip install uv`)
  - Ollama installed and running locally
  - Access to your ETL system's logs

-----

## 🚀 Quick Start

1.  Navigate to the project and install dependencies:

<!-- end list -->

```bash
cd dagnostics
uv sync
```

2.  Set up pre-commit hooks:

<!-- end list -->

```bash
uv run pre-commit install
```

3.  Set up Ollama with your preferred model:

<!-- end list -->

```bash
ollama pull mistral
```

4.  Configure your environment:

<!-- end list -->

```bash
cp config/config.yaml.example config/config.yaml
```

-----

## 📁 Project Structure

```
dagnostics/
├── data/
│   ├── clusters/              # Drain3 cluster persistence
│   ├── baselines/            # Baseline cluster data
│   ├── raw/
│   └── processed/
├── src/dagnostics/
│   ├── api/                  # FastAPI application (NEW)
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   └── schemas.py
│   ├── core/
│   │   ├── models.py         # Enhanced with new data models
│   │   └── config.py         # Extended configuration
│   ├── llm/
│   │   ├── log_clusterer.py  # Drain3 integration (NEW)
│   │   ├── pattern_filter.py # Error pattern filtering (NEW)
│   │   ├── engine.py         # Enhanced LLM engine
│   │   ├── providers/        # LLM provider implementations (NEW)
│   │   └── prompts.py
│   ├── monitoring/
│   │   ├── airflow_client.py # Airflow integration (NEW)
│   │   ├── collector.py      # Enhanced log collection
│   │   ├── analyzer.py       # Main analysis logic (NEW)
│   │   ├── monitor.py        # Continuous monitoring (NEW)
│   │   └── alert.py
│   ├── reporting/
│   │   ├── generator.py      # Enhanced reporting
│   │   ├── exporters.py      # Multiple export formats (NEW)
│   │   └── templates/
│   ├── web/                  # Web dashboard (NEW)
│   │   ├── static/
│   │   ├── templates/
│   │   └── app.py
│   └── utils/
├── config/
│   ├── drain3_config.yaml    # Drain3 configuration (NEW)
│   ├── llm_providers.yaml    # LLM provider configs (NEW)
│   └── monitoring.yaml       # Monitoring settings (NEW)
└── migrations/               # Database migrations (NEW)
```

-----

## 🔧 Configuration

The application is configured through `config/config.yaml`.

-----

## 📊 Usage

### Command-Line Interface (CLI)

DAGnostics provides a CLI for managing the monitoring and reporting system. Use the following commands:

#### Start the System

```bash
uv run dagnostics start
```

#### Analyze a Specific DAG

```bash
uv run dagnostics analyze <dag-name>
```

#### Generate a Report

```bash
# Generate a standard report
uv run dagnostics report

# Generate a daily report
uv run dagnostics report --daily
```

### Python API

```python
from dagnostics.monitoring import DAGMonitor
from dagnostics.reporting import ReportGenerator

# Initialize monitor
monitor = DAGMonitor()

# Generate report
generator = ReportGenerator()
report = generator.create_daily_report()
```

-----

## 🛠 Development Tasks

The `tasks/` folder contains utility scripts for common development tasks, such as setting up the environment, linting, formatting, and running tests. These tasks are powered by [Invoke](http://www.pyinvoke.org/).

### Available Tasks

Run the following commands from the root of the project:

| Command                  | Description                                      |
|--------------------------|--------------------------------------------------|
| `invoke dev.setup`       | Set up the development environment.             |
| `invoke dev.clean`       | Clean build artifacts and temporary files.      |
| `invoke dev.format`      | Format the code using `black` and `isort`.      |
| `invoke dev.lint`        | Lint the code using `flake8` and `mypy`.        |
| `invoke dev.test`        | Run all tests with `pytest`.                    |

-----

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=dagnostics

# Run specific test file
uv run pytest tests/llm/test_parser.py
```

-----

## 📝 Development

1.  Create a new branch:

<!-- end list -->

```bash
git checkout -b feature/amazing-feature
```

2.  Make your changes and ensure tests pass:

<!-- end list -->

```bash
./scripts/test.sh
```

3.  Format and lint your code:

<!-- end list -->

```bash
./scripts/lint.sh
```

4.  Commit your changes:

<!-- end list -->

```bash
git commit -m "Add amazing feature"
```

-----

## 🤝 Contributing

See [CONTRIBUTING.md](https://www.google.com/search?q=docs/contributing.md) for detailed guidelines.

-----

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## 🙏 Acknowledgments

  - Inspired by the daily L1 support rotation practice
  - Built with Python, **uv**, Ollama, and LangChain
  - Special thanks to the open-source community

-----

## 📞 Support

For questions and support, please open an issue in the GitHub repository.
