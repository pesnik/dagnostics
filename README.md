# DAGnostics 🔍

DAGnostics is an intelligent ETL monitoring system that leverages LLMs to analyze, categorize, and report DAG failures in data pipelines. It provides automated parsing of DAG errors and generates comprehensive statistics for better observability.

## 🌟 Features

- Automated DAG error log parsing and categorization
- Daily statistics and trend analysis
- Error pattern recognition using LLMs
- Customizable reporting formats
- Integration with existing ETL monitoring systems

## 🛠 Tech Stack

- Python 3.10+
- Poetry for dependency management
- Ollama for local LLM deployment
- LangChain for LLM orchestration
- FastAPI for API endpoints
- Typer for CLI interface

## 📋 Prerequisites

- Python 3.10 or higher
- Poetry installed on your system
- Ollama installed and running locally
- Access to your ETL system's logs

## 🚀 Quick Start

1. Navigate to the project and install dependencies:
```bash
cd dagnostics
poetry install
```

2. Set up pre-commit hooks:
```bash
poetry shell
pre-commit install
```

3. Set up Ollama with your preferred model:
```bash
ollama pull mistral
```

4. Configure your environment:
```bash
cp config/config.yaml.example config/config.yaml
```

## 📁 Project Structure

```
dagnostics/
├── data/                   # Directory for pre-collected data (added)
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files
│   └── README.md           # Documentation for the data
├── src/
│   └── dagnostics/
│       ├── core/           # Core functionality and config
│       ├── llm/            # LLM integration and parsing
│       ├── monitoring/     # Log collection and analysis
│       ├── reporting/      # Report generation
│       ├── utils/          # Shared utilities
│       └── cli/            # Command-line interface
├── tests/                  # Test suite
├── config/                 # Configuration files
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── tasks/                  # Development tasks (Invoke)
├── .github/                # GitHub Actions
├── pyproject.toml          # Project metadata and dependencies
└── README.md
```

## 🔧 Configuration

The application is configured through `config/config.yaml`:

## 📊 Usage

### Command-Line Interface (CLI)

DAGnostics provides a CLI for managing the monitoring and reporting system. Use the following commands:

#### Start the System
```bash
poetry run dagnostics start
```

#### Analyze a Specific DAG
```bash
poetry run dagnostics analyze <dag-name>
```

#### Generate a Report
```bash
# Generate a standard report
poetry run dagnostics report

# Generate a daily report
poetry run dagnostics report --daily
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

## 🛠 Development Tasks

The `tasks/` folder contains utility scripts for common development tasks, such as setting up the environment, linting, formatting, and running tests. These tasks are powered by [Invoke](http://www.pyinvoke.org/).

### Available Tasks

Run the following commands from the root of the project:

| Command                  | Description                                      |
|--------------------------|--------------------------------------------------|
| `invoke dev.setup`       | Set up the development environment.             |
| `invoke dev.clean`       | Clean build artifacts and temporary files.      |
| `invoke dev.format`      | Format the code using `black` and `isort`.      |
| `invoke dev.lint`        | Lint the code using `flake8` and `mypy`.        |
| `invoke dev.test`        | Run all tests with `pytest`.                    |

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=dagnostics

# Run specific test file
poetry run pytest tests/llm/test_parser.py
```

## 📝 Development

1. Create a new branch:
```bash
git checkout -b feature/amazing-feature
```

2. Make your changes and ensure tests pass:
```bash
./scripts/test.sh
```

3. Format and lint your code:
```bash
./scripts/lint.sh
```

4. Commit your changes:
```bash
git commit -m "Add amazing feature"
```

## 🤝 Contributing

See [CONTRIBUTING.md](docs/contributing.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the daily L1 support rotation practice
- Built with Python, Poetry, Ollama, and LangChain
- Special thanks to the open-source community

## 📞 Support

For questions and support, please open an issue in the GitHub repository.
