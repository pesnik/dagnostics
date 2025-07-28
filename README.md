# DAGnostics 🔍

DAGnostics is an intelligent ETL monitoring system that leverages LLMs to analyze, categorize, and report DAG failures in data pipelines. It provides automated parsing of DAG errors and is designed to generate comprehensive statistics for better observability.

## 🌟 Features (Current Implementation)

- Automated DAG error log parsing and categorization using LLMs (Ollama, OpenAI, Anthropic, Gemini)
- Error pattern recognition and log clustering
- Airflow integration for log collection
- Web dashboard UI for monitoring (backend API may be incomplete)
- CLI for analysis and monitoring commands

**Planned / Not Yet Implemented:**

- Report generation and export (HTML, JSON, etc.)
- Monitoring daemon (background process)
- Alerting (email/SMS)
- Full integration with existing ETL monitoring systems

---

## 🛠 Tech Stack

- Python 3.10+
- **uv** for dependency management
- Ollama for local LLM deployment (default, fully integrated)
- OpenAI, Anthropic, Gemini LLM support (requires configuration)
- FastAPI for API endpoints
- Typer for CLI interface

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
```

---

## 📁 Project Structure

```
dagnostics/
├── data/
│   ├── clusters/              # Drain3 cluster persistence
│   ├── baselines/            # Baseline cluster data
│   ├── raw/
│   └── processed/
├── src/dagnostics/
│   ├── api/                  # FastAPI application
│   ├── core/                 # Data models and configuration
│   ├── llm/                  # LLM engine and providers
│   ├── monitoring/           # Airflow integration and analysis
│   ├── reporting/            # (Stub) Reporting logic
│   ├── web/                  # Web dashboard UI
│   └── utils/
├── config/
└── migrations/
```

---

## 🔧 Configuration

The application is configured through `config/config.yaml`.

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

#### Generate a Report (Not Yet Implemented)

```bash
uv run dagnostics report
uv run dagnostics report --daily
```

_Note: Report generation and export are not yet implemented. These commands are placeholders._

### Python API (Planned)

```python
# Example usage (not yet implemented)
from dagnostics.monitoring import DAGMonitor
from dagnostics.reporting import ReportGenerator

monitor = DAGMonitor()
generator = ReportGenerator()
report = generator.create_daily_report()
```

_Note: The Python API for monitoring and reporting is not yet implemented._

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

- **Report generation and export:** Not yet implemented. No HTML, JSON, or other report files are produced.
- **Monitoring daemon:** The background monitoring process is a stub.
- **Alerting:** Email/SMS alerting is not implemented.
- **Python API:** Not yet implemented.
- **Web dashboard:** UI is present, but backend data may be incomplete.
- **LLM providers:** Only Ollama is fully integrated by default. OpenAI, Anthropic, and Gemini require additional setup and may not be fully tested.

See [CONTRIBUTING.md](docs/contributing.md) for how to help!

---

## 🤝 Contributing

See [CONTRIBUTING.md](https://www.google.com/search?q=docs/contributing.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by the daily L1 support rotation practice
- Built with Python, **uv**, Ollama, and LangChain
- Special thanks to the open-source community

---

## 📞 Support

For questions and support, please open an issue in the GitHub repository.
