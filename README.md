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
- FastAPI for API endpoints (optional)

## 📋 Prerequisites

- Python 3.10 or higher
- Poetry installed on your system
- Ollama installed and running locally
- Access to your ETL system's logs

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/rhasanm/dagnostics.git
cd dagnostics
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up Ollama with your preferred model:
```bash
ollama pull mistral
# or any other supported model
```

4. Configure your environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

## 🔧 Configuration

Create a `config.yaml` file in the root directory:

```yaml
llm:
  model: mistral
  temperature: 0.1
  max_tokens: 500

monitoring:
  categories:
    - system_error
    - resource_constraint
    - connectivity
    - data_availability
    - data_quality
    - syntax_error

reporting:
  format: markdown
  include_trends: true
```

## 📊 Usage

1. Start the DAGnostics service:
```bash
poetry run python -m dagnostics.main
```

2. Generate daily report:
```bash
poetry run python -m dagnostics.report
```

Example output:
```
DAGnostics Daily Report
----------------------
Total Failed DAGs: 4
Total Failed Tasks: 7
Total Attempts: 36
Failed DAG Of The Day: adtech_customer_interaction_info_campaign
Failed Categories:
- system_error: 5
- resource_constraint: 12
- connectivity: 1
- data_availability: 12
- data_quality: 1
- syntax_error: 1
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Project Structure

```
dagnostics/
├── pyproject.toml
├── README.md
├── dagnostics/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   └── prompts.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── collector.py
│   └── reporting/
│       ├── __init__.py
│       └── generator.py
├── tests/
│   └── ...
└── examples/
    └── ...
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the daily L1 support rotation practice
- Built with Python, Poetry, Ollama, and LangChain
- Special thanks to the open-source community

## 📞 Contact

For questions and support, please open an issue in the GitHub repository.
