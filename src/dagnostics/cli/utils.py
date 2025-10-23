"""
Reusable utilities for CLI commands
"""

import json
from datetime import datetime
from typing import List, Optional, Tuple, Union

import typer

from dagnostics.analysis.analyzer import DAGAnalyzer
from dagnostics.clustering.log_clusterer import LogClusterer
from dagnostics.core.airflow_client import AirflowClient
from dagnostics.core.config import load_config
from dagnostics.core.models import (
    AppConfig,
    GeminiLLMConfig,
    LogEntry,
    OllamaLLMConfig,
    OpenAILLMConfig,
)
from dagnostics.heuristics.filter_factory import FilterFactory
from dagnostics.llm.engine import (
    GeminiProvider,
    LLMEngine,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""

    def default(self, o):  # Changed from 'obj' to 'o'
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def initialize_components(
    config_file: Optional[str], llm_provider: str
) -> Tuple[AppConfig, DAGAnalyzer]:
    """Initialize common components used across CLI commands"""
    # Load configuration
    config = load_config(config_file)

    # Initialize Airflow client
    airflow_client = AirflowClient(
        base_url=config.airflow.base_url,
        username=config.airflow.username,
        password=config.airflow.password,
        db_connection=config.airflow.database_url,
        verify_ssl=False,
        db_timezone_offset=getattr(config.airflow, "db_timezone_offset", "+00:00"),
    )

    # Initialize clusterer
    clusterer = LogClusterer(
        persistence_path=config.drain3.persistence_path,
        app_config=config,
        config_path=config.drain3.config_path,
    )

    # Initialize filter
    filter = FilterFactory.create_for_notifications(config)

    # Initialize LLM
    llm_provider_instance = initialize_llm_provider(config, llm_provider)
    llm = LLMEngine(llm_provider_instance)

    # Create analyzer
    analyzer = DAGAnalyzer(airflow_client, clusterer, filter, llm, config)

    return config, analyzer


def initialize_components_for_notifications(
    config_file: Optional[str], llm_provider: str
) -> Tuple[AppConfig, DAGAnalyzer]:
    """Initialize components specifically for notification commands with proper filter"""
    # Load configuration
    config = load_config(config_file)

    # Initialize Airflow client
    airflow_client = AirflowClient(
        base_url=config.airflow.base_url,
        username=config.airflow.username,
        password=config.airflow.password,
        db_connection=config.airflow.database_url,
        verify_ssl=False,
        db_timezone_offset=getattr(config.airflow, "db_timezone_offset", "+00:00"),
    )

    # Initialize clusterer based on config
    if config.monitoring.baseline_usage == "stored":
        clusterer = LogClusterer(
            persistence_path=config.drain3.persistence_path,
            app_config=config,
            config_path=config.drain3.config_path,
        )
    else:
        clusterer = LogClusterer(
            app_config=config,
            config_path=config.drain3.config_path,
        )

    # Use FilterFactory for notifications
    filter = FilterFactory.create_for_notifications(config)

    # Initialize LLM
    llm_provider_instance = initialize_llm_provider(config, llm_provider)
    llm = LLMEngine(llm_provider_instance)

    # Create analyzer
    analyzer = DAGAnalyzer(airflow_client, clusterer, filter, llm, config)

    return config, analyzer


def initialize_llm_provider(config: AppConfig, llm_provider: str) -> LLMProvider:
    """Initialize LLM provider based on selection"""
    llm_provider_instance: Union[
        OllamaProvider, OpenAIProvider, GeminiProvider, LLMProvider, None
    ] = None

    if llm_provider == "ollama":
        ollama_config = config.llm.providers.get("ollama")
        if not ollama_config or not isinstance(ollama_config, OllamaLLMConfig):
            typer.echo(
                "Error: Ollama LLM configuration not found or invalid.", err=True
            )
            raise typer.Exit(code=1)
        llm_provider_instance = OllamaProvider(
            base_url=ollama_config.base_url or "http://localhost:11434",
            model=ollama_config.model,
            timeout=ollama_config.timeout,
        )

    elif llm_provider == "openai":
        openai_config = config.llm.providers.get("openai")
        if not openai_config or not isinstance(openai_config, OpenAILLMConfig):
            typer.echo(
                "Error: OpenAI LLM configuration not found or invalid.", err=True
            )
            raise typer.Exit(code=1)
        llm_provider_instance = OpenAIProvider(
            api_key=openai_config.api_key,
            model=openai_config.model,
        )

    elif llm_provider == "gemini":
        gemini_config = config.llm.providers.get("gemini")
        if not gemini_config or not isinstance(gemini_config, GeminiLLMConfig):
            typer.echo(
                "Error: Gemini LLM configuration not found or invalid.", err=True
            )
            raise typer.Exit(code=1)
        llm_provider_instance = GeminiProvider(
            api_key=gemini_config.api_key,
            model=gemini_config.model,
        )
    else:
        typer.echo(f"Error: Unknown LLM provider '{llm_provider}'", err=True)
        raise typer.Exit(code=1)

    return llm_provider_instance


def get_error_message(
    dag_id: str,
    task_id: str,
    run_id: str,
    try_number: int,
    config_file: Optional[str] = None,
    llm_provider: str = "ollama",
) -> tuple[str, Optional[List[LogEntry]], str]:
    """
    Internal function to get error message - can be used by CLI commands and other functions
    """
    try:
        _, analyzer = initialize_components(config_file, llm_provider)
        error_message, error_candidates, error_line = (
            analyzer.extract_task_error_for_sms(dag_id, task_id, run_id, try_number)
        )
        return error_message, error_candidates, error_line
    except Exception as e:
        return f"Error extraction failed: {e}", None, ""


def get_error_candidates(
    dag_id: str,
    task_id: str,
    run_id: str,
    try_number: int,
    config_file: Optional[str] = None,
    llm_provider: str = "ollama",
) -> List[LogEntry] | str:
    """
    Internal function to get error message - can be used by CLI commands and other functions
    """
    try:
        _, analyzer = initialize_components(config_file, llm_provider)
        error_candidates = analyzer.extract_error_candidates(
            dag_id, task_id, run_id, try_number
        )
        return error_candidates
    except Exception as e:
        return f"Error extraction failed: {e}"
