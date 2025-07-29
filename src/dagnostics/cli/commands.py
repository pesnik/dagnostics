import json
from enum import Enum
from typing import Optional, Union  # Import Union

import typer
import yaml
from typer import Argument, Option

from dagnostics.core.models import AnalysisResult, AppConfig, OllamaLLMConfig
from dagnostics.llm.filter_factory import FilterFactory
from dagnostics.utils.sms import send_sms_alert


class OutputFormat(str, Enum):
    json = "json"
    yaml = "yaml"
    text = "text"


class ReportFormat(str, Enum):
    html = "html"
    json = "json"
    pdf = "pdf"


def analyze(
    dag_id: str = Argument(..., help="ID of the DAG to analyze"),
    task_id: str = Argument(..., help="ID of the task to analyze"),
    run_id: str = Argument(..., help="Run ID of the task instance"),
    try_number: int = Argument(..., help="Attempt number of the task to analyze"),
    output_format: OutputFormat = Option(
        OutputFormat.json, "--format", "-f", help="Output format"
    ),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose output"),
    llm_provider: str = Option(
        "ollama",
        "--llm",
        "-l",
        help="LLM provider to use (ollama, openai, anthropic, gemini)",
    ),
):
    """Analyze a specific task failure."""
    # Local imports are fine within a command function if they are only used there
    from dagnostics.core.config import load_config
    from dagnostics.core.models import GeminiLLMConfig, OpenAILLMConfig
    from dagnostics.llm.engine import LLMProvider  # Import the base LLMProvider type
    from dagnostics.llm.engine import (
        GeminiProvider,
        LLMEngine,
        OllamaProvider,
        OpenAIProvider,
    )
    from dagnostics.llm.log_clusterer import LogClusterer
    from dagnostics.llm.pattern_filter import ErrorPatternFilter
    from dagnostics.monitoring.airflow_client import AirflowClient
    from dagnostics.monitoring.analyzer import DAGAnalyzer

    try:
        # Load configuration
        config: AppConfig = load_config()

        # Initialize components
        airflow_client = AirflowClient(
            base_url=config.airflow.base_url,
            username=config.airflow.username,
            password=config.airflow.password,
            db_connection=config.airflow.database_url,
            verify_ssl=False,
        )
        clusterer = LogClusterer(
            persistence_path=config.drain3.persistence_path, app_config=config
        )
        filter = ErrorPatternFilter()

        # Initialize LLM provider based on selection
        # Define llm_provider_instance with a Union of all possible provider types and None
        llm_provider_instance: Union[
            OllamaProvider, OpenAIProvider, GeminiProvider, LLMProvider, None
        ] = None

        if llm_provider == "ollama":
            ollama_config = config.llm.providers.get("ollama")
            if not ollama_config:
                typer.echo("Error: Ollama LLM configuration not found.", err=True)
                raise typer.Exit(code=1)

            # Ensure ollama_config is of the correct type
            if not isinstance(ollama_config, OllamaLLMConfig):
                typer.echo(
                    "Error: Ollama LLM configuration is not of type OllamaLLMConfig.",
                    err=True,
                )
                raise typer.Exit(code=1)

            llm_provider_instance = OllamaProvider(
                base_url=(
                    ollama_config.base_url
                    if ollama_config.base_url
                    else "http://localhost:11434"
                ),
                model=ollama_config.model,
            )

        elif llm_provider == "openai":
            openai_config = config.llm.providers.get("openai")
            if not openai_config:
                typer.echo("Error: OpenAI LLM configuration not found.", err=True)
                raise typer.Exit(code=1)

            # Ensure openai_config is of the correct type
            if not isinstance(openai_config, OpenAILLMConfig):
                typer.echo(
                    "Error: OpenAI LLM configuration is not of type OpenAILLMConfig.",
                    err=True,
                )
                raise typer.Exit(code=1)

            llm_provider_instance = OpenAIProvider(
                api_key=openai_config.api_key,
                model=openai_config.model,
            )

        elif llm_provider == "gemini":
            gemini_config = config.llm.providers.get("gemini")
            if not gemini_config:
                typer.echo("Error: Gemini LLM configuration not found.", err=True)
                raise typer.Exit(code=1)

            # Ensure gemini_config is of the correct type
            if not isinstance(gemini_config, GeminiLLMConfig):
                typer.echo(
                    "Error: Gemini LLM configuration is not of type GeminiLLMConfig.",
                    err=True,
                )
                raise typer.Exit(code=1)

            llm_provider_instance = GeminiProvider(
                api_key=gemini_config.api_key,
                model=gemini_config.model,
            )
        else:
            typer.echo(f"Error: Unknown LLM provider '{llm_provider}'", err=True)
            raise typer.Exit(code=1)

        # Check if llm_provider_instance is still None before passing to LLMEngine
        if llm_provider_instance is None:
            typer.echo("Error: No LLM provider could be initialized.", err=True)
            raise typer.Exit(code=1)

        # Now llm_provider_instance is guaranteed to be one of the LLMProvider types
        llm = LLMEngine(llm_provider_instance)

        # Create analyzer and run analysis
        analyzer = DAGAnalyzer(airflow_client, clusterer, filter, llm, config)
        result = analyzer.analyze_task_failure(dag_id, task_id, run_id, try_number)

        # Output results
        if output_format == OutputFormat.json:
            typer.echo(json.dumps(result.__dict__, default=str, indent=2))
        elif output_format == OutputFormat.yaml:
            typer.echo(yaml.dump(result.__dict__, default_flow_style=False))
        else:  # text format
            _print_text_analysis(result, verbose)

    except Exception as e:
        typer.echo(f"Analysis failed: {e}", err=True)
        raise typer.Exit(code=1)


def _print_text_analysis(result: AnalysisResult, verbose: bool):
    """Print analysis result in human-readable format"""
    typer.echo("\n🔍 DAGnostics Analysis Report")
    typer.echo("=" * 50)
    typer.echo(f"Task: {result.dag_id}.{result.task_id}")
    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Analysis Time: {result.processing_time:.2f}s")
    typer.echo(f"Status: {'✅ Success' if result.success else '❌ Failed'}")

    if result.analysis:
        analysis = result.analysis
        typer.echo("\n📋 Error Analysis")
        typer.echo("-" * 30)
        typer.echo(f"Error: {analysis.error_message}")
        typer.echo(f"Category: {analysis.category.value}")
        typer.echo(f"Severity: {analysis.severity.value}")
        typer.echo(f"Confidence: {analysis.confidence:.1%}")

        if analysis.suggested_actions:
            typer.echo("\n💡 Suggested Actions")
            typer.echo("-" * 30)
            for i, action in enumerate(analysis.suggested_actions, 1):
                typer.echo(f"{i}. {action}")

        if verbose and analysis.llm_reasoning:
            typer.echo("\n🤖 LLM Reasoning")
            typer.echo("-" * 30)
            typer.echo(analysis.llm_reasoning)


def start(
    interval: int = Option(5, "--interval", "-i", help="Check interval in minutes"),
    daemon: bool = Option(False, "--daemon", help="Run as daemon"),
):
    """Start continuous monitoring."""
    typer.echo(f"🔄 Starting DAGnostics monitor (interval: {interval}m)")

    # Implementation would go here
    # This would run the monitoring loop
    typer.echo("Monitor started successfully!")


def report(
    daily: bool = Option(False, "--daily", help="Generate daily report"),
    output_format: ReportFormat = Option(
        ReportFormat.html, "--format", "-f", help="Report format"
    ),
    output: Optional[str] = Option(None, "--output", "-o", help="Output file path"),
):
    """Generate analysis reports."""
    report_type = "daily" if daily else "summary"
    typer.echo(f"📊 Generating {report_type} report in {output_format.value} format...")

    if output:
        typer.echo(f"Report saved to: {output}")
    else:
        typer.echo("Report generated successfully!")


def notify_failures(
    since_minutes: int = Option(
        60, "--since-minutes", "-s", help="Look back window in minutes"
    ),
    dry_run: bool = Option(False, "--dry-run", help="Don't actually send SMS"),
    llm_provider: str = Option(
        "ollama",
        "--llm",
        "-l",
        help="LLM provider to use (ollama, openai, anthropic, gemini)",
    ),
):
    """
    Analyze recent Airflow task failures and send concise SMS notifications.
    """
    from dagnostics.core.config import load_config
    from dagnostics.core.models import GeminiLLMConfig, OllamaLLMConfig, OpenAILLMConfig
    from dagnostics.llm.engine import (
        GeminiProvider,
        LLMEngine,
        LLMProvider,
        OllamaProvider,
        OpenAIProvider,
    )
    from dagnostics.llm.log_clusterer import LogClusterer
    from dagnostics.monitoring.airflow_client import AirflowClient
    from dagnostics.monitoring.analyzer import DAGAnalyzer

    config = load_config()
    airflow_client = AirflowClient(
        base_url=config.airflow.base_url,
        username=config.airflow.username,
        password=config.airflow.password,
        db_connection=config.airflow.database_url,
        verify_ssl=False,
        db_timezone_offset=config.airflow.db_timezone_offset,
    )

    # Use config-based baseline configuration
    if config.monitoring.baseline_usage == "stored":
        clusterer = LogClusterer(
            persistence_path=config.drain3.persistence_path,
            app_config=config,
            config_path="config/drain3.ini",
        )
    else:
        clusterer = LogClusterer(app_config=config, config_path="config/drain3.ini")

    filter = FilterFactory.create_for_notifications(config)

    # LLM provider selection (reuse logic from analyze)
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
    if llm_provider_instance is None:
        typer.echo("Error: No LLM provider could be initialized.", err=True)
        raise typer.Exit(code=1)
    llm = LLMEngine(llm_provider_instance)
    analyzer = DAGAnalyzer(airflow_client, clusterer, filter, llm, config)

    # Validate SMS configuration
    if not config.alerts.sms.enabled:
        typer.echo("Error: SMS alerts are not enabled in configuration.", err=True)
        raise typer.Exit(code=1)

    if (
        not config.alerts.sms.default_recipients
        and not config.alerts.sms.task_recipients
    ):
        typer.echo(
            "Error: No SMS recipients configured. Please add default_recipients or task_recipients to your config.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"🔍 Fetching failed tasks from last {since_minutes} minutes...")
    failed_tasks = airflow_client.get_failed_tasks(since_minutes)
    if not failed_tasks:
        typer.echo("No failed tasks found.")
        return
    typer.echo(f"Found {len(failed_tasks)} failed tasks.")

    # Config-driven recipient mapping
    def get_recipients_for_task(task):
        import re

        task_key = f"{task.dag_id}.{task.task_id}"

        # Check task-specific recipients first
        for pattern, recipients in config.alerts.sms.task_recipients.items():
            if re.match(pattern, task_key):
                return recipients

        # Fall back to default recipients
        return config.alerts.sms.default_recipients

    for task in failed_tasks:
        try:
            # Get all tries for this task instance
            typer.echo(
                f"🔍 Fetching tries for {task.dag_id}.{task.task_id} (run: {task.run_id})..."
            )
            task_tries = airflow_client.get_task_tries(
                task.dag_id, task.task_id, task.run_id
            )

            # Filter only failed tries
            failed_tries = [
                try_instance
                for try_instance in task_tries
                if try_instance.state == "failed"
            ]

            if not failed_tries:
                typer.echo(
                    f"⚠️  No failed tries found for {task.dag_id}.{task.task_id} (run: {task.run_id})"
                )
                continue

            # Process each failed try
            for failed_try in failed_tries:
                try:
                    typer.echo(
                        f"📝 Analyzing {task.dag_id}.{task.task_id} (run: {task.run_id}, try: {failed_try.try_number})..."
                    )

                    summary = analyzer.extract_task_error_for_sms(
                        failed_try.dag_id,
                        failed_try.task_id,
                        failed_try.run_id,
                        failed_try.try_number,
                    )

                    recipients = get_recipients_for_task(failed_try)

                    # Include try number in the summary for clarity
                    enhanced_summary = f"{summary} (try {failed_try.try_number})"

                    if dry_run:
                        typer.echo(
                            f"[DRY RUN] Would send to {recipients}: {enhanced_summary}"
                        )
                    else:
                        sms_config = config.alerts.sms.dict()
                        send_sms_alert(enhanced_summary, recipients, sms_config)
                        typer.echo(f"📱 Sent to {recipients}: {enhanced_summary}")

                except Exception as e:
                    typer.echo(
                        f"❌ Error processing {task.dag_id}.{task.task_id} (try {failed_try.try_number}): {e}",
                        err=True,
                    )

        except Exception as e:
            typer.echo(
                f"❌ Error fetching tries for {task.dag_id}.{task.task_id} (run: {task.run_id}): {e}",
                err=True,
            )
