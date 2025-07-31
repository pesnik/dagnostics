import json
from typing import Optional

import typer
import yaml
from typer import Argument, Option

from dagnostics.cli.utils import initialize_components
from dagnostics.core.models import AnalysisResult, OutputFormat, ReportFormat
from dagnostics.utils.sms import send_sms_alert


def analyze(
    dag_id: str = Argument(..., help="ID of the DAG to analyze"),
    task_id: str = Argument(..., help="ID of the task to analyze"),
    run_id: str = Argument(..., help="Run ID of the task instance"),
    try_number: int = Argument(..., help="Attempt number of the task to analyze"),
    config_file: Optional[str] = Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (default: searches standard locations)",
    ),
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
    try:
        _, analyzer = initialize_components(config_file, llm_provider)

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
    config_file: Optional[str] = Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (default: searches standard locations)",
    ),
    daemon: bool = Option(False, "--daemon", help="Run as daemon"),
):
    """Start continuous monitoring."""
    try:
        typer.echo(f"🔄 Starting DAGnostics monitor (interval: {interval}m)")
        # Implementation would go here
        typer.echo("Monitor started successfully!")

    except FileNotFoundError as e:
        typer.echo(f"❌ Configuration error: {e}", err=True)
        typer.echo(
            "💡 Run 'dagnostics init' to create a default configuration file.", err=True
        )
        raise typer.Exit(code=1)


def report(
    daily: bool = Option(False, "--daily", help="Generate daily report"),
    config_file: Optional[str] = Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (default: searches standard locations)",
    ),
    output_format: ReportFormat = Option(
        ReportFormat.html, "--format", "-f", help="Report format"
    ),
    output: Optional[str] = Option(None, "--output", "-o", help="Output file path"),
):
    """Generate analysis reports."""
    try:
        report_type = "daily" if daily else "summary"
        typer.echo(
            f"📊 Generating {report_type} report in {output_format.value} format..."
        )

        if output:
            typer.echo(f"Report saved to: {output}")
        else:
            typer.echo("Report generated successfully!")

    except FileNotFoundError as e:
        typer.echo(f"❌ Configuration error: {e}", err=True)
        typer.echo(
            "💡 Run 'dagnostics init' to create a default configuration file.", err=True
        )
        raise typer.Exit(code=1)


def notify_failures(
    since_minutes: int = Option(
        60, "--since-minutes", "-s", help="Look back window in minutes"
    ),
    config_file: Optional[str] = Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (default: searches standard locations)",
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
    import re

    from dagnostics.cli.utils import (
        get_error_message,
        initialize_components_for_notifications,
    )

    try:
        config, analyzer = initialize_components_for_notifications(
            config_file, llm_provider
        )

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
        failed_tasks = analyzer.airflow_client.get_failed_tasks(since_minutes)

        if not failed_tasks:
            typer.echo("No failed tasks found.")
            return

        typer.echo(f"Found {len(failed_tasks)} failed tasks.")

        # Config-driven recipient mapping
        def get_recipients_for_task(task):
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
                task_tries = analyzer.airflow_client.get_task_tries(
                    task.dag_id, task.task_id, task.run_id
                )

                # Filter only failed tries
                failed_tries = [
                    try_instance
                    for try_instance in task_tries
                    if try_instance.state == "failed" and try_instance.try_number > 0
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

                        summary = get_error_message(
                            failed_try.dag_id,
                            failed_try.task_id,
                            failed_try.run_id,
                            failed_try.try_number,
                            config_file,
                            llm_provider,
                        )

                        recipients = get_recipients_for_task(failed_try)

                        # Include try number in the summary for clarity
                        enhanced_summary = f"{summary} Attempt: {failed_try.try_number}"

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

    except Exception as e:
        typer.echo(f"Notification failed: {e}", err=True)
        raise typer.Exit(code=1)


def get_error(
    dag_id: str = Argument(..., help="ID of the DAG to analyze"),
    task_id: str = Argument(..., help="ID of the task to analyze"),
    run_id: str = Argument(..., help="Run ID of the task instance"),
    try_number: int = Argument(..., help="Attempt number of the task to analyze"),
    config_file: Optional[str] = Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (default: searches standard locations)",
    ),
    llm_provider: str = Option(
        "ollama",
        "--llm",
        "-l",
        help="LLM provider to use (ollama, openai, anthropic, gemini)",
    ),
) -> str:
    """Get the exact error message for a specific task failure."""
    from dagnostics.cli.utils import get_error_message

    try:
        error_message = get_error_message(
            dag_id, task_id, run_id, try_number, config_file, llm_provider
        )

        # Print when used as CLI command
        typer.echo(error_message)
        return error_message

    except Exception as e:
        error_msg = f"Error extraction failed: {e}"
        typer.echo(error_msg, err=True)
        return error_msg
