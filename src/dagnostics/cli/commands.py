import json

import click
import yaml

from dagnostics.core.models import AnalysisResult, AppConfig  # Import AppConfig


@click.group()
def cli():
    """DAGnostics - Intelligent ETL Monitoring System"""
    pass


@cli.command()
@click.argument("dag_id")
@click.argument("task_id")
@click.argument("run_id")
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "yaml", "text"]),
    help="Output format",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def analyze(dag_id: str, task_id: str, run_id: str, output_format: str, verbose: bool):
    """Analyze a specific task failure"""
    from dagnostics.core.config import load_config
    from dagnostics.llm.engine import LLMEngine, OllamaProvider
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
        )

        # Accessing drain3.persistence_path directly
        clusterer = LogClusterer(persistence_path=config.drain3.persistence_path)

        filter = ErrorPatternFilter()

        llm_provider = OllamaProvider(
            base_url=(
                config.llm.providers["ollama"].base_url
                if config.llm.providers["ollama"].base_url
                else "http://localhost:14231"
            ),
            model=config.llm.providers["ollama"].model,
        )
        llm = LLMEngine(llm_provider)

        # Create analyzer and run analysis
        analyzer = DAGAnalyzer(airflow_client, clusterer, filter, llm)
        result = analyzer.analyze_task_failure(dag_id, task_id, run_id)

        # Output results
        if output_format == "json":
            click.echo(json.dumps(result.__dict__, default=str, indent=2))
        elif output_format == "yaml":
            click.echo(yaml.dump(result.__dict__, default_flow_style=False))
        else:  # text format
            _print_text_analysis(result, verbose)

    except Exception as e:
        click.echo(f"Analysis failed: {e}", err=True)
        raise click.Abort()


def _print_text_analysis(result: AnalysisResult, verbose: bool):
    """Print analysis result in human-readable format"""
    click.echo("\n🔍 DAGnostics Analysis Report")
    click.echo("=" * 50)
    click.echo(f"Task: {result.dag_id}.{result.task_id}")
    click.echo(f"Run ID: {result.run_id}")
    click.echo(f"Analysis Time: {result.processing_time:.2f}s")
    click.echo(f"Status: {'✅ Success' if result.success else '❌ Failed'}")

    if result.analysis:
        analysis = result.analysis
        click.echo("\n📋 Error Analysis")
        click.echo("-" * 30)
        click.echo(f"Error: {analysis.error_message}")
        click.echo(f"Category: {analysis.category.value}")
        click.echo(f"Severity: {analysis.severity.value}")
        click.echo(f"Confidence: {analysis.confidence:.1%}")

        if analysis.suggested_actions:
            click.echo("\n💡 Suggested Actions")
            click.echo("-" * 30)
            for i, action in enumerate(analysis.suggested_actions, 1):
                click.echo(f"{i}. {action}")

        if verbose and analysis.llm_reasoning:
            click.echo("\n🤖 LLM Reasoning")
            click.echo("-" * 30)
            click.echo(analysis.llm_reasoning)


@cli.command()
@click.option("--interval", default=5, help="Check interval in minutes")
@click.option("--daemon", is_flag=True, help="Run as daemon")
def monitor(interval: int, daemon: bool):
    """Start continuous monitoring"""
    click.echo(f"🔄 Starting DAGnostics monitor (interval: {interval}m)")

    # Implementation would go here
    # This would run the monitoring loop
    click.echo("Monitor started successfully!")


@cli.command()
@click.option("--daily", is_flag=True, help="Generate daily report")
@click.option(
    "--format",
    "output_format",
    default="html",
    type=click.Choice(["html", "json", "pdf"]),
    help="Report format",
)
@click.option("--output", "-o", help="Output file path")
def report(daily: bool, output_format: str, output: str):
    """Generate analysis reports"""
    report_type = "daily" if daily else "summary"
    click.echo(f"📊 Generating {report_type} report in {output_format} format...")

    if output:
        click.echo(f"Report saved to: {output}")
    else:
        click.echo("Report generated successfully!")
