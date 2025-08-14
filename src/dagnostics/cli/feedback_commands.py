"""
CLI Commands for Feedback Management

Commands for reviewing daily errors, managing feedback, and creating
feedback sessions for continuous model improvement.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from dagnostics.storage.feedback_db import get_feedback_db
from dagnostics.web.feedback import (
    FeedbackCollector,
    FeedbackCorrection,
    FeedbackSubmission,
)

logger = logging.getLogger(__name__)
console = Console()

# Create feedback CLI app
feedback_app = typer.Typer(
    name="feedback", help="Feedback collection and review commands"
)


@feedback_app.command("review-daily")
def review_daily_errors(
    date: Optional[str] = typer.Option(
        None, help="Date to review (YYYY-MM-DD), defaults to today"
    ),
    limit: int = typer.Option(10, help="Maximum errors to review"),
    auto_open_browser: bool = typer.Option(
        False, help="Automatically open browser for each error"
    ),
):
    """Review daily errors and provide feedback interactively"""

    if date:
        review_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        review_date = datetime.now()

    console.print(
        f"[bold blue]Reviewing errors for {review_date.strftime('%Y-%m-%d')}[/bold blue]"
    )

    # Load daily error logs and analysis
    daily_errors = load_daily_errors(review_date)

    if not daily_errors:
        console.print(
            f"[yellow]No errors found for {review_date.strftime('%Y-%m-%d')}[/yellow]"
        )
        return

    console.print(
        f"Found {len(daily_errors)} errors. Showing first {min(limit, len(daily_errors))}:"
    )

    feedback_collector = FeedbackCollector()

    for i, error_entry in enumerate(daily_errors[:limit], 1):
        console.print(
            f"\n[bold cyan]═══ Error {i}/{min(limit, len(daily_errors))} ═══[/bold cyan]"
        )

        # Display error summary
        display_error_summary(error_entry)

        # Ask if user wants to provide feedback
        if Confirm.ask("Provide feedback for this error?"):
            collect_interactive_feedback(error_entry, feedback_collector)

        # Continue or stop
        if i < min(limit, len(daily_errors)):
            if not Confirm.ask("Continue to next error?"):
                break

    console.print("\n[green]✅ Review completed![/green]")


@feedback_app.command("list-errors")
def list_recent_errors(
    days: int = typer.Option(7, help="Number of days to look back"),
    category: Optional[str] = typer.Option(None, help="Filter by error category"),
    dag_id: Optional[str] = typer.Option(None, help="Filter by DAG ID"),
    show_analyzed: bool = typer.Option(
        True, help="Show errors that have been analyzed"
    ),
    show_feedback: bool = typer.Option(True, help="Show errors that have feedback"),
):
    """List recent errors with their analysis status"""

    console.print(f"[bold blue]Recent errors (last {days} days)[/bold blue]")

    # Load error data
    errors = load_recent_errors(days, category, dag_id)
    feedback_collector = FeedbackCollector()

    # Create summary table
    table = Table(title="Error Summary")
    table.add_column("Date", style="cyan")
    table.add_column("DAG", style="green")
    table.add_column("Task", style="yellow")
    table.add_column("Category", style="magenta")
    table.add_column("Analysis", style="blue")
    table.add_column("Feedback", style="red")

    for error in errors:
        # Check if error has analysis and feedback
        has_analysis = "✅" if error.get("analysis") else "❌"
        has_feedback = "✅" if check_has_feedback(error, feedback_collector) else "❌"

        if (show_analyzed or not error.get("analysis")) and (
            show_feedback or not check_has_feedback(error, feedback_collector)
        ):
            table.add_row(
                error["date"],
                error["dag_id"][:20],
                error["task_id"][:20],
                error.get("category", "Unknown"),
                has_analysis,
                has_feedback,
            )

    console.print(table)


@feedback_app.command("bulk-feedback")
def bulk_feedback_session(
    days: int = typer.Option(1, help="Number of days to review"),
    category: Optional[str] = typer.Option(None, help="Filter by error category"),
    max_errors: int = typer.Option(20, help="Maximum errors to process"),
    require_analysis: bool = typer.Option(
        True, help="Only show errors that have been analyzed"
    ),
):
    """Start a bulk feedback session for multiple errors"""

    console.print("[bold blue]Starting bulk feedback session[/bold blue]")
    console.print(f"Reviewing last {days} days, max {max_errors} errors")

    # Load candidates for feedback
    candidates = load_feedback_candidates(days, category, max_errors, require_analysis)

    if not candidates:
        console.print("[yellow]No errors found matching criteria[/yellow]")
        return

    console.print(f"\n[green]Found {len(candidates)} errors ready for feedback[/green]")

    # Start feedback session
    feedback_collector = FeedbackCollector()
    feedback_count = 0

    for i, error_entry in enumerate(candidates, 1):
        console.print(f"\n[bold cyan]═══ Error {i}/{len(candidates)} ═══[/bold cyan]")

        # Show error and analysis side by side
        display_error_analysis_comparison(error_entry)

        # Collect feedback
        if Confirm.ask("Provide feedback?"):
            success = collect_interactive_feedback(error_entry, feedback_collector)
            if success:
                feedback_count += 1

        # Progress check
        if i % 5 == 0:
            console.print(
                f"[blue]Progress: {i}/{len(candidates)} reviewed, {feedback_count} feedback collected[/blue]"
            )
            if not Confirm.ask("Continue?"):
                break

    console.print("\n[green]✅ Bulk feedback session completed![/green]")
    console.print(f"Total feedback collected: {feedback_count}")


@feedback_app.command("export-daily")
def export_daily_feedback(
    date: Optional[str] = typer.Option(None, help="Date to export (YYYY-MM-DD)"),
    output_file: str = typer.Option("daily_feedback_export.jsonl", help="Output file"),
):
    """Export all feedback for a specific date"""

    if date:
        export_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        export_date = datetime.now()

    console.print(
        f"[bold blue]Exporting feedback for {export_date.strftime('%Y-%m-%d')}[/bold blue]"
    )

    feedback_collector = FeedbackCollector()
    daily_feedback = get_daily_feedback(feedback_collector, export_date)

    if not daily_feedback:
        console.print("[yellow]No feedback found for this date[/yellow]")
        return

    # Export to file
    with open(output_file, "w") as f:
        for feedback in daily_feedback:
            f.write(json.dumps(feedback) + "\n")

    console.print(
        f"[green]✅ Exported {len(daily_feedback)} feedback records to {output_file}[/green]"
    )


@feedback_app.command("stats")
def show_feedback_stats(
    days: int = typer.Option(30, help="Number of days for statistics")
):
    """Show comprehensive feedback statistics"""

    console.print(f"[bold blue]Feedback Statistics (Last {days} days)[/bold blue]")

    feedback_collector = FeedbackCollector()
    stats = feedback_collector.get_feedback_stats()

    # Main stats
    stats_table = Table(title="Overview")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Total Feedback", str(stats.total_feedback_count))
    stats_table.add_row("Average Rating", f"{stats.avg_user_rating:.1f}/5.0")
    stats_table.add_row("Recent (7 days)", str(stats.recent_feedback_count))

    console.print(stats_table)

    # Category distribution
    if stats.category_distribution:
        cat_table = Table(title="Category Distribution")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", style="green")
        cat_table.add_column("Percentage", style="yellow")

        total = sum(stats.category_distribution.values())
        for category, count in sorted(stats.category_distribution.items()):
            percentage = (count / total) * 100 if total > 0 else 0
            cat_table.add_row(category, str(count), f"{percentage:.1f}%")

        console.print(cat_table)


@feedback_app.command("storage-info")
def show_storage_info(
    config_path: Optional[str] = typer.Option(None, help="Path to config file")
):
    """Show storage configuration and usage information"""

    console.print("[bold blue]Storage Configuration & Usage[/bold blue]")

    try:
        # Get database instance with configuration
        db = get_feedback_db(config_path)
        storage_info = db.get_storage_info()

        # Configuration table
        config_table = Table(title="Storage Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Database Path", storage_info["database_path"])
        config_table.add_row("Training Data Dir", storage_info["training_data_dir"])
        config_table.add_row("JSONL Backup", storage_info["jsonl_backup_path"])
        config_table.add_row("Retention Days", str(storage_info["retention_days"]))
        config_table.add_row(
            "Cleanup Enabled", "✅" if storage_info["cleanup_enabled"] else "❌"
        )
        config_table.add_row(
            "Backup Enabled", "✅" if storage_info["backup_enabled"] else "❌"
        )

        console.print(config_table)

        # Usage table
        usage_table = Table(title="Storage Usage")
        usage_table.add_column("Metric", style="cyan")
        usage_table.add_column("Value", style="yellow")

        usage_table.add_row("Database Size", f"{storage_info['database_size_mb']} MB")
        usage_table.add_row("Backup Count", str(storage_info["backup_count"]))

        console.print(usage_table)

        # Show configured categories and severities
        categories = db.get_configured_categories()
        severities = db.get_configured_severities()

        cat_table = Table(title="Configured Categories & Severities")
        cat_table.add_column("Categories", style="green")
        cat_table.add_column("Severities", style="yellow")

        max_rows = max(len(categories), len(severities))
        for i in range(max_rows):
            cat = categories[i] if i < len(categories) else ""
            sev = severities[i] if i < len(severities) else ""
            cat_table.add_row(cat, sev)

        console.print(cat_table)

        # Check auto-export status
        if db.should_auto_export():
            console.print(
                "\n[yellow]⚠ Auto-export threshold reached - consider running training pipeline[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error loading storage info: {e}[/red]")


@feedback_app.command("cleanup")
def cleanup_old_data(
    retention_days: Optional[int] = typer.Option(
        None, help="Days to retain (uses config default if not specified)"
    ),
    config_path: Optional[str] = typer.Option(None, help="Path to config file"),
    dry_run: bool = typer.Option(
        False, help="Show what would be deleted without actually deleting"
    ),
):
    """Clean up old feedback data based on retention policy"""

    try:
        db = get_feedback_db(config_path)

        if retention_days is None:
            retention_days = db.feedback_config.storage.retention_days

        console.print(
            f"[bold blue]Data Cleanup (Retention: {retention_days} days)[/bold blue]"
        )

        if dry_run:
            console.print("[yellow]DRY RUN - No data will be deleted[/yellow]")

            # Show what would be deleted
            cutoff_date = (datetime.now() - timedelta(days=retention_days)).strftime(
                "%Y-%m-%d"
            )

            with sqlite3.connect(db.db_path) as conn:
                old_count = conn.execute(
                    "SELECT COUNT(*) FROM error_logs WHERE date < ?", (cutoff_date,)
                ).fetchone()[0]

            console.print(
                f"Would delete {old_count} error records older than {cutoff_date}"
            )
        else:
            if not Confirm.ask(f"Delete data older than {retention_days} days?"):
                console.print("Cleanup cancelled")
                return

            deleted_count = db.cleanup_old_data(retention_days)
            console.print(f"[green]✅ Deleted {deleted_count} old records[/green]")

    except Exception as e:
        console.print(f"[red]Cleanup failed: {e}[/red]")


@feedback_app.command("backup")
def create_backup(
    backup_dir: Optional[str] = typer.Option(
        None, help="Backup directory (uses config default if not specified)"
    ),
    config_path: Optional[str] = typer.Option(None, help="Path to config file"),
):
    """Create backup of feedback database"""

    try:
        db = get_feedback_db(config_path)

        console.print("[bold blue]Creating Database Backup[/bold blue]")

        backup_file = db.backup_data(backup_dir)
        console.print(f"[green]✅ Backup created: {backup_file}[/green]")

        # Show backup info
        backup_path = Path(backup_file).parent
        backup_files = list(backup_path.glob("feedback_backup_*.db"))
        console.print(f"Total backups: {len(backup_files)}")

    except Exception as e:
        console.print(f"[red]Backup failed: {e}[/red]")


# Helper functions


def load_daily_errors(date: datetime) -> List[dict]:
    """Load error data for a specific date"""
    # This would typically query your error logging system
    # For now, simulate with training data

    training_file = Path("data/training_data.jsonl")
    if not training_file.exists():
        return []

    errors = []
    with open(training_file, "r") as f:
        for line in f:
            try:
                error_data = json.loads(line.strip())
                # Add simulated metadata
                error_data.update(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "dag_id": "simulated_dag",
                        "task_id": "simulated_task",
                        "category": "configuration_error",
                    }
                )
                errors.append(error_data)
            except json.JSONDecodeError:
                continue

    return errors[:10]  # Limit for demo


def load_recent_errors(
    days: int, category: Optional[str], dag_id: Optional[str]
) -> List[dict]:
    """Load recent errors with optional filtering"""
    # Simulate recent errors
    return load_daily_errors(datetime.now())


def load_feedback_candidates(
    days: int, category: Optional[str], max_errors: int, require_analysis: bool
) -> List[dict]:
    """Load errors that are candidates for feedback"""
    return load_daily_errors(datetime.now())[:max_errors]


def check_has_feedback(error: dict, feedback_collector: FeedbackCollector) -> bool:
    """Check if an error already has feedback"""
    # This would check the feedback database
    return False


def display_error_summary(error_entry: dict):
    """Display a formatted error summary"""

    panel_content = f"""[bold]Error Details:[/bold]
• DAG: {error_entry.get('dag_id', 'N/A')}
• Task: {error_entry.get('task_id', 'N/A')}
• Category: {error_entry.get('category', 'Unknown')}

[bold]Error Message:[/bold]
{error_entry.get('error', 'No error message')[:200]}...

[bold]Log Context:[/bold]
{error_entry.get('candidates', 'No log context')[:300]}..."""

    console.print(Panel(panel_content, title="Error Summary", border_style="blue"))


def display_error_analysis_comparison(error_entry: dict):
    """Display error and its analysis side by side"""

    # Error panel
    error_content = f"""[bold red]Original Error:[/bold red]
{error_entry.get('error', 'No error message')}

[bold blue]Log Context:[/bold blue]
{error_entry.get('candidates', 'No log context')[:400]}..."""

    # Analysis panel (if exists)
    analysis_content = f"""[bold green]AI Analysis:[/bold green]
{error_entry.get('analysis', 'No analysis available')}

[bold yellow]Confidence:[/bold yellow]
{error_entry.get('confidence', 'N/A')}"""

    console.print(Panel(error_content, title="Error", border_style="red"))
    console.print(
        Panel(analysis_content, title="Current Analysis", border_style="green")
    )


def collect_interactive_feedback(
    error_entry: dict, feedback_collector: FeedbackCollector
) -> bool:
    """Collect feedback interactively from the user"""

    try:
        console.print("\n[bold yellow]Collecting Feedback[/bold yellow]")

        # Collect corrected error message
        corrected_error = Prompt.ask(
            "Corrected error message", default=error_entry.get("error", "")
        )

        # Category selection
        categories = [
            "configuration_error",
            "timeout_error",
            "data_quality",
            "dependency_failure",
            "resource_error",
            "permission_error",
            "unknown",
        ]

        console.print(
            "Categories: "
            + ", ".join(f"{i+1}. {cat}" for i, cat in enumerate(categories))
        )
        cat_choice = Prompt.ask("Select category (1-7)", default="1")
        try:
            category = categories[int(cat_choice) - 1]
        except (ValueError, IndexError):
            category = "unknown"

        # Severity
        severities = ["low", "medium", "high", "critical"]
        console.print(
            "Severities: "
            + ", ".join(f"{i+1}. {sev}" for i, sev in enumerate(severities))
        )
        sev_choice = Prompt.ask("Select severity (1-4)", default="2")
        try:
            severity = severities[int(sev_choice) - 1]
        except (ValueError, IndexError):
            severity = "medium"

        # Confidence
        confidence = float(Prompt.ask("Confidence (0.0-1.0)", default="0.8"))

        # Reasoning
        reasoning = Prompt.ask("Brief reasoning", default="Manual correction")

        # Rating
        rating = int(Prompt.ask("Rate AI analysis (1-5)", default="3"))

        # Comments
        comments = Prompt.ask("Additional comments (optional)", default="")

        # Create feedback submission
        feedback = FeedbackSubmission(
            log_context=error_entry.get("candidates", ""),
            dag_id=error_entry.get("dag_id", "unknown"),
            task_id=error_entry.get("task_id", "unknown"),
            original_analysis=FeedbackCorrection(
                error_message=error_entry.get("error", ""),
                category="unknown",
                severity="medium",
                confidence=0.5,
                reasoning="Original analysis",
            ),
            corrected_analysis=FeedbackCorrection(
                error_message=corrected_error,
                category=category,
                severity=severity,
                confidence=confidence,
                reasoning=reasoning,
            ),
            user_rating=rating,
            user_id="cli_user",
            comments=comments,
        )

        # Save feedback
        success = feedback_collector.save_feedback(feedback)

        if success:
            console.print("[green]✅ Feedback saved successfully[/green]")
        else:
            console.print("[red]❌ Failed to save feedback[/red]")

        return success

    except Exception as e:
        console.print(f"[red]Error collecting feedback: {e}[/red]")
        return False


def get_daily_feedback(
    feedback_collector: FeedbackCollector, date: datetime
) -> List[dict]:
    """Get all feedback for a specific date"""

    if not feedback_collector.feedback_file.exists():
        return []

    daily_feedback = []
    target_date = date.strftime("%Y-%m-%d")

    with open(feedback_collector.feedback_file, "r") as f:
        for line in f:
            try:
                feedback = json.loads(line.strip())
                feedback_date = feedback.get("timestamp", "").split("T")[0]
                if feedback_date == target_date:
                    daily_feedback.append(feedback)
            except json.JSONDecodeError:
                continue

    return daily_feedback


if __name__ == "__main__":
    feedback_app()
