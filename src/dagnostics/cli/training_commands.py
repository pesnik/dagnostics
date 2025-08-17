"""
CLI Commands for Fine-tuning Pipeline

Command-line interface for managing the automated fine-tuning pipeline,
dataset generation, and model deployment.
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Import feedback collector (no ML dependencies)
from dagnostics.web.feedback import FeedbackCollector

# Lazy import flag for training modules
HAS_TRAINING_MODULES = None
TRAINING_IMPORT_ERROR = None

logger = logging.getLogger(__name__)
console = Console()

# Create training CLI app
training_app = typer.Typer(
    name="training", help="Fine-tuning and model training commands"
)


def _check_training_dependencies():
    """Check if training dependencies are available"""
    global HAS_TRAINING_MODULES, TRAINING_IMPORT_ERROR

    if HAS_TRAINING_MODULES is None:
        # Lazy check - only import when needed
        try:
            from dagnostics.training.dataset_generator import (  # noqa: F401
                DatasetGenerator,
            )
            from dagnostics.training.fine_tuner import SLMFineTuner  # noqa: F401

            HAS_TRAINING_MODULES = True
        except ImportError as e:
            HAS_TRAINING_MODULES = False
            TRAINING_IMPORT_ERROR = str(e)

    if not HAS_TRAINING_MODULES:
        console.print("[red]❌ Training dependencies not available[/red]")
        console.print(f"[yellow]Error: {TRAINING_IMPORT_ERROR}[/yellow]")
        console.print("\n[bold]Set up training environment:[/bold]")
        console.print("1. Use training machine: copy codebase + install ML deps")
        console.print("2. Use Docker: docker-compose up training")
        console.print(
            "3. Install locally: pip install torch transformers datasets peft"
        )
        raise typer.Exit(1)

    return True


@training_app.command("generate-dataset")
def generate_dataset(
    output_dir: str = typer.Option(
        "data/training", help="Output directory for training data"
    ),
    min_examples: int = typer.Option(10, help="Minimum examples required for training"),
    _include_feedback: bool = typer.Option(
        True, help="Include user feedback in dataset"
    ),
):
    """Generate training dataset from logs and user feedback"""

    _check_training_dependencies()

    console.print("[bold blue]Generating training dataset...[/bold blue]")

    try:
        # Import and initialize dataset generator
        from dagnostics.training.dataset_generator import DatasetGenerator

        generator = DatasetGenerator(output_dir=output_dir)

        # Generate dataset
        dataset_info = generator.generate_full_dataset()

        if not dataset_info:
            console.print("[red]❌ No training data available[/red]")
            console.print(
                "Add raw logs to data/training_data.jsonl or collect user feedback first"
            )
            return

        # Check if we have enough examples
        if int(dataset_info["total_size"]) < min_examples:
            console.print(
                f"[yellow]⚠ Warning: Only {dataset_info['total_size']} examples generated (minimum: {min_examples})[/yellow]"
            )

        # Display results
        table = Table(title="Training Dataset Generated")
        table.add_column("Dataset", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Path", style="yellow")

        table.add_row(
            "Training", str(dataset_info["train_size"]), dataset_info["train_path"]
        )
        table.add_row(
            "Validation",
            str(dataset_info["validation_size"]),
            dataset_info["validation_path"],
        )
        table.add_row("Total", str(dataset_info["total_size"]), "")

        console.print(table)
        console.print("[green]✅ Dataset generation completed![/green]")

    except Exception as e:
        console.print(f"[red]❌ Dataset generation failed: {e}[/red]")
        raise typer.Exit(1)


@training_app.command("train-model")
def train_model(
    model_name: str = typer.Option(
        "microsoft/DialoGPT-small", help="Base model to fine-tune"
    ),
    train_dataset: str = typer.Option(
        "data/training/train_dataset.jsonl", help="Training dataset path"
    ),
    val_dataset: Optional[str] = typer.Option(
        "data/training/validation_dataset.jsonl", help="Validation dataset path"
    ),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    batch_size: int = typer.Option(2, help="Training batch size"),
    use_quantization: bool = typer.Option(
        True, help="Use 4-bit quantization for memory efficiency"
    ),
    output_dir: str = typer.Option(
        "models/fine_tuned", help="Output directory for trained model"
    ),
):
    """Fine-tune a small language model for error analysis"""

    _check_training_dependencies()

    console.print(f"[bold blue]Fine-tuning model: {model_name}[/bold blue]")

    # Check if training dataset exists
    if not Path(train_dataset).exists():
        console.print(f"[red]❌ Training dataset not found: {train_dataset}[/red]")
        console.print("Run 'dagnostics training generate-dataset' first")
        raise typer.Exit(1)

    try:
        # Import and initialize fine-tuner
        from dagnostics.training.fine_tuner import SLMFineTuner

        fine_tuner = SLMFineTuner(
            model_name=model_name,
            output_dir=output_dir,
            use_quantization=use_quantization,
        )

        # Display training configuration
        config_table = Table(title="Training Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Base Model", model_name)
        config_table.add_row("Training Dataset", train_dataset)
        config_table.add_row("Validation Dataset", val_dataset or "None")
        config_table.add_row("Epochs", str(epochs))
        config_table.add_row("Learning Rate", str(learning_rate))
        config_table.add_row("Batch Size", str(batch_size))
        config_table.add_row("Quantization", "Yes" if use_quantization else "No")

        console.print(config_table)

        # Start training
        model_path = fine_tuner.train_model(
            train_dataset_path=train_dataset,
            validation_dataset_path=(
                val_dataset if val_dataset and Path(val_dataset).exists() else None
            ),
            num_epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        console.print("[green]✅ Model training completed![/green]")
        console.print(f"[bold]Model saved to:[/bold] {model_path}")

        # Evaluate model if validation set exists
        if val_dataset and Path(val_dataset).exists():
            console.print("[blue]Evaluating model...[/blue]")
            eval_results = fine_tuner.evaluate_model(model_path, val_dataset)

            eval_table = Table(title="Model Evaluation")
            eval_table.add_column("Metric", style="cyan")
            eval_table.add_column("Value", style="green")

            eval_table.add_row("Perplexity", f"{eval_results['perplexity']:.2f}")
            eval_table.add_row("Average Loss", f"{eval_results['average_loss']:.4f}")
            eval_table.add_row("Test Examples", str(eval_results["num_test_examples"]))

            console.print(eval_table)

    except Exception as e:
        console.print(f"[red]❌ Model training failed: {e}[/red]")
        raise typer.Exit(1)


@training_app.command("deploy-ollama")
def deploy_to_ollama(
    model_path: str = typer.Argument(..., help="Path to fine-tuned model"),
    model_name: str = typer.Option("dagnostics-slm", help="Name for Ollama model"),
    auto_build: bool = typer.Option(False, help="Automatically build Ollama model"),
):
    """Deploy fine-tuned model to Ollama"""

    console.print(f"[bold blue]Deploying model to Ollama: {model_name}[/bold blue]")

    if not Path(model_path).exists():
        console.print(f"[red]❌ Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    try:
        # Initialize fine-tuner for export
        from dagnostics.training.fine_tuner import SLMFineTuner

        fine_tuner = SLMFineTuner()

        # Export model for Ollama
        export_path = fine_tuner.export_for_ollama(model_path, model_name)

        console.print(f"[green]✅ Model exported to:[/green] {export_path}")

        if auto_build:
            import shutil
            import subprocess  # nosec - subprocess needed for ollama integration

            console.print("[blue]Building Ollama model...[/blue]")

            # Check if ollama is available
            ollama_path = shutil.which("ollama")
            if not ollama_path:
                console.print("[red]❌ Ollama not found in PATH[/red]")
                raise typer.Exit(1)

            # Build Ollama model - subprocess call is safe, path validated
            result = subprocess.run(  # nosec
                [ollama_path, "create", model_name, "-f", f"{export_path}/Modelfile"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                console.print(
                    f"[green]✅ Ollama model '{model_name}' built successfully![/green]"
                )
                console.print(f"[bold]Test with:[/bold] ollama run {model_name}")
            else:
                console.print(f"[red]❌ Ollama build failed: {result.stderr}[/red]")
        else:
            console.print("[yellow]Manual deployment required:[/yellow]")
            console.print(f"  cd {export_path}")
            console.print(f"  ollama create {model_name} -f Modelfile")
            console.print(f"  ollama run {model_name}")

    except Exception as e:
        console.print(f"[red]❌ Deployment failed: {e}[/red]")
        raise typer.Exit(1)


@training_app.command("feedback-stats")
def show_feedback_stats():
    """Show user feedback statistics"""

    console.print("[bold blue]User Feedback Statistics[/bold blue]")

    try:
        feedback_collector = FeedbackCollector()
        stats = feedback_collector.get_feedback_stats()

        # Main stats table
        stats_table = Table(title="Feedback Overview")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Feedback", str(stats.total_feedback_count))
        stats_table.add_row("Average Rating", f"{stats.avg_user_rating}/5.0")
        stats_table.add_row("Recent (7 days)", str(stats.recent_feedback_count))

        console.print(stats_table)

        # Category distribution
        if stats.category_distribution:
            cat_table = Table(title="Error Category Distribution")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="green")
            cat_table.add_column("Percentage", style="yellow")

            total_categorized = sum(stats.category_distribution.values())
            for category, count in sorted(stats.category_distribution.items()):
                percentage = (count / total_categorized) * 100
                cat_table.add_row(category, str(count), f"{percentage:.1f}%")

            console.print(cat_table)

        if stats.total_feedback_count == 0:
            console.print(
                "[yellow]No feedback collected yet. Users can provide feedback through the web interface.[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]❌ Failed to load feedback stats: {e}[/red]")


@training_app.command("export-feedback")
def export_feedback(
    min_rating: int = typer.Option(3, help="Minimum rating for quality feedback"),
    _output_path: str = typer.Option(
        "data/training/feedback_export.jsonl", help="Export file path"
    ),
):
    """Export user feedback for training"""

    console.print(
        f"[bold blue]Exporting feedback (min rating: {min_rating})[/bold blue]"
    )

    try:
        feedback_collector = FeedbackCollector()
        export_path = feedback_collector.export_for_training(min_rating)

        # Count exported records
        export_count = 0
        if Path(export_path).exists():
            with open(export_path, "r") as f:
                export_count = sum(1 for _ in f)

        console.print(
            f"[green]✅ Exported {export_count} quality feedback records[/green]"
        )
        console.print(f"[bold]Export path:[/bold] {export_path}")

        if export_count > 0:
            console.print(
                "[blue]Use this feedback by regenerating the training dataset:[/blue]"
            )
            console.print("  dagnostics training generate-dataset")

    except Exception as e:
        console.print(f"[red]❌ Export failed: {e}[/red]")
        raise typer.Exit(1)


@training_app.command("pipeline")
def run_full_pipeline(
    model_name: str = typer.Option(
        "microsoft/DialoGPT-small", help="Base model to fine-tune"
    ),
    min_feedback: int = typer.Option(5, help="Minimum feedback examples required"),
    epochs: int = typer.Option(3, help="Training epochs"),
    auto_deploy: bool = typer.Option(
        False, help="Auto-deploy to Ollama after training"
    ),
):
    """Run complete training pipeline: dataset generation → training → deployment"""

    console.print("[bold blue]🚀 Running complete fine-tuning pipeline[/bold blue]")

    try:
        # Step 1: Generate dataset
        console.print("\n[bold cyan]Step 1: Generating training dataset[/bold cyan]")
        from dagnostics.training.dataset_generator import DatasetGenerator

        generator = DatasetGenerator()
        dataset_info = generator.generate_full_dataset()

        if not dataset_info or int(dataset_info["total_size"]) < min_feedback:
            console.print(
                f"[red]❌ Insufficient training data: {dataset_info['total_size'] if dataset_info else 0} examples (minimum: {min_feedback})[/red]"
            )
            console.print("Collect more user feedback or reduce min_feedback parameter")
            raise typer.Exit(1)

        console.print(
            f"[green]✅ Dataset ready: {dataset_info['total_size']} examples[/green]"
        )

        # Step 2: Train model
        console.print("\n[bold cyan]Step 2: Training model[/bold cyan]")
        from dagnostics.training.fine_tuner import SLMFineTuner

        fine_tuner = SLMFineTuner(model_name=model_name)
        model_path = fine_tuner.train_model(
            train_dataset_path=dataset_info["train_path"],
            validation_dataset_path=dataset_info["validation_path"],
            num_epochs=epochs,
            batch_size=2,
        )

        console.print(f"[green]✅ Model trained: {model_path}[/green]")

        # Step 3: Deploy to Ollama
        if auto_deploy:
            console.print("\n[bold cyan]Step 3: Deploying to Ollama[/bold cyan]")
            timestamp = dataset_info["created_at"][:10].replace("-", "")
            model_version = f"dagnostics-slm-v{timestamp}"

            export_path = fine_tuner.export_for_ollama(model_path, model_version)
            console.print(f"[green]✅ Ready for deployment: {export_path}[/green]")

            console.print("[bold yellow]Manual deployment step:[/bold yellow]")
            console.print(f"  cd {export_path}")
            console.print(f"  ollama create {model_version} -f Modelfile")

        console.print("\n[bold green]🎉 Pipeline completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Pipeline failed: {e}[/red]")
        raise typer.Exit(1)


@training_app.command("remote-train")
def remote_train(
    model_name: str = typer.Option(
        "microsoft/DialoGPT-small", help="Base model to fine-tune"
    ),
    train_dataset: str = typer.Option(
        "data/training/train_dataset.jsonl", help="Training dataset path"
    ),
    val_dataset: Optional[str] = typer.Option(
        "data/training/validation_dataset.jsonl", help="Validation dataset path"
    ),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    server_url: str = typer.Option("http://localhost:8001", help="Training server URL"),
    wait: bool = typer.Option(True, help="Wait for training completion"),
):
    """Submit training job to remote training server"""

    console.print("[bold blue]Submitting remote training job...[/bold blue]")

    try:
        from dagnostics.training.remote_trainer import remote_train_command

        result = remote_train_command(
            model_name=model_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            server_url=server_url,
            wait=wait,
        )

        if wait:
            console.print(f"[green]✅ Training completed! Model: {result}[/green]")
        else:
            console.print(f"[green]✅ Job submitted: {result}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Remote training failed: {e}[/red]")
        raise typer.Exit(1)


@training_app.command("remote-status")
def remote_status(
    job_id: str = typer.Argument(..., help="Training job ID"),
    server_url: str = typer.Option("http://localhost:8001", help="Training server URL"),
):
    """Check status of remote training job"""

    try:
        from dagnostics.training.remote_trainer import remote_status_command

        remote_status_command(job_id, server_url)

    except Exception as e:
        console.print(f"[red]❌ Failed to get status: {e}[/red]")
        raise typer.Exit(1)


@training_app.command("remote-download")
def remote_download(
    job_id: str = typer.Argument(..., help="Training job ID"),
    output_dir: str = typer.Option("models/fine_tuned", help="Output directory"),
    server_url: str = typer.Option("http://localhost:8001", help="Training server URL"),
):
    """Download trained model from remote server"""

    try:
        from dagnostics.training.remote_trainer import remote_download_command

        model_path = remote_download_command(job_id, output_dir, server_url)
        console.print(f"[green]✅ Model downloaded: {model_path}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Download failed: {e}[/red]")
        raise typer.Exit(1)


# New commands for enhanced fine-tuning capabilities


@training_app.command("prepare-data")
def prepare_training_data(
    dataset_path: str = typer.Argument(
        "data/training_dataset_2025-08-17T11-15-10.json",
        help="Path to human-reviewed training dataset",
    ),
    output_dir: str = typer.Option(
        "data/fine_tuning", help="Output directory for prepared datasets"
    ),
):
    """Prepare fine-tuning datasets from human-reviewed data"""

    console.print("[bold blue]Preparing fine-tuning datasets...[/bold blue]")

    import subprocess
    import sys
    from pathlib import Path

    # Check if dataset exists
    if not Path(dataset_path).exists():
        console.print(f"[red]❌ Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    try:
        # Run the data preparation script
        result = subprocess.run(
            [sys.executable, "scripts/prepare_training_data.py", dataset_path],
            capture_output=True,
            text=True,
            check=True,
        )

        console.print("[green]✅ Training datasets prepared successfully![/green]")
        console.print(result.stdout)

        # Show what was created
        output_path = Path(output_dir)
        if output_path.exists():
            files = list(output_path.glob("*.jsonl"))
            if files:
                table = Table(title="Generated Datasets")
                table.add_column("File", style="cyan")
                table.add_column("Path", style="yellow")

                for file in files:
                    table.add_row(file.name, str(file))

                console.print(table)

    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Data preparation failed: {e.stderr}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@training_app.command("train-local")
def train_local_model(
    model_name: str = typer.Option(
        "microsoft/DialoGPT-small", help="Base model to fine-tune"
    ),
    train_dataset: str = typer.Option(
        "data/fine_tuning/train_dataset.jsonl", help="Training dataset path"
    ),
    val_dataset: str = typer.Option(
        "data/fine_tuning/validation_dataset.jsonl", help="Validation dataset path"
    ),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    batch_size: int = typer.Option(2, help="Training batch size"),
    model_output_name: str = typer.Option(
        "dagnostics-error-extractor", help="Output model name"
    ),
    use_quantization: bool = typer.Option(
        True, help="Use 4-bit quantization for memory efficiency"
    ),
    export_for_ollama: bool = typer.Option(
        True, help="Export model for Ollama after training"
    ),
):
    """Fine-tune a local model using prepared datasets"""

    _check_training_dependencies()

    console.print(f"[bold blue]Fine-tuning local model: {model_name}[/bold blue]")

    # Check if datasets exist
    if not Path(train_dataset).exists():
        console.print(f"[red]❌ Training dataset not found: {train_dataset}[/red]")
        console.print("Run 'dagnostics training prepare-data' first")
        raise typer.Exit(1)

    try:
        from dagnostics.training.fine_tuner import train_from_prepared_data

        model_path = train_from_prepared_data(
            model_name=model_name,
            train_dataset_path=train_dataset,
            validation_dataset_path=val_dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            model_output_name=model_output_name,
            use_quantization=use_quantization,
            export_for_ollama=export_for_ollama,
        )

        console.print("[green]✅ Local fine-tuning completed successfully![/green]")
        console.print(f"[bold]Model saved to:[/bold] {model_path}")

    except Exception as e:
        console.print(f"[red]❌ Local fine-tuning failed: {e}[/red]")
        logger.error(f"Local fine-tuning error: {e}", exc_info=True)
        raise typer.Exit(1)


@training_app.command("train-openai")
def train_openai_model(
    train_dataset: str = typer.Option(
        "data/fine_tuning/train_dataset.jsonl", help="Training dataset path"
    ),
    val_dataset: str = typer.Option(
        "data/fine_tuning/validation_dataset.jsonl", help="Validation dataset path"
    ),
    model: str = typer.Option("gpt-3.5-turbo", help="OpenAI base model"),
    api_key: Optional[str] = typer.Option(
        None, help="OpenAI API key (or set OPENAI_API_KEY env var)"
    ),
    suffix: str = typer.Option("dagnostics-error-extractor", help="Model name suffix"),
    wait: bool = typer.Option(True, help="Wait for training completion"),
):
    """Fine-tune an OpenAI model using their API"""

    console.print(f"[bold blue]Fine-tuning OpenAI model: {model}[/bold blue]")

    # Check if datasets exist
    if not Path(train_dataset).exists():
        console.print(f"[red]❌ Training dataset not found: {train_dataset}[/red]")
        console.print("Run 'dagnostics training prepare-data' first")
        raise typer.Exit(1)

    try:
        from dagnostics.training.api_fine_tuner import fine_tune_openai

        result = fine_tune_openai(
            train_dataset_path=train_dataset,
            validation_dataset_path=val_dataset if Path(val_dataset).exists() else None,
            model=model,
            api_key=api_key,
            suffix=suffix,
            wait_for_completion=wait,
        )

        if result["status"] == "succeeded":
            console.print(
                "[green]✅ OpenAI fine-tuning completed successfully![/green]"
            )
            console.print(
                f"[bold]Fine-tuned model:[/bold] {result['fine_tuned_model']}"
            )
            console.print(f"[bold]Tokens trained:[/bold] {result['trained_tokens']:,}")
        elif result["status"] == "running":
            console.print(
                f"[yellow]🔄 Training job submitted: {result['job_id']}[/yellow]"
            )
            console.print("Check status in OpenAI dashboard or wait for completion")
        else:
            console.print(
                f"[red]❌ Training failed: {result.get('error', 'Unknown error')}[/red]"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ OpenAI fine-tuning failed: {e}[/red]")
        logger.error(f"OpenAI fine-tuning error: {e}", exc_info=True)
        raise typer.Exit(1)


@training_app.command("train-anthropic")
def train_anthropic_model(
    train_dataset: str = typer.Option(
        "data/fine_tuning/train_dataset.jsonl", help="Training dataset path"
    ),
    val_dataset: str = typer.Option(
        "data/fine_tuning/validation_dataset.jsonl", help="Validation dataset path"
    ),
    model: str = typer.Option("claude-3-haiku-20240307", help="Anthropic base model"),
    api_key: Optional[str] = typer.Option(
        None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    ),
):
    """Prepare data for Anthropic fine-tuning (when available)"""

    console.print(
        f"[bold blue]Preparing Anthropic fine-tuning data for: {model}[/bold blue]"
    )

    # Check if datasets exist
    if not Path(train_dataset).exists():
        console.print(f"[red]❌ Training dataset not found: {train_dataset}[/red]")
        console.print("Run 'dagnostics training prepare-data' first")
        raise typer.Exit(1)

    try:
        from dagnostics.training.api_fine_tuner import fine_tune_anthropic

        result = fine_tune_anthropic(
            train_dataset_path=train_dataset,
            validation_dataset_path=val_dataset if Path(val_dataset).exists() else None,
            model=model,
            api_key=api_key,
        )

        console.print("[green]✅ Anthropic data preparation completed![/green]")
        console.print(f"[bold]Status:[/bold] {result['message']}")
        console.print(f"[bold]Train dataset:[/bold] {result['train_dataset']}")
        if result["validation_dataset"]:
            console.print(
                f"[bold]Validation dataset:[/bold] {result['validation_dataset']}"
            )

    except Exception as e:
        console.print(f"[red]❌ Anthropic preparation failed: {e}[/red]")
        logger.error(f"Anthropic preparation error: {e}", exc_info=True)
        raise typer.Exit(1)


@training_app.command("evaluate")
def evaluate_model(
    model_path: str = typer.Argument(..., help="Path to model to evaluate"),
    test_dataset: str = typer.Option(
        "data/fine_tuning/validation_dataset.jsonl", help="Test dataset path"
    ),
    model_type: str = typer.Option(
        "local", help="Model type: local, openai, or anthropic"
    ),
    output_dir: str = typer.Option(
        "evaluations", help="Output directory for evaluation results"
    ),
):
    """Evaluate a fine-tuned model on test data"""

    console.print(f"[bold blue]Evaluating {model_type} model: {model_path}[/bold blue]")

    # Check if test dataset exists
    if not Path(test_dataset).exists():
        console.print(f"[red]❌ Test dataset not found: {test_dataset}[/red]")
        raise typer.Exit(1)

    try:
        from dagnostics.training.model_evaluator import evaluate_model as eval_model

        results_path = eval_model(
            model_path=model_path,
            test_dataset_path=test_dataset,
            model_type=model_type,
            output_dir=output_dir,
        )

        console.print("[green]✅ Model evaluation completed![/green]")
        console.print(f"[bold]Results saved to:[/bold] {results_path}")

        # Show summary report path
        report_path = Path(results_path).with_suffix(".md")
        if report_path.exists():
            console.print(f"[bold]Detailed report:[/bold] {report_path}")

    except Exception as e:
        console.print(f"[red]❌ Model evaluation failed: {e}[/red]")
        logger.error(f"Model evaluation error: {e}", exc_info=True)
        raise typer.Exit(1)


@training_app.command("status")
def show_training_status():
    """Show training environment and dataset status"""

    console.print("[bold blue]DAGnostics Training Status[/bold blue]")

    # Check training dependencies
    status_table = Table(title="Training Environment")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="yellow")

    try:
        _check_training_dependencies()
        status_table.add_row(
            "ML Dependencies", "✅ Available", "Ready for local training"
        )
    except typer.Exit:
        status_table.add_row(
            "ML Dependencies",
            "❌ Missing",
            "Install with: pip install 'dagnostics[finetuning]'",
        )

    # Check API keys
    import os

    openai_key = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set"
    anthropic_key = "✅ Set" if os.getenv("ANTHROPIC_API_KEY") else "❌ Not set"

    status_table.add_row("OpenAI API Key", openai_key, "For OpenAI fine-tuning")
    status_table.add_row(
        "Anthropic API Key", anthropic_key, "For Anthropic preparation"
    )

    console.print(status_table)

    # Check datasets
    dataset_table = Table(title="Available Datasets")
    dataset_table.add_column("Dataset", style="cyan")
    dataset_table.add_column("Status", style="green")
    dataset_table.add_column("Path", style="yellow")

    datasets = [
        ("Human-reviewed data", "data/training_dataset_2025-08-17T11-15-10.json"),
        ("Training set", "data/fine_tuning/train_dataset.jsonl"),
        ("Validation set", "data/fine_tuning/validation_dataset.jsonl"),
        ("Full dataset", "data/fine_tuning/full_dataset.jsonl"),
    ]

    for name, path in datasets:
        status = "✅ Available" if Path(path).exists() else "❌ Missing"
        dataset_table.add_row(name, status, path)

    console.print(dataset_table)

    # Show next steps
    console.print("\n[bold yellow]Next Steps:[/bold yellow]")
    if not Path("data/fine_tuning/train_dataset.jsonl").exists():
        console.print(
            "1. Prepare training data: [cyan]dagnostics training prepare-data[/cyan]"
        )
    else:
        console.print("1. ✅ Training data ready")

    console.print("2. Choose training method:")
    console.print("   • Local: [cyan]dagnostics training train-local[/cyan]")
    console.print("   • OpenAI: [cyan]dagnostics training train-openai[/cyan]")
    console.print("   • Remote: [cyan]dagnostics training remote-train[/cyan]")
    console.print(
        "3. Evaluate model: [cyan]dagnostics training evaluate <model_path>[/cyan]"
    )


@training_app.command("start-server")
def start_training_server(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8001, help="Port to bind to"),
    workers: int = typer.Option(1, help="Number of workers"),
    log_level: str = typer.Option("info", help="Log level"),
):
    """Start the remote training server"""

    console.print(
        f"[bold blue]Starting DAGnostics Training Server on {host}:{port}[/bold blue]"
    )

    try:
        import sys

        from dagnostics.training.training_server import main as server_main

        # Override sys.argv for the server main function
        original_argv = sys.argv.copy()
        sys.argv = [
            "training_server.py",
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            str(workers),
            "--log-level",
            log_level,
        ]

        try:
            server_main()
        finally:
            sys.argv = original_argv

    except ImportError as e:
        console.print(f"[red]❌ Training server dependencies missing: {e}[/red]")
        console.print("Install with: [cyan]pip install 'dagnostics[finetuning]'[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Failed to start training server: {e}[/red]")
        raise typer.Exit(1)


@training_app.command("setup-remote")
def setup_remote_training(
    mode: str = typer.Option(None, help="Setup mode: local, docker, or cloud"),
):
    """Setup remote training infrastructure"""

    console.print("[bold blue]Setting up remote training infrastructure...[/bold blue]")

    try:
        import subprocess
        import sys

        script_path = "scripts/setup_remote_training.py"

        cmd = [sys.executable, script_path]
        if mode:
            cmd.extend(["--mode", mode])

        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            console.print("[green]✅ Remote training setup completed![/green]")
        else:
            console.print("[red]❌ Remote training setup failed[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Setup failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    training_app()
