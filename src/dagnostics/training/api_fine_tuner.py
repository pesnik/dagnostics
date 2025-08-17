"""
API Fine-tuning for OpenAI and Anthropic Models

Provides fine-tuning capabilities for cloud-based LLM providers
using their respective APIs and training formats.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _import_optional_dependency(name: str, extra: str = ""):
    """Import optional dependency with helpful error message."""
    msg = (
        f"Missing optional dependency '{name}'. {extra}\n"
        f"Install with: pip install 'dagnostics[llm]' or pip install {name}"
    )
    try:
        module = __import__(name)
        return module
    except ImportError:
        raise ImportError(msg) from None


class OpenAIFineTuner:
    """Fine-tune OpenAI models using their API"""

    def __init__(self, api_key: Optional[str] = None):
        self.openai = _import_optional_dependency(
            "openai", "Install with: pip install openai"
        )

        if api_key:
            self.client = self.openai.OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY env var
            self.client = self.openai.OpenAI()

    def prepare_openai_dataset(self, jsonl_path: str, output_path: str) -> str:
        """Convert DAGnostics format to OpenAI fine-tuning format"""

        logger.info(f"Converting dataset from {jsonl_path} to OpenAI format")

        with open(jsonl_path, "r") as infile, open(output_path, "w") as outfile:
            for line in infile:
                data = json.loads(line.strip())

                # Convert to OpenAI chat format
                openai_format = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert data engineer analyzing Airflow ETL task failure logs. Extract the core error message from log context.",
                        },
                        {
                            "role": "user",
                            "content": f"{data['instruction']}\n\n{data['input']}",
                        },
                        {"role": "assistant", "content": data["output"]},
                    ]
                }

                outfile.write(json.dumps(openai_format) + "\n")

        logger.info(f"OpenAI dataset saved to: {output_path}")
        return output_path

    def upload_dataset(self, file_path: str) -> str:
        """Upload dataset to OpenAI"""

        logger.info(f"Uploading dataset: {file_path}")

        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="fine-tune")

        logger.info(f"Dataset uploaded with ID: {file_obj.id}")
        return file_obj.id

    def create_fine_tuning_job(
        self,
        training_file_id: str,
        model: str = "gpt-3.5-turbo",
        validation_file_id: Optional[str] = None,
        suffix: Optional[str] = None,
        hyperparameters: Optional[Dict] = None,
    ) -> str:
        """Create fine-tuning job"""

        logger.info(f"Creating fine-tuning job for model: {model}")

        kwargs = {
            "training_file": training_file_id,
            "model": model,
        }

        if validation_file_id:
            kwargs["validation_file"] = validation_file_id

        if suffix:
            kwargs["suffix"] = suffix

        if hyperparameters:
            kwargs["hyperparameters"] = hyperparameters

        job = self.client.fine_tuning.jobs.create(**kwargs)

        logger.info(f"Fine-tuning job created: {job.id}")
        return job.id

    def monitor_job(self, job_id: str, poll_interval: int = 60) -> Dict:
        """Monitor fine-tuning job progress"""

        logger.info(f"Monitoring fine-tuning job: {job_id}")

        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)

            logger.info(f"Job status: {job.status}")

            if job.status in ["succeeded", "failed", "cancelled"]:
                logger.info(f"Job completed with status: {job.status}")
                return {
                    "job_id": job.id,
                    "status": job.status,
                    "fine_tuned_model": job.fine_tuned_model,
                    "trained_tokens": job.trained_tokens,
                    "result_files": job.result_files,
                    "error": job.error if job.status == "failed" else None,
                    "completed_at": datetime.now().isoformat(),
                }

            time.sleep(poll_interval)

    def fine_tune_from_dataset(
        self,
        train_dataset_path: str,
        validation_dataset_path: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        suffix: str = "dagnostics-error-extractor",
        hyperparameters: Optional[Dict] = None,
        wait_for_completion: bool = True,
    ) -> Dict:
        """Complete fine-tuning workflow"""

        logger.info(f"Starting OpenAI fine-tuning workflow")

        # Prepare datasets
        output_dir = Path("data/openai_training")
        output_dir.mkdir(exist_ok=True)

        train_openai_path = output_dir / "train_openai.jsonl"
        self.prepare_openai_dataset(train_dataset_path, str(train_openai_path))

        # Upload training data
        train_file_id = self.upload_dataset(str(train_openai_path))

        # Upload validation data if provided
        validation_file_id = None
        if validation_dataset_path:
            val_openai_path = output_dir / "validation_openai.jsonl"
            self.prepare_openai_dataset(validation_dataset_path, str(val_openai_path))
            validation_file_id = self.upload_dataset(str(val_openai_path))

        # Create fine-tuning job
        job_id = self.create_fine_tuning_job(
            training_file_id=train_file_id,
            validation_file_id=validation_file_id,
            model=model,
            suffix=suffix,
            hyperparameters=hyperparameters,
        )

        if wait_for_completion:
            # Monitor until completion
            result = self.monitor_job(job_id)

            # Save results
            results_path = output_dir / f"fine_tuning_results_{job_id}.json"
            with open(results_path, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"Fine-tuning results saved to: {results_path}")
            return result
        else:
            return {"job_id": job_id, "status": "running"}


class AnthropicFineTuner:
    """Fine-tune Anthropic models (when available)"""

    def __init__(self, api_key: Optional[str] = None):
        self.anthropic = _import_optional_dependency(
            "anthropic", "Install with: pip install anthropic"
        )

        if api_key:
            self.client = self.anthropic.Anthropic(api_key=api_key)
        else:
            # Will use ANTHROPIC_API_KEY env var
            self.client = self.anthropic.Anthropic()

    def prepare_anthropic_dataset(self, jsonl_path: str, output_path: str) -> str:
        """Convert DAGnostics format to Anthropic fine-tuning format"""

        logger.info(f"Converting dataset from {jsonl_path} to Anthropic format")

        # Note: Anthropic's fine-tuning format may differ
        # This is a placeholder implementation

        with open(jsonl_path, "r") as infile, open(output_path, "w") as outfile:
            for line in infile:
                data = json.loads(line.strip())

                # Convert to Anthropic format (adjust as needed)
                anthropic_format = {
                    "prompt": f"Human: {data['instruction']}\n\n{data['input']}\n\nAssistant:",
                    "completion": f" {data['output']}",
                }

                outfile.write(json.dumps(anthropic_format) + "\n")

        logger.info(f"Anthropic dataset saved to: {output_path}")
        return output_path

    def fine_tune_from_dataset(
        self,
        train_dataset_path: str,
        validation_dataset_path: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",
        **kwargs,
    ) -> Dict:
        """Fine-tune Anthropic model (placeholder)"""

        logger.warning("Anthropic fine-tuning is not yet generally available")
        logger.info("Converting dataset to Anthropic format for future use")

        # Prepare datasets for future use
        output_dir = Path("data/anthropic_training")
        output_dir.mkdir(exist_ok=True)

        train_anthropic_path = output_dir / "train_anthropic.jsonl"
        self.prepare_anthropic_dataset(train_dataset_path, str(train_anthropic_path))

        if validation_dataset_path:
            val_anthropic_path = output_dir / "validation_anthropic.jsonl"
            self.prepare_anthropic_dataset(
                validation_dataset_path, str(val_anthropic_path)
            )

        return {
            "status": "prepared",
            "message": "Dataset prepared for future Anthropic fine-tuning",
            "train_dataset": str(train_anthropic_path),
            "validation_dataset": (
                str(output_dir / "validation_anthropic.jsonl")
                if validation_dataset_path
                else None
            ),
        }


def fine_tune_openai(
    train_dataset_path: str = "data/fine_tuning/train_dataset.jsonl",
    validation_dataset_path: str = "data/fine_tuning/validation_dataset.jsonl",
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    suffix: str = "dagnostics-error-extractor",
    wait_for_completion: bool = True,
) -> Dict:
    """Fine-tune OpenAI model from prepared datasets"""

    if not Path(train_dataset_path).exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_dataset_path}\n"
            "Run 'python scripts/prepare_training_data.py' first"
        )

    tuner = OpenAIFineTuner(api_key=api_key)

    return tuner.fine_tune_from_dataset(
        train_dataset_path=train_dataset_path,
        validation_dataset_path=(
            validation_dataset_path if Path(validation_dataset_path).exists() else None
        ),
        model=model,
        suffix=suffix,
        wait_for_completion=wait_for_completion,
    )


def fine_tune_anthropic(
    train_dataset_path: str = "data/fine_tuning/train_dataset.jsonl",
    validation_dataset_path: str = "data/fine_tuning/validation_dataset.jsonl",
    model: str = "claude-3-haiku-20240307",
    api_key: Optional[str] = None,
) -> Dict:
    """Prepare data for Anthropic fine-tuning"""

    if not Path(train_dataset_path).exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_dataset_path}\n"
            "Run 'python scripts/prepare_training_data.py' first"
        )

    tuner = AnthropicFineTuner(api_key=api_key)

    return tuner.fine_tune_from_dataset(
        train_dataset_path=train_dataset_path,
        validation_dataset_path=(
            validation_dataset_path if Path(validation_dataset_path).exists() else None
        ),
        model=model,
    )


if __name__ == "__main__":
    # Example usage
    try:
        print("🔄 Starting OpenAI fine-tuning...")
        result = fine_tune_openai()

        if result["status"] == "succeeded":
            print(f"✅ Fine-tuning completed!")
            print(f"📦 Model: {result['fine_tuned_model']}")
            print(f"🎯 Tokens trained: {result['trained_tokens']:,}")
        else:
            print(f"❌ Fine-tuning failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Fine-tuning error: {e}", exc_info=True)
