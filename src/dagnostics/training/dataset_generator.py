"""
Dataset Generator for Fine-tuning SLMs on Error Analysis

Converts user feedback and raw logs into training datasets suitable for
fine-tuning small language models on DAGnostics error analysis tasks.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TrainingExample(BaseModel):
    """Single training example for fine-tuning"""

    instruction: str
    input: str
    output: str
    metadata: Dict[str, str]


class DatasetGenerator:
    """Generate training datasets from user feedback and logs"""

    def __init__(
        self, output_dir: str = "data/training", analyzer=None, feedback_candidates=None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.analyzer = analyzer
        self.feedback_candidates = feedback_candidates or []

    def get_failed_tasks_data(self, ndays: int = 1) -> List[Dict]:
        """Generate raw training data from live Airflow failed tasks"""
        if not self.analyzer:
            logger.warning("No analyzer provided, returning empty dataset")
            return []

        examples = []
        try:
            # Get failed tasks from Airflow directly
            failed_tasks = self.analyzer.airflow_client.get_failed_tasks(
                ndays * 24 * 60
            )

            if not failed_tasks:
                logger.info("No failed tasks found in Airflow")
                return []

            logger.info(f"Found {len(failed_tasks)} failed tasks from Airflow")

            for task in failed_tasks:
                try:
                    # Get task tries
                    task_tries = self.analyzer.airflow_client.get_task_tries(
                        task.dag_id, task.task_id, task.run_id
                    )

                    # Filter failed tries
                    failed_tries = [
                        try_instance
                        for try_instance in task_tries
                        if try_instance.state == "failed"
                        and try_instance.try_number > 0
                    ]

                    for failed_try in failed_tries:
                        try:
                            # Extract error using your existing method
                            _, candidates, error_message = (
                                self.analyzer.extract_task_error_for_sms(
                                    failed_try.dag_id,
                                    failed_try.task_id,
                                    failed_try.run_id,
                                    failed_try.try_number,
                                )
                            )

                            if candidates and error_message:
                                error_logs = "\n".join(
                                    [error.message for error in candidates]
                                )
                                examples.append(
                                    {
                                        "candidates": error_logs,
                                        "error": error_message,
                                        "dag_id": failed_try.dag_id,
                                        "task_id": failed_try.task_id,
                                        "run_id": failed_try.run_id,
                                        "try_number": failed_try.try_number,
                                    }
                                )

                        except Exception as e:
                            logger.warning(
                                f"Error processing try {failed_try.try_number}: {e}"
                            )
                            continue

                except Exception as e:
                    logger.warning(
                        f"Error processing task {task.dag_id}.{task.task_id}: {e}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Error fetching failed tasks: {e}")

        logger.info(f"Generated {len(examples)} training examples from live data")
        return examples

    def get_feedback_data(self) -> List[Dict]:
        """Get feedback data from web UI candidates"""
        feedback = []

        for candidate in self.feedback_candidates:
            # Only include reviewed candidates with human corrections
            if candidate.get("status") in [
                "approved",
                "rejected",
                "modified",
            ] and candidate.get("human_category"):
                feedback.append(
                    {
                        "log_context": candidate.get("raw_logs", ""),
                        "original_error": candidate.get("error_message", ""),
                        "corrected_analysis": {
                            "error_message": candidate.get("error_message", ""),
                            "category": candidate.get("human_category")
                            or candidate.get("llm_category"),
                            "severity": candidate.get("human_severity")
                            or candidate.get("llm_severity"),
                            "confidence": (
                                1.0 if candidate.get("status") == "approved" else 0.9
                            ),
                            "reasoning": candidate.get("human_feedback")
                            or "Human correction",
                            "error_lines": [candidate.get("error_message", "")],
                        },
                        "user_id": candidate.get("reviewed_by", "web_annotator"),
                        "confidence_rating": (
                            1.0 if candidate.get("status") == "approved" else 0.9
                        ),
                        "dag_id": candidate.get("dag_id", "unknown"),
                        "task_id": candidate.get("task_id", "unknown"),
                        "run_id": candidate.get("run_id", "unknown"),
                        "feedback_type": candidate.get("status"),
                    }
                )

        logger.info(f"Generated {len(feedback)} feedback examples from web UI")
        return feedback

    def create_instruction_dataset(self) -> List[TrainingExample]:
        """Create instruction-following dataset for fine-tuning"""

        base_instruction = (
            "You are an expert data engineer analyzing Airflow ETL task failure logs "
            "from a telecom data warehouse. Analyze the log and extract the root cause error.\n\n"
            "Common error patterns:\n"
            "- TPT (Teradata Parallel Transporter) errors: Configuration issues, command line problems\n"
            "- SSH/SFTP timeouts: Network connectivity to data sources\n"
            "- Missing data files: Upstream data dependencies (MSISDN files, reports)\n"
            "- Teradata database issues: Deadlocks, hostname lookups, connection failures\n"
            "- BTEQ command failures: SQL execution problems\n\n"
            "Respond with JSON containing error_message, confidence, category, severity, reasoning, and error_lines."
        )

        examples = []

        # Process raw training data from live Airflow
        raw_data = self.get_failed_tasks_data()
        for item in raw_data:
            # Create structured training example
            log_context = item.get("candidates", "")
            expected_error = item.get("error", "")

            # Create more sophisticated expected output
            expected_output = self._create_structured_output(
                expected_error, log_context
            )

            example = TrainingExample(
                instruction=base_instruction,
                input=f"Log Context:\n{log_context}",
                output=expected_output,
                metadata={
                    "source": "raw_training_data",
                    "created_at": datetime.now().isoformat(),
                },
            )
            examples.append(example)

        # Process user feedback corrections
        feedback_data = self.get_feedback_data()
        for item in feedback_data:
            log_context = item.get("log_context", "")
            corrected_analysis = item.get("corrected_analysis", {})

            example = TrainingExample(
                instruction=base_instruction,
                input=f"Log Context:\n{log_context}",
                output=json.dumps(corrected_analysis, indent=2),
                metadata={
                    "source": "user_feedback",
                    "user_id": item.get("user_id", "unknown"),
                    "feedback_confidence": str(item.get("confidence_rating", 0)),
                    "created_at": datetime.now().isoformat(),
                },
            )
            examples.append(example)

        logger.info(f"Created {len(examples)} instruction training examples")
        return examples

    def _create_structured_output(self, error_message: str, log_context: str) -> str:
        """Create structured JSON output from basic error message"""

        # Simple heuristics to categorize errors
        category = self._categorize_error(error_message.lower())
        severity = self._assess_severity(error_message.lower(), log_context.lower())
        confidence = self._estimate_confidence(error_message, log_context)

        structured_output = {
            "error_message": error_message,
            "confidence": confidence,
            "category": category,
            "severity": severity,
            "reasoning": self._generate_reasoning(error_message, category),
            "error_lines": [error_message],  # Simplified
        }

        return json.dumps(structured_output, indent=2)

    def _categorize_error(self, error_text: str) -> str:
        """Simple rule-based categorization"""
        if any(term in error_text for term in ["tpt", "tbuild", "command line"]):
            return "configuration_error"
        elif any(term in error_text for term in ["timeout", "connection timed out"]):
            return "timeout_error"
        elif any(
            term in error_text for term in ["no such file", "missing", "not found"]
        ):
            return "data_quality"
        elif any(term in error_text for term in ["deadlock", "abort"]):
            return "resource_error"
        elif any(term in error_text for term in ["hostname", "lookup", "connection"]):
            return "configuration_error"
        else:
            return "unknown"

    def _assess_severity(self, error_text: str, _log_context: str) -> str:
        """Simple severity assessment"""
        if any(term in error_text for term in ["critical", "fatal", "hostname lookup"]):
            return "high"
        elif any(term in error_text for term in ["timeout", "deadlock"]):
            return "medium"
        else:
            return "medium"

    def _estimate_confidence(self, error_message: str, _log_context: str) -> float:
        """Estimate confidence based on message clarity"""
        if len(error_message) > 10 and "error" in error_message.lower():
            return 0.85
        elif len(error_message) > 5:
            return 0.70
        else:
            return 0.50

    def _generate_reasoning(self, _error_message: str, category: str) -> str:
        """Generate reasoning text"""
        reasoning_map = {
            "configuration_error": "Configuration or setup issue detected",
            "timeout_error": "Network timeout indicates connectivity problems",
            "data_quality": "Missing or malformed data dependency",
            "resource_error": "Database resource contention detected",
            "unknown": "Error type requires further investigation",
        }
        return reasoning_map.get(category, "Error analysis needed")

    def save_dataset(
        self,
        examples: List[TrainingExample],
        filename: str = "fine_tuning_dataset.jsonl",
    ):
        """Save dataset in JSONL format for training"""
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            for example in examples:
                # Convert to HuggingFace format
                hf_format = {
                    "instruction": example.instruction,
                    "input": example.input,
                    "output": example.output,
                    "metadata": example.metadata,
                }
                f.write(json.dumps(hf_format) + "\n")

        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return str(output_path)

    def create_validation_split(
        self, examples: List[TrainingExample], test_ratio: float = 0.2
    ) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """Split dataset into train/validation sets"""
        import random

        random.shuffle(examples)

        split_idx = int(len(examples) * (1 - test_ratio))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        logger.info(
            f"Split: {len(train_examples)} train, {len(val_examples)} validation"
        )
        return train_examples, val_examples

    def generate_full_dataset(self) -> Dict[str, str]:
        """Generate complete training dataset with train/val split"""

        # Create instruction dataset
        examples = self.create_instruction_dataset()

        if not examples:
            logger.warning("No training examples generated")
            return {}

        # Split into train/validation
        train_examples, val_examples = self.create_validation_split(examples)

        # Save datasets
        train_path = self.save_dataset(train_examples, "train_dataset.jsonl")
        val_path = self.save_dataset(val_examples, "validation_dataset.jsonl")

        # Create dataset info
        info = {
            "train_path": train_path,
            "validation_path": val_path,
            "train_size": len(train_examples),
            "validation_size": len(val_examples),
            "total_size": len(examples),
            "created_at": datetime.now().isoformat(),
        }

        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Dataset generation complete: {info}")
        return info


def main():
    """Generate training dataset from existing data"""
    generator = DatasetGenerator()
    dataset_info = generator.generate_full_dataset()

    if dataset_info:
        print("Training dataset created:")
        print(f"  Train: {dataset_info['train_size']} examples")
        print(f"  Validation: {dataset_info['validation_size']} examples")
        print(
            f"  Paths: {dataset_info['train_path']}, {dataset_info['validation_path']}"
        )
    else:
        print("No training data available. Add raw logs or user feedback first.")


if __name__ == "__main__":
    main()
