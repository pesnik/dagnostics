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

try:
    import pandas  # noqa: F401

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

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
        self,
        raw_data_path: str = "data/training_data.jsonl",
        feedback_data_path: str = "data/feedback_data.jsonl",
        output_dir: str = "data/training",
    ):
        self.raw_data_path = Path(raw_data_path)
        self.feedback_data_path = Path(feedback_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_raw_training_data(self) -> List[Dict]:
        """Load existing training data from JSONL"""
        examples = []
        if self.raw_data_path.exists():
            with open(self.raw_data_path, "r") as f:
                for line in f:
                    examples.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(examples)} raw training examples")
        return examples

    def load_feedback_data(self) -> List[Dict]:
        """Load user feedback corrections"""
        feedback = []
        if self.feedback_data_path.exists():
            with open(self.feedback_data_path, "r") as f:
                for line in f:
                    feedback.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(feedback)} feedback examples")
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

        # Process raw training data
        raw_data = self.load_raw_training_data()
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
        feedback_data = self.load_feedback_data()
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
