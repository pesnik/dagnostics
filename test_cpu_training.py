#!/usr/bin/env python3
"""
Test script for CPU fallback training
Creates a minimal dataset and tests the CPU training workflow
"""

import json
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_dataset(output_path: str, num_examples: int = 5):
    """Create a minimal test dataset for CPU training"""

    test_examples = [
        {
            "instruction": "Analyze the following Airflow task failure log and extract the primary error.",
            "input": (
                "[2025-08-17T10:01:00.000+0600] {ttu.py:97} INFO - *** Failure 2450 CREATE_TABLE:Transaction ABORTed due to Deadlock.\n"
                "[2025-08-17T10:01:00.100+0600] {logging_mixin.py:190} INFO - BTEQ command exited with return code 45\n"
                "[2025-08-17T10:01:00.200+0600] {standard_task_runner.py:124} ERROR - Failed to execute job 67890 for task create_table"
            ),
            "output": "Transaction ABORTed due to Deadlock",
        },
        {
            "instruction": "Analyze the following Airflow task failure log and extract the primary error.",
            "input": (
                "[2025-08-17T10:02:00.000+0600] {logging_mixin.py:190} INFO - mget: Access failed: No such file (DATA_20250817.csv)\n"
                "[2025-08-17T10:02:00.100+0600] {subprocess.py:97} INFO - Command exited with return code 1\n"
                "[2025-08-17T10:02:00.200+0600] {standard_task_runner.py:124} ERROR - Failed to execute job 11111 for task download_data"
            ),
            "output": "mget: Access failed: No such file (DATA_20250817.csv)",
        },
        {
            "instruction": "Analyze the following Airflow task failure log and extract the primary error.",
            "input": (
                "[2025-08-17T10:03:00.000+0600] {logging_mixin.py:190} INFO - TPT Error: TPT_INFRA: TPT04183: Error: Line 1 of Command Line:\n"
                "[2025-08-17T10:03:00.100+0600] {logging_mixin.py:190} INFO - Return Code: 8\n"
                "[2025-08-17T10:03:00.200+0600] {standard_task_runner.py:124} ERROR - Failed to execute job 22222 for task tpt_export"
            ),
            "output": "TPT Error: TPT_INFRA: TPT04183: Error: Line 1 of Command Line:",
        },
        {
            "instruction": "Analyze the following Airflow task failure log and extract the primary error.",
            "input": (
                "[2025-08-17T10:04:00.000+0600] {subprocess.py:93} INFO - ls: write error: No space left on device\n"
                "[2025-08-17T10:04:00.100+0600] {logging_mixin.py:190} INFO - Bash command failed with return code 1\n"
                "[2025-08-17T10:04:00.200+0600] {standard_task_runner.py:124} ERROR - Failed to execute job 33333 for task check_disk"
            ),
            "output": "ls: write error: No space left on device",
        },
    ]

    # Cycle through examples to create the requested number
    examples = []
    for i in range(num_examples):
        examples.append(test_examples[i % len(test_examples)])

    # Write to JSONL format
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    logger.info(f"Created test dataset with {num_examples} examples: {output_path}")
    return output_path


def test_cpu_training():
    """Test the CPU training workflow"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test datasets
        train_dataset = temp_path / "train_dataset.jsonl"
        val_dataset = temp_path / "val_dataset.jsonl"

        create_test_dataset(str(train_dataset), num_examples=10)
        create_test_dataset(str(val_dataset), num_examples=3)

        logger.info("Testing CPU fine-tuning workflow...")

        try:
            from dagnostics.training.fine_tuner import train_from_prepared_data

            # Test with CPU mode and minimal settings
            model_path = train_from_prepared_data(
                model_name="microsoft/DialoGPT-small",  # Small model for testing
                train_dataset_path=str(train_dataset),
                validation_dataset_path=str(val_dataset),
                epochs=1,  # Just 1 epoch for testing
                learning_rate=2e-4,
                batch_size=1,  # Small batch size for CPU
                model_output_name="test-cpu-model",
                use_quantization=False,  # No quantization for CPU
                export_for_ollama=False,  # Skip export for testing
                force_cpu=True,  # Force CPU mode
            )

            logger.info("✅ CPU training test completed successfully!")
            logger.info("📁 Test model saved to: %s", model_path)
            return True

        except Exception as e:
            logger.error(f"❌ CPU training test failed: {e}")
            return False


if __name__ == "__main__":
    print("🧪 Testing CPU fallback training...")
    success = test_cpu_training()

    if success:
        print(
            "\n✅ CPU fallback training works! You can now run fine-tuning on CPU-only machines."
        )
        print("\n🚀 To use CPU training in production:")
        print(
            "   dagnostics training train-local --force-cpu --batch-size 1 --epochs 1"
        )
    else:
        print("\n❌ CPU fallback testing failed. Check the logs for details.")
