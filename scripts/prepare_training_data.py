#!/usr/bin/env python3
"""
Prepare training data from the collected dataset for fine-tuning.

This script processes the training_dataset JSON file and creates:
1. JSONL format for fine-tuning with input/output pairs
2. Filtered dataset containing only human-reviewed examples
3. Training/validation split
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_training_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load the training dataset from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def filter_human_reviewed(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to only include examples with human feedback."""
    return [
        item
        for item in data
        if item.get("human_feedback")
        and item["human_feedback"] != "null"
        and item["human_feedback"].strip() != ""
        and item.get("status") in ["approved", "modify", "modified"]
    ]


def create_fine_tuning_examples(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Create fine-tuning examples in instruction-response format."""
    examples = []

    for item in data:
        # Create instruction prompt
        instruction = (
            "You are an expert data engineer analyzing Airflow ETL task failure logs. "
            "Extract the core error message from the log context below. "
            "Focus on the essential technical error, removing timestamps and verbose details.\n\n"
            f"Log Context:\n{item['raw_logs']}\n\n"
            "Extract the core error message:"
        )

        # Use human feedback as the target response
        response = item["human_feedback"].strip()

        examples.append({"instruction": instruction, "input": "", "output": response})

    return examples


def create_jsonl_dataset(examples: List[Dict[str, str]], output_path: str):
    """Save examples in JSONL format for fine-tuning."""
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def split_train_val(examples: List[Dict[str, str]], train_ratio: float = 0.8):
    """Split data into training and validation sets."""
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]


def main():
    if len(sys.argv) != 2:
        print("Usage: python prepare_training_data.py <path_to_training_dataset.json>")
        sys.exit(1)

    input_file = sys.argv[1]

    if not Path(input_file).exists():
        print(f"Error: File {input_file} not found")
        sys.exit(1)

    print(f"📊 Loading training dataset from {input_file}")
    data = load_training_dataset(input_file)
    print(f"   Total examples: {len(data)}")

    print("🔍 Filtering human-reviewed examples")
    filtered_data = filter_human_reviewed(data)
    print(f"   Human-reviewed examples: {len(filtered_data)}")

    if len(filtered_data) < 10:
        print(
            "⚠️  Warning: Very few human-reviewed examples. Consider reviewing more data."
        )

    print("🛠  Creating fine-tuning examples")
    examples = create_fine_tuning_examples(filtered_data)

    print("📁 Splitting into train/validation sets")
    train_examples, val_examples = split_train_val(examples)
    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")

    # Create output directory
    output_dir = Path("data/fine_tuning")
    output_dir.mkdir(exist_ok=True)

    # Save datasets
    train_path = output_dir / "train_dataset.jsonl"
    val_path = output_dir / "validation_dataset.jsonl"
    full_path = output_dir / "full_dataset.jsonl"

    print("💾 Saving fine-tuning datasets")
    create_jsonl_dataset(train_examples, str(train_path))
    create_jsonl_dataset(val_examples, str(val_path))
    create_jsonl_dataset(examples, str(full_path))

    print(f"✅ Fine-tuning datasets saved:")
    print(f"   Training: {train_path} ({len(train_examples)} examples)")
    print(f"   Validation: {val_path} ({len(val_examples)} examples)")
    print(f"   Full dataset: {full_path} ({len(examples)} examples)")

    # Show sample
    if examples:
        print("\n📝 Sample fine-tuning example:")
        print("=" * 60)
        sample = examples[0]
        print(f"INSTRUCTION:\n{sample['instruction'][:200]}...")
        print(f"\nOUTPUT:\n{sample['output']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
