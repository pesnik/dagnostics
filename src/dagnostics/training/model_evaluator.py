"""
Model Evaluation and Validation

Comprehensive evaluation suite for fine-tuned models including
accuracy metrics, error analysis, and production readiness assessment.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _import_optional_dependency(name: str, extra: str = ""):
    """Import optional dependency with helpful error message."""
    msg = (
        f"Missing optional dependency '{name}'. {extra}\n"
        f"Install with: pip install 'dagnostics[finetuning]' or pip install {name}"
    )
    try:
        module = __import__(name)
        return module
    except ImportError:
        raise ImportError(msg) from None


class ModelEvaluator:
    """Evaluate fine-tuned models on error extraction tasks"""

    def __init__(self, model_path: str, model_type: str = "local"):
        self.model_path = model_path
        self.model_type = model_type  # "local", "openai", "anthropic"

        if model_type == "local":
            self._setup_local_model()
        elif model_type == "openai":
            self._setup_openai_client()
        elif model_type == "anthropic":
            self._setup_anthropic_client()

    def _setup_local_model(self):
        """Setup local transformers model"""
        try:
            transformers = _import_optional_dependency(
                "transformers", "Install with: pip install transformers"
            )
            torch = _import_optional_dependency(
                "torch", "Install with: pip install torch"
            )

            logger.info(f"Loading local model from: {self.model_path}")

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def _setup_openai_client(self):
        """Setup OpenAI client"""
        openai = _import_optional_dependency(
            "openai", "Install with: pip install openai"
        )
        self.client = openai.OpenAI()
        self.model_name = self.model_path  # For OpenAI, this is the model ID

    def _setup_anthropic_client(self):
        """Setup Anthropic client"""
        anthropic = _import_optional_dependency(
            "anthropic", "Install with: pip install anthropic"
        )
        self.client = anthropic.Anthropic()
        self.model_name = self.model_path  # For Anthropic, this is the model name

    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from model"""

        if self.model_type == "local":
            return self._generate_local(prompt, max_length)
        elif self.model_type == "openai":
            return self._generate_openai(prompt)
        elif self.model_type == "anthropic":
            return self._generate_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _generate_local(self, prompt: str, max_length: int) -> str:
        """Generate from local model"""
        try:
            # Format prompt
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

            # Tokenize
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()

            return response

        except Exception as e:
            logger.error(f"Local generation error: {e}")
            return f"Error: {e}"

    def _generate_openai(self, prompt: str) -> str:
        """Generate from OpenAI model"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data engineer analyzing Airflow ETL task failure logs. Extract the core error message from log context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"Error: {e}"

    def _generate_anthropic(self, prompt: str) -> str:
        """Generate from Anthropic model"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=512,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return f"Error: {e}"

    def evaluate_on_dataset(self, test_dataset_path: str) -> Dict:
        """Evaluate model on test dataset"""

        logger.info(f"Evaluating model on dataset: {test_dataset_path}")

        if not Path(test_dataset_path).exists():
            raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")

        # Load test data
        test_examples = []
        with open(test_dataset_path, "r") as f:
            for line in f:
                test_examples.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(test_examples)} test examples")

        # Evaluate each example
        results = []
        correct_extractions = 0
        exact_matches = 0

        for i, example in enumerate(test_examples):
            logger.info(f"Evaluating example {i+1}/{len(test_examples)}")

            # Create prompt
            prompt = f"{example['instruction']}\n\n{example['input']}"

            # Generate response
            generated = self.generate_response(prompt)
            expected = example["output"]

            # Evaluate
            evaluation = self._evaluate_single_example(generated, expected)

            results.append(
                {
                    "example_id": i,
                    "prompt": prompt,
                    "generated": generated,
                    "expected": expected,
                    "evaluation": evaluation,
                }
            )

            if evaluation["correct_extraction"]:
                correct_extractions += 1
            if evaluation["exact_match"]:
                exact_matches += 1

        # Calculate metrics
        accuracy = correct_extractions / len(test_examples)
        exact_match_rate = exact_matches / len(test_examples)

        evaluation_results = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "test_dataset": test_dataset_path,
            "num_examples": len(test_examples),
            "accuracy": accuracy,
            "exact_match_rate": exact_match_rate,
            "correct_extractions": correct_extractions,
            "exact_matches": exact_matches,
            "results": results,
            "evaluated_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Evaluation completed: Accuracy = {accuracy:.2%}, Exact Match = {exact_match_rate:.2%}"
        )
        return evaluation_results

    def _evaluate_single_example(self, generated: str, expected: str) -> Dict:
        """Evaluate a single example"""

        # Clean responses for comparison
        generated_clean = self._clean_response(generated)
        expected_clean = self._clean_response(expected)

        # Check for exact match
        exact_match = generated_clean.lower() == expected_clean.lower()

        # Check for correct error extraction (more lenient)
        correct_extraction = self._check_error_extraction(
            generated_clean, expected_clean
        )

        # Calculate similarity score
        similarity = self._calculate_similarity(generated_clean, expected_clean)

        return {
            "exact_match": exact_match,
            "correct_extraction": correct_extraction,
            "similarity_score": similarity,
            "generated_clean": generated_clean,
            "expected_clean": expected_clean,
        }

    def _clean_response(self, response: str) -> str:
        """Clean response for evaluation"""
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", response.strip())

        # Remove common prefixes/suffixes
        prefixes = ["Error:", "The error is:", "Core error:", "Error message:"]
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()

        return cleaned

    def _check_error_extraction(self, generated: str, expected: str) -> bool:
        """Check if the core error was correctly extracted"""

        # Extract key error terms from expected
        expected_terms = self._extract_error_terms(expected)

        # Check if most important terms are present in generated
        found_terms = 0
        for term in expected_terms:
            if term.lower() in generated.lower():
                found_terms += 1

        # Consider correct if at least 70% of key terms are found
        return len(expected_terms) > 0 and (found_terms / len(expected_terms)) >= 0.7

    def _extract_error_terms(self, error_text: str) -> List[str]:
        """Extract key error terms"""
        # Common error patterns
        error_patterns = [
            r"TPT\d+",  # TPT errors
            r"Error \d+",  # Numbered errors
            r"timeout|connection|failed|missing|not found|deadlock",
            r"[A-Z_]+\[[^\]]+\]",  # Error codes in brackets
        ]

        terms = []
        for pattern in error_patterns:
            matches = re.findall(pattern, error_text, re.IGNORECASE)
            terms.extend(matches)

        # Also extract quoted strings and error codes
        quoted = re.findall(r'"([^"]*)"', error_text)
        terms.extend(quoted)

        return list(set(terms))  # Remove duplicates

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity score"""
        # Simple token-based similarity
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)

    def save_evaluation_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to: {output_path}")

    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate human-readable evaluation report"""

        report = f"""
# Model Evaluation Report

**Model**: {results['model_path']}
**Model Type**: {results['model_type']}
**Test Dataset**: {results['test_dataset']}
**Evaluated At**: {results['evaluated_at']}

## Overall Metrics

- **Test Examples**: {results['num_examples']:,}
- **Accuracy**: {results['accuracy']:.2%} ({results['correct_extractions']}/{results['num_examples']})
- **Exact Match Rate**: {results['exact_match_rate']:.2%} ({results['exact_matches']}/{results['num_examples']})

## Performance Analysis

### Top 5 Best Performing Examples
"""

        # Sort by similarity score
        sorted_results = sorted(
            results["results"],
            key=lambda x: x["evaluation"]["similarity_score"],
            reverse=True,
        )

        for i, result in enumerate(sorted_results[:5]):
            eval_data = result["evaluation"]
            report += f"""
**Example {result['example_id']+1}**
- Similarity: {eval_data['similarity_score']:.2%}
- Exact Match: {"✅" if eval_data['exact_match'] else "❌"}
- Correct Extraction: {"✅" if eval_data['correct_extraction'] else "❌"}
"""

        report += "\n### Top 5 Worst Performing Examples\n"

        for i, result in enumerate(sorted_results[-5:]):
            eval_data = result["evaluation"]
            report += f"""
**Example {result['example_id']+1}**
- Similarity: {eval_data['similarity_score']:.2%}
- Exact Match: {"✅" if eval_data['exact_match'] else "❌"}
- Correct Extraction: {"✅" if eval_data['correct_extraction'] else "❌"}
- Generated: `{result['generated'][:100]}...`
- Expected: `{result['expected'][:100]}...`
"""

        return report


def evaluate_model(
    model_path: str,
    test_dataset_path: str = "data/fine_tuning/validation_dataset.jsonl",
    model_type: str = "local",
    output_dir: str = "evaluations",
) -> str:
    """Evaluate a fine-tuned model"""

    logger.info(f"Starting model evaluation: {model_path}")

    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, model_type)

    # Run evaluation
    results = evaluator.evaluate_on_dataset(test_dataset_path)

    # Save results
    output_path = (
        Path(output_dir) / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    evaluator.save_evaluation_results(results, str(output_path))

    # Generate report
    report = evaluator.generate_evaluation_report(results)
    report_path = output_path.with_suffix(".md")

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n📊 Evaluation completed!")
    print(f"🎯 Accuracy: {results['accuracy']:.2%}")
    print(f"✅ Exact Match Rate: {results['exact_match_rate']:.2%}")
    print(f"📁 Results: {output_path}")
    print(f"📋 Report: {report_path}")

    return str(output_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python model_evaluator.py <model_path> [test_dataset_path] [model_type]"
        )
        sys.exit(1)

    model_path = sys.argv[1]
    test_dataset = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "data/fine_tuning/validation_dataset.jsonl"
    )
    model_type = sys.argv[3] if len(sys.argv) > 3 else "local"

    try:
        evaluate_model(model_path, test_dataset, model_type)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)
