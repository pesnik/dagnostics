"""
SLM Fine-tuner for DAGnostics Error Analysis

Implements automated fine-tuning pipeline for small language models
specialized in Airflow error analysis using LoRA and QLoRA techniques.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    # Type-only imports - available for type checking but not at runtime
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _import_optional_dependency(name: str, extra: str = ""):
    """
    Import optional dependency with helpful error message.

    This is the standard pattern used by pandas, scipy, etc.
    """
    msg = (
        f"Missing optional dependency '{name}'. {extra}\n"
        f"Install with: pip install 'dagnostics[finetuning]' or pip install {name}"
    )
    try:
        module = __import__(name)
        return module
    except ImportError:
        raise ImportError(msg) from None


def _check_ml_dependencies():
    """Check if ML training dependencies are available"""
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True, None
    except ImportError as e:
        return False, str(e)


HAS_ML_DEPS, MISSING_DEPS = _check_ml_dependencies()


class SLMFineTuner:
    """Fine-tune small language models for error analysis"""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        output_dir: str = "models/fine_tuned",
        use_quantization: bool = True,
        force_cpu: bool = False,
    ):

        if not HAS_ML_DEPS:
            raise ImportError(
                f"Fine-tuning dependencies not available: {MISSING_DEPS}\n"
                "Install with: pip install 'dagnostics[finetuning]' for all fine-tuning dependencies\n"
                "Or install individually: pip install torch transformers datasets peft bitsandbytes"
            )

        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_quantization = use_quantization
        self.force_cpu = force_cpu

        # Model will be loaded during training
        self.model: Optional["AutoModelForCausalLM"] = None
        self.tokenizer: Optional["AutoTokenizer"] = None

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with quantization if enabled"""

        # Import dependencies at runtime
        torch = _import_optional_dependency("torch", "Install with: pip install torch")
        transformers = _import_optional_dependency(
            "transformers", "Install with: pip install transformers"
        )
        peft = _import_optional_dependency("peft", "Install with: pip install peft")

        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        # Add pad token if it doesn't exist
        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure device and data type based on CPU/GPU mode
        if self.force_cpu:
            device_map = None
            torch_dtype = torch.float32  # CPU works better with float32
            use_quantization = False  # Disable quantization for CPU
            logger.info("Using CPU mode - disabling quantization and GPU optimizations")
        else:
            device_map = "auto"
            torch_dtype = torch.float16
            use_quantization = self.use_quantization

        # Quantization config for memory efficiency (GPU only)
        if use_quantization and not self.force_cpu:
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )

            # Prepare for k-bit training
            self.model = peft.prepare_model_for_kbit_training(self.model)
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

            # Move to CPU if forced
            if self.force_cpu and device_map is None:
                self.model = self.model.to("cpu")

        logger.info("Model and tokenizer loaded successfully")

    def setup_lora_config(self):
        """Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning"""

        # Import peft at runtime
        peft = _import_optional_dependency("peft", "Install with: pip install peft")

        lora_config = peft.LoraConfig(
            r=16,  # Rank of adaptation
            lora_alpha=32,  # LoRA scaling parameter
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",  # Attention layers
                "gate_proj",
                "up_proj",
                "down_proj",  # MLP layers
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        return lora_config

    def prepare_dataset(self, dataset_path: str):
        """Load and prepare dataset for training"""

        # Import datasets at runtime
        datasets = _import_optional_dependency(
            "datasets", "Install with: pip install datasets"
        )

        logger.info(f"Loading dataset from: {dataset_path}")

        # Load JSONL dataset
        dataset = datasets.load_dataset("json", data_files=dataset_path, split="train")

        def format_example(example):
            """Format training example as instruction-following conversation"""

            # Create a conversation format
            instruction = example["instruction"]
            input_text = example["input"]
            output_text = example["output"]

            # Format as chat-like interaction
            formatted_text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""

            return {"text": formatted_text}

        # Format all examples
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)

        def tokenize_function(examples):
            """Tokenize the formatted text"""

            # Tokenize with truncation and padding
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer must be initialized")
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=1024,  # Adjust based on your needs
                return_tensors="pt",
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )

        logger.info(f"Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset

    def train_model(
        self,
        train_dataset_path: str,
        validation_dataset_path: Optional[str] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        model_output_name: str = "dagnostics-error-extractor",
    ) -> str:
        """Fine-tune the model"""

        # Import training dependencies at runtime
        peft = _import_optional_dependency("peft", "Install with: pip install peft")
        transformers = _import_optional_dependency(
            "transformers", "Install with: pip install transformers"
        )

        # Setup model and tokenizer
        if self.model is None:
            self.setup_model_and_tokenizer()

        # Setup LoRA (skip for CPU-only mode with basic models)
        if not self.force_cpu or "DialoGPT" not in self.model_name:
            lora_config = self.setup_lora_config()
            self.model = peft.get_peft_model(self.model, lora_config)
        else:
            logger.info(
                "Skipping LoRA for CPU mode with basic model - using full fine-tuning"
            )

        # Print trainable parameters
        trainable_params = 0
        total_params = 0
        if self.model is not None:
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)"
            )

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset_path)
        eval_dataset = None
        if validation_dataset_path:
            eval_dataset = self.prepare_dataset(validation_dataset_path)

        # Training arguments
        training_args = transformers.TrainingArguments(
            output_dir=str(
                self.output_dir
                / f"checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            optim="adamw_hf",
            num_train_epochs=num_epochs,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            logging_steps=10,
            save_steps=100,
            learning_rate=learning_rate,
            weight_decay=0.01,
            fp16=not self.force_cpu,  # Disable fp16 for CPU
            bf16=False,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="none",  # Disable wandb/tensorboard for simplicity
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
        )

        # Data collator
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        # Initialize trainer
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Start training
        logger.info("Starting fine-tuning...")
        trainer.train()

        # Save the final model
        final_model_path = (
            self.output_dir
            / f"{model_output_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        trainer.save_model(str(final_model_path))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(final_model_path))

        # Save training info
        training_info = {
            "model_name": self.model_name,
            "final_model_path": str(final_model_path),
            "training_args": training_args.to_dict(),
            "lora_config": lora_config.to_dict(),
            "train_dataset_path": train_dataset_path,
            "validation_dataset_path": validation_dataset_path,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "training_completed_at": datetime.now().isoformat(),
        }

        info_path = final_model_path / "training_info.json"
        with open(info_path, "w") as f:
            json.dump(training_info, f, indent=2)

        logger.info(f"Training completed! Model saved to: {final_model_path}")
        return str(final_model_path)

    def evaluate_model(self, model_path: str, test_dataset_path: str) -> Dict:
        """Evaluate fine-tuned model on test set"""

        # Import dependencies at runtime
        torch = _import_optional_dependency("torch", "Install with: pip install torch")
        transformers = _import_optional_dependency(
            "transformers", "Install with: pip install transformers"
        )

        logger.info(f"Evaluating model: {model_path}")

        # Load the fine-tuned model
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        # tokenizer = AutoTokenizer.from_pretrained(model_path)  # Not used in evaluation

        # Load test dataset
        test_dataset = self.prepare_dataset(test_dataset_path)

        # Simple evaluation - compute perplexity
        model.eval()
        total_loss = 0
        num_examples = 0

        with torch.no_grad():
            for example in test_dataset:
                inputs = {
                    k: v.unsqueeze(0) for k, v in example.items() if k != "labels"
                }
                labels = example["labels"].unsqueeze(0)

                outputs = model(**inputs, labels=labels)
                total_loss += outputs.loss.item()
                num_examples += 1

        avg_loss = total_loss / num_examples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        eval_results = {
            "average_loss": avg_loss,
            "perplexity": perplexity,
            "num_test_examples": num_examples,
            "model_path": model_path,
            "evaluated_at": datetime.now().isoformat(),
        }

        # Save evaluation results
        eval_path = Path(model_path) / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"Evaluation completed: Perplexity = {perplexity:.2f}")
        return eval_results

    def export_for_ollama(
        self, model_path: str, output_name: str = "dagnostics-slm"
    ) -> str:
        """Export fine-tuned model for Ollama deployment"""

        logger.info(f"Exporting model for Ollama: {model_path}")

        # This is a simplified approach - actual GGUF conversion requires additional tools
        # like llama.cpp or specific conversion scripts

        export_dir = self.output_dir / "ollama_export" / output_name
        export_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        model_path_obj = Path(model_path)
        for file in model_path_obj.glob("*"):
            if file.is_file():
                shutil.copy2(file, export_dir / file.name)

        # Create Ollama Modelfile
        modelfile_content = f"""FROM {export_dir}
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40

TEMPLATE \"\"\"### Instruction:
{{{{ .System }}}}

### Input:
{{{{ .Prompt }}}}

### Response:
\"\"\"

SYSTEM \"\"\"You are an expert data engineer analyzing Airflow ETL task failure logs from a telecom data warehouse.
Analyze logs and extract root cause errors in JSON format.\"\"\"
"""

        modelfile_path = export_dir / "Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        # Create deployment instructions
        instructions = f"""# Deploy to Ollama

1. Build the model:
   ```bash
   cd {export_dir}
   ollama create {output_name} -f Modelfile
   ```

2. Test the model:
   ```bash
   ollama run {output_name}
   ```

3. Use in DAGnostics:
   ```yaml
   llm:
     default_provider: "ollama"
     providers:
       ollama:
         base_url: "http://localhost:11434"
         model: "{output_name}"
   ```
"""

        instructions_path = export_dir / "README.md"
        with open(instructions_path, "w") as f:
            f.write(instructions)

        logger.info(f"Model exported for Ollama: {export_dir}")
        return str(export_dir)


def train_from_prepared_data(
    model_name: str = "microsoft/DialoGPT-small",
    train_dataset_path: str = "data/fine_tuning/train_dataset.jsonl",
    validation_dataset_path: str = "data/fine_tuning/validation_dataset.jsonl",
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    model_output_name: str = "dagnostics-error-extractor",
    use_quantization: bool = True,
    export_for_ollama: bool = True,
    force_cpu: bool = False,
) -> str:
    """Train model from prepared fine-tuning datasets"""

    logger.info(f"Starting fine-tuning with prepared data from {train_dataset_path}")

    # Initialize fine-tuner
    fine_tuner = SLMFineTuner(
        model_name=model_name, use_quantization=use_quantization, force_cpu=force_cpu
    )

    # Check if training datasets exist
    if not Path(train_dataset_path).exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_dataset_path}\n"
            "Run 'dagnostics training prepare-data' first to create training datasets"
        )

    val_path = (
        validation_dataset_path if Path(validation_dataset_path).exists() else None
    )
    if not val_path:
        logger.warning(f"Validation dataset not found: {validation_dataset_path}")

    # Fine-tune the model
    model_path = fine_tuner.train_model(
        train_dataset_path=train_dataset_path,
        validation_dataset_path=val_path,
        num_epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_output_name=model_output_name,
    )

    # Evaluate the model
    if val_path:
        logger.info("Evaluating model on validation set...")
        eval_results = fine_tuner.evaluate_model(model_path, val_path)
        logger.info(
            f"Model evaluation completed: Perplexity = {eval_results['perplexity']:.2f}"
        )

    # Export for Ollama if requested
    if export_for_ollama:
        logger.info("Exporting model for Ollama deployment...")
        ollama_path = fine_tuner.export_for_ollama(model_path, model_output_name)
        logger.info(f"Model ready for Ollama deployment: {ollama_path}")

    logger.info(f"Fine-tuning completed! Model saved to: {model_path}")
    return model_path


def main():
    """Example usage of the fine-tuner with prepared datasets"""
    try:
        model_path = train_from_prepared_data(
            model_name="microsoft/DialoGPT-small",
            epochs=3,
            batch_size=2,
            use_quantization=True,
            export_for_ollama=True,
            force_cpu=False,  # Set to True for CPU-only training
        )
        print("\n✅ Fine-tuning completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print("\n🚀 To use the model:")
        print(
            "   1. For Ollama: Check the deployment instructions in the model directory"
        )
        print(
            "   2. For local inference: Use the model path directly with transformers"
        )

    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        logger.error(f"Fine-tuning error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
