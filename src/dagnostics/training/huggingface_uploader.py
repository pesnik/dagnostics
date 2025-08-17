"""
HuggingFace Model Uploader

Handles uploading fine-tuned models to HuggingFace Hub with proper
metadata, model cards, and Ollama compatibility.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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


class HuggingFaceUploader:
    """Upload fine-tuned models to HuggingFace Hub"""

    def __init__(self, token: Optional[str] = None):
        self.huggingface_hub = _import_optional_dependency(
            "huggingface_hub", "Install with: pip install huggingface_hub"
        )
        
        # Initialize HF API
        if token:
            self.api = self.huggingface_hub.HfApi(token=token)
        else:
            # Will use HF_TOKEN env var or local token
            self.api = self.huggingface_hub.HfApi()

    def create_model_card(
        self,
        model_name: str,
        base_model: str,
        training_info: Dict,
        evaluation_results: Optional[Dict] = None,
    ) -> str:
        """Create a comprehensive model card"""
        
        # Calculate training metrics
        total_params = training_info.get("total_parameters", 0)
        trainable_params = training_info.get("trainable_parameters", 0)
        efficiency = (trainable_params / total_params * 100) if total_params > 0 else 0
        
        # Format evaluation metrics
        eval_section = ""
        if evaluation_results:
            eval_section = f"""
## Evaluation Results

- **Accuracy**: {evaluation_results.get('accuracy', 0):.2%}
- **Exact Match Rate**: {evaluation_results.get('exact_match_rate', 0):.2%}
- **Test Examples**: {evaluation_results.get('num_examples', 0):,}

### Performance Metrics
- **Correct Extractions**: {evaluation_results.get('correct_extractions', 0)}/{evaluation_results.get('num_examples', 0)}
- **Exact Matches**: {evaluation_results.get('exact_matches', 0)}/{evaluation_results.get('num_examples', 0)}
"""

        model_card = f"""---
license: mit
base_model: {base_model}
tags:
- dagnostics
- error-analysis
- etl-monitoring
- fine-tuned
- lora
- airflow
- telecom
datasets:
- custom
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# {model_name}

A fine-tuned language model specialized in ETL error analysis for telecom data pipelines.

## Model Description

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) specifically optimized for analyzing Airflow ETL task failure logs in a telecom environment. It excels at extracting core error messages from verbose log data, focusing on critical technical issues while filtering out noise.

### Model Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Efficiency**: {efficiency:.1f}% of parameters trained ({trainable_params:,}/{total_params:,})
- **Specialized For**: ETL error analysis, Airflow task failures, telecom data pipelines
- **Training Data**: {training_info.get('train_size', 'Unknown')} human-reviewed production error examples

### Key Capabilities

- **Error Extraction**: Identifies core technical errors from verbose logs
- **Pattern Recognition**: Understands common ETL failure patterns (TPT, SSH, database errors)
- **Noise Filtering**: Removes timestamps, debug info, and irrelevant details
- **Telecom Context**: Optimized for telecom-specific error patterns and terminology

## Intended Use

### Primary Use Cases

- **Production ETL Monitoring**: Real-time analysis of Airflow task failures
- **Error Categorization**: Automatic classification of ETL errors by type and severity
- **Alert Generation**: Concise error summaries for SMS/email notifications
- **Log Analysis**: Batch processing of historical failure logs

### Supported Error Types

- **TPT Errors**: Teradata Parallel Transporter configuration and execution issues
- **SSH/SFTP Timeouts**: Network connectivity problems with data sources
- **Database Issues**: Connection failures, deadlocks, hostname lookup errors
- **Data Quality**: Missing files, format issues, validation failures
- **Resource Errors**: Memory, disk space, and system resource problems

{eval_section}

## Technical Details

### Training Configuration

- **Epochs**: {training_info.get('epochs', 'Unknown')}
- **Learning Rate**: {training_info.get('learning_rate', 'Unknown')}
- **Batch Size**: {training_info.get('batch_size', 'Unknown')}
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Quantization**: 4-bit (QLoRA)

### Training Data

The model was trained on production data from a telecom data warehouse environment, including:

- **Human-reviewed examples**: Real ETL failures with expert annotations
- **Diverse error patterns**: Covering all major ETL failure categories
- **Production context**: Actual Banglalink telecom infrastructure logs

## Usage

### Direct Usage with Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModelForCausalLM.from_pretrained("{model_name}")

# Analyze an error log
log_context = '''
[2025-08-17T10:15:23.456+0600] {{logging_mixin.py:190}} INFO - Failure caused by TPT10508: RDBMS error 3803: Table 'X_STAGE_TABLE' already exists.
[2025-08-17T10:15:23.467+0600] {{standard_task_runner.py:124}} ERROR - Failed to execute job 15517343 for task load_staging_data
'''

prompt = f"Extract the core error message from this log:\\n\\n{{log_context}}\\n\\nCore error:"

inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=inputs.shape[1] + 50, temperature=0.1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("Core error:")[-1].strip())
```

### DAGnostics Integration

```bash
# Update your DAGnostics config to use the fine-tuned model
# config/config.yaml:
llm:
  default_provider: "ollama"
  providers:
    ollama:
      base_url: "http://localhost:11434"
      model: "{model_name.replace('/', '-')}"
```

### Ollama Deployment

```bash
# The model includes Ollama-compatible exports
# Deploy with:
ollama create {model_name.replace('/', '-')} -f /path/to/Modelfile
ollama run {model_name.replace('/', '-')} "Analyze this error log..."
```

## Performance

The model demonstrates strong performance on ETL error analysis tasks:

- **High Accuracy**: Consistently identifies correct error messages
- **Context Awareness**: Understands telecom-specific terminology and patterns
- **Efficient**: Optimized for real-time production use
- **Robust**: Handles diverse log formats and error types

## Limitations

- **Domain Specific**: Optimized for telecom ETL environments
- **Language**: Primarily English error messages
- **Format Dependency**: Best performance on Airflow-style log formats
- **Context Window**: Limited by base model's context length

## Training Details

### Dataset

- **Source**: Production Banglalink telecom data warehouse
- **Size**: {training_info.get('train_size', 'Unknown')} training examples
- **Quality**: Human-reviewed and validated error extractions
- **Coverage**: Comprehensive ETL failure scenarios

### Methodology

- **Fine-tuning Approach**: LoRA (Low-Rank Adaptation)
- **Memory Optimization**: QLoRA for efficient training
- **Instruction Format**: Custom instruction-response format for error analysis
- **Evaluation**: Comprehensive accuracy and similarity metrics

## Ethical Considerations

- **Privacy**: Trained on anonymized production logs
- **Bias**: May reflect patterns specific to telecom ETL environments
- **Use Cases**: Intended for technical error analysis, not human communication

## Citation

If you use this model in your research or production systems, please cite:

```bibtex
@misc{{{model_name.replace('/', '_').replace('-', '_')},
  title={{DAGnostics Fine-tuned Error Analysis Model}},
  author={{DAGnostics Team}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/{model_name}}}}},
}}
```

## Model Card Authors

DAGnostics Development Team

## Contact

For questions and support regarding this model, please open an issue in the [DAGnostics repository](https://github.com/your-org/dagnostics).

---

*This model was trained using the DAGnostics fine-tuning infrastructure. For more information about DAGnostics, visit the [project documentation](https://github.com/your-org/dagnostics).*
"""

        return model_card

    def prepare_upload_directory(
        self,
        model_path: str,
        model_name: str,
        training_info: Dict,
        evaluation_results: Optional[Dict] = None,
    ) -> str:
        """Prepare a directory for HuggingFace upload"""
        
        logger.info(f"Preparing upload directory for {model_name}")
        
        # Create upload directory
        upload_dir = Path("uploads") / model_name.replace("/", "_")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_path_obj = Path(model_path)
        for file_path in model_path_obj.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_path_obj)
                target_path = upload_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target_path)
        
        # Create model card
        base_model = training_info.get("model_name", "unknown")
        model_card = self.create_model_card(
            model_name, base_model, training_info, evaluation_results
        )
        
        with open(upload_dir / "README.md", "w") as f:
            f.write(model_card)
        
        # Create .gitattributes for LFS
        gitattributes = """*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
"""
        
        with open(upload_dir / ".gitattributes", "w") as f:
            f.write(gitattributes)
        
        # Create training metadata
        metadata = {
            "training_info": training_info,
            "evaluation_results": evaluation_results,
            "upload_timestamp": datetime.now().isoformat(),
            "dagnostics_version": "0.5.0",
        }
        
        with open(upload_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Upload directory prepared: {upload_dir}")
        return str(upload_dir)

    def upload_model(
        self,
        model_path: str,
        repo_id: str,
        training_info: Dict,
        evaluation_results: Optional[Dict] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> str:
        """Upload model to HuggingFace Hub"""
        
        logger.info(f"Uploading model to HuggingFace Hub: {repo_id}")
        
        # Prepare upload directory
        upload_dir = self.prepare_upload_directory(
            model_path, repo_id, training_info, evaluation_results
        )
        
        try:
            # Create repository if it doesn't exist
            try:
                self.api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
                logger.info(f"Repository created/verified: {repo_id}")
            except Exception as e:
                logger.warning(f"Repository creation warning: {e}")
            
            # Upload files
            if not commit_message:
                commit_message = f"Upload DAGnostics fine-tuned model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            self.api.upload_folder(
                folder_path=upload_dir,
                repo_id=repo_id,
                commit_message=commit_message,
                commit_description=f"Fine-tuned model for ETL error analysis\\n\\nTraining details:\\n- Base model: {training_info.get('model_name', 'unknown')}\\n- Training examples: {training_info.get('train_size', 'unknown')}\\n- Validation examples: {training_info.get('validation_size', 'unknown')}",
            )
            
            model_url = f"https://huggingface.co/{repo_id}"
            logger.info(f"Model uploaded successfully: {model_url}")
            
            return model_url
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise
        
        finally:
            # Cleanup upload directory
            if Path(upload_dir).exists():
                shutil.rmtree(upload_dir)

    def create_ollama_modelfile(self, model_path: str, model_name: str) -> str:
        """Create Ollama Modelfile for the uploaded model"""
        
        modelfile_content = f"""FROM {model_name}

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048

TEMPLATE \"\"\"### Instruction:
{{{{ .System }}}}

### Input:
{{{{ .Prompt }}}}

### Response:
\"\"\"

SYSTEM \"\"\"You are an expert data engineer analyzing Airflow ETL task failure logs from a telecom data warehouse.

Your task is to extract the core error message from verbose log data. Focus on:
- The essential technical error that caused the failure
- Remove timestamps, debug info, and verbose details
- Identify the root cause, not just symptoms
- Use technical terminology familiar to data engineers

Common error patterns you should recognize:
- TPT (Teradata Parallel Transporter) errors
- SSH/SFTP connectivity issues
- Database connection and query failures
- Missing data files and dependencies
- Resource constraints and system errors

Respond with just the extracted core error message.\"\"\"
"""
        
        # Save Modelfile
        modelfile_path = Path(model_path) / "Modelfile.hf"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        logger.info(f"Ollama Modelfile created: {modelfile_path}")
        return str(modelfile_path)


def upload_to_huggingface(
    model_path: str,
    repo_id: str,
    training_info: Dict,
    evaluation_results: Optional[Dict] = None,
    token: Optional[str] = None,
    private: bool = False,
) -> str:
    """
    Upload a fine-tuned model to HuggingFace Hub
    
    Args:
        model_path: Path to the fine-tuned model
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        training_info: Training configuration and metrics
        evaluation_results: Model evaluation results
        token: HuggingFace API token (optional, uses HF_TOKEN env var)
        private: Whether to create a private repository
    
    Returns:
        URL of the uploaded model
    """
    
    uploader = HuggingFaceUploader(token=token)
    
    return uploader.upload_model(
        model_path=model_path,
        repo_id=repo_id,
        training_info=training_info,
        evaluation_results=evaluation_results,
        private=private,
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("model_path", help="Path to fine-tuned model")
    parser.add_argument("repo_id", help="HuggingFace repository ID")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--token", help="HuggingFace API token")
    
    args = parser.parse_args()
    
    # Mock training info for standalone usage
    training_info = {
        "model_name": "microsoft/DialoGPT-small",
        "train_size": 153,
        "validation_size": 39,
        "epochs": 3,
        "learning_rate": 2e-4,
        "batch_size": 2,
    }
    
    try:
        url = upload_to_huggingface(
            model_path=args.model_path,
            repo_id=args.repo_id,
            training_info=training_info,
            token=args.token,
            private=args.private,
        )
        print(f"✅ Model uploaded successfully: {url}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        exit(1)