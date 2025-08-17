"""
Remote Training Server

FastAPI server for handling remote fine-tuning jobs.
Runs on GPU machines to offload training computation.
"""

import asyncio
import json
import logging
import shutil
import tarfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import training modules with dependency check
try:
    from dagnostics.training.fine_tuner import train_from_prepared_data

    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DAGnostics Training Server",
    description="Remote training server for fine-tuning models",
    version="0.5.0",
)

# In-memory job storage (use Redis/DB for production)
jobs: Dict[str, Dict] = {}
job_lock = threading.Lock()

# Server configuration
UPLOAD_DIR = Path("server_data/uploads")
MODELS_DIR = Path("server_data/models")
DATASETS_DIR = Path("server_data/datasets")

# Create directories
for dir_path in [UPLOAD_DIR, MODELS_DIR, DATASETS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class TrainingJobRequest(BaseModel):
    """Training job request"""

    job_id: str
    model_name: str
    train_dataset_path: str
    validation_dataset_path: Optional[str] = None
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    use_quantization: bool = True
    created_at: str


class TrainingJobStatus(BaseModel):
    """Training job status"""

    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def update_job_status(job_id: str, **updates):
    """Thread-safe job status update"""
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def run_training_job(job_request: TrainingJobRequest):
    """Run training job in background thread"""
    job_id = job_request.job_id

    try:
        # Update status to running
        update_job_status(
            job_id, status="running", started_at=datetime.now().isoformat()
        )

        logger.info(f"Starting training job {job_id}")

        # Check if training dependencies are available
        if not HAS_TRAINING_DEPS:
            raise ImportError("Training dependencies not installed")

        # Prepare local dataset paths
        train_dataset = DATASETS_DIR / job_id / "train.jsonl"
        val_dataset = DATASETS_DIR / job_id / "val.jsonl"

        if not train_dataset.exists():
            raise FileNotFoundError(f"Training dataset not found: {train_dataset}")

        # Update progress
        update_job_status(job_id, progress=0.1)

        # Start training
        model_path = train_from_prepared_data(
            model_name=job_request.model_name,
            train_dataset_path=str(train_dataset),
            validation_dataset_path=str(val_dataset) if val_dataset.exists() else None,
            epochs=job_request.epochs,
            learning_rate=job_request.learning_rate,
            batch_size=job_request.batch_size,
            model_output_name=f"remote-{job_id}",
            use_quantization=job_request.use_quantization,
            export_for_ollama=False,  # Skip Ollama export on server
        )

        # Move model to server models directory
        server_model_path = MODELS_DIR / job_id
        if Path(model_path).exists():
            shutil.move(model_path, server_model_path)

        # Create model archive for download
        archive_path = MODELS_DIR / f"{job_id}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(server_model_path, arcname=job_id)

        # Update status to completed
        update_job_status(
            job_id,
            status="completed",
            progress=1.0,
            model_path=str(server_model_path),
            completed_at=datetime.now().isoformat(),
        )

        logger.info(f"Training job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        update_job_status(
            job_id,
            status="failed",
            error_message=str(e),
            completed_at=datetime.now().isoformat(),
        )


@app.get("/")
async def root():
    """Server status"""
    return {
        "service": "DAGnostics Training Server",
        "version": "0.5.0",
        "status": "running",
        "has_training_deps": HAS_TRAINING_DEPS,
        "active_jobs": len([j for j in jobs.values() if j["status"] == "running"]),
        "total_jobs": len(jobs),
    }


@app.post("/training/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload training dataset"""
    try:
        # Extract job_id and filename from file.filename
        # Expected format: "datasets/{job_id}/train.jsonl" or similar
        file_path = Path(file.filename)
        job_id = file_path.parts[1] if len(file_path.parts) > 1 else "unknown"
        filename = file_path.name

        # Create job directory
        job_upload_dir = DATASETS_DIR / job_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        upload_path = job_upload_dir / filename
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"File uploaded: {upload_path}")
        return {"message": "File uploaded", "path": str(upload_path)}

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/submit")
async def submit_training_job(job_request: TrainingJobRequest):
    """Submit training job"""
    try:
        # Initialize job status
        with job_lock:
            jobs[job_request.job_id] = {
                "job_id": job_request.job_id,
                "status": "pending",
                "progress": 0.0,
                "model_name": job_request.model_name,
                "epochs": job_request.epochs,
                "created_at": job_request.created_at,
                "current_epoch": None,
                "total_epochs": job_request.epochs,
                "loss": None,
                "model_path": None,
                "error_message": None,
                "started_at": None,
                "completed_at": None,
            }

        # Start training in background thread
        training_thread = threading.Thread(
            target=run_training_job, args=(job_request,), daemon=True
        )
        training_thread.start()

        logger.info(f"Training job {job_request.job_id} submitted")
        return {"message": "Training job submitted", "job_id": job_request.job_id}

    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/status/{job_id}")
async def get_job_status(job_id: str):
    """Get training job status"""
    with job_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        return TrainingJobStatus(**jobs[job_id])


@app.get("/training/jobs")
async def list_jobs():
    """List all training jobs"""
    with job_lock:
        job_list = [TrainingJobStatus(**job) for job in jobs.values()]
        return {"jobs": job_list}


@app.get("/training/download/{job_id}")
async def download_model(job_id: str):
    """Download trained model"""
    with job_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")

    # Return model archive
    archive_path = MODELS_DIR / f"{job_id}.tar.gz"
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="Model archive not found")

    return FileResponse(
        path=archive_path,
        filename=f"model_{job_id}.tar.gz",
        media_type="application/gzip",
    )


@app.post("/training/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel training job"""
    with job_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        if job["status"] in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="Job already finished")

        # Update status (actual cancellation would need more complex logic)
        jobs[job_id]["status"] = "cancelled"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

    return {"message": "Job cancelled", "job_id": job_id}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "has_training_deps": HAS_TRAINING_DEPS,
    }


def main():
    """Run the training server"""
    import argparse

    parser = argparse.ArgumentParser(description="DAGnostics Training Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    if not HAS_TRAINING_DEPS:
        print("❌ Training dependencies not installed!")
        print("Install with: pip install 'dagnostics[finetuning]'")
        exit(1)

    print(f"🚀 Starting DAGnostics Training Server")
    print(f"📍 Server: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🏃 Workers: {args.workers}")
    print("")

    uvicorn.run(
        "dagnostics.training.training_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
