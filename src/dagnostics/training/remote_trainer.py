"""
Remote Training Client

Submit training jobs to a separate training server/machine without
requiring ML dependencies on the main application.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TrainingJob(BaseModel):
    """Training job specification"""

    job_id: str
    model_name: str
    train_dataset_path: str
    validation_dataset_path: Optional[str] = None
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    use_quantization: bool = True
    created_at: str
    status: str = "pending"  # pending, running, completed, failed


class TrainingJobStatus(BaseModel):
    """Training job status response"""

    job_id: str
    status: str
    progress: float  # 0.0 - 1.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class RemoteTrainer:
    """Client for submitting training jobs to remote training server"""

    def __init__(self, training_server_url: str = "http://localhost:8001"):
        self.server_url = training_server_url.rstrip("/")
        self.session = requests.Session()

    def submit_training_job(
        self,
        model_name: str,
        train_dataset_path: str,
        validation_dataset_path: Optional[str] = None,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 2,
    ) -> str:
        """Submit a training job to remote server"""

        # Generate job ID
        job_id = f"train_{int(datetime.now().timestamp())}"

        # Prepare job data
        job_data = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            train_dataset_path=train_dataset_path,
            validation_dataset_path=validation_dataset_path,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            created_at=datetime.now().isoformat(),
        )

        try:
            # Upload datasets first
            if Path(train_dataset_path).exists():
                self._upload_file(train_dataset_path, f"datasets/{job_id}/train.jsonl")

            if validation_dataset_path and Path(validation_dataset_path).exists():
                self._upload_file(
                    validation_dataset_path, f"datasets/{job_id}/val.jsonl"
                )

            # Submit training job
            response = self.session.post(
                f"{self.server_url}/training/submit",
                json=job_data.model_dump(),
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Training job submitted: {job_id} -> {result}")
            return job_id

        except requests.RequestException as e:
            logger.error(f"Failed to submit training job: {e}")
            raise

    def get_job_status(self, job_id: str) -> TrainingJobStatus:
        """Get status of a training job"""

        try:
            response = self.session.get(
                f"{self.server_url}/training/status/{job_id}", timeout=10
            )
            response.raise_for_status()

            status_data = response.json()
            return TrainingJobStatus(**status_data)

        except requests.RequestException as e:
            logger.error(f"Failed to get job status: {e}")
            raise

    def wait_for_completion(
        self, job_id: str, poll_interval: int = 30, timeout: int = 3600
    ) -> TrainingJobStatus:
        """Wait for training job to complete"""

        logger.info(f"Waiting for training job {job_id} to complete...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            logger.info(
                f"Job {job_id}: {status.status} (progress: {status.progress:.1%})"
            )

            if status.status == "completed":
                logger.info(f"Training completed! Model: {status.model_path}")
                return status
            elif status.status == "failed":
                logger.error(f"Training failed: {status.error_message}")
                return status

            time.sleep(poll_interval)

        # Timeout
        logger.error(f"Training job {job_id} timed out after {timeout}s")
        raise TimeoutError(f"Training job {job_id} timed out")

    def download_model(self, job_id: str, output_dir: str = "models/fine_tuned") -> str:
        """Download trained model from remote server"""

        try:
            # Get job status to find model path
            status = self.get_job_status(job_id)

            if status.status != "completed" or not status.model_path:
                raise ValueError(f"Job {job_id} not completed or no model available")

            # Download model files
            output_path = Path(output_dir) / f"remote_model_{job_id}"
            output_path.mkdir(parents=True, exist_ok=True)

            response = self.session.get(
                f"{self.server_url}/training/download/{job_id}",
                stream=True,
                timeout=300,
            )
            response.raise_for_status()

            # Save model archive
            model_archive = output_path / "model.tar.gz"
            with open(model_archive, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract model files
            import tarfile

            with tarfile.open(model_archive, "r:gz") as tar:
                # Validate and extract safely
                def is_safe_path(path):
                    return not (path.startswith("/") or ".." in path)

                safe_members = [m for m in tar.getmembers() if is_safe_path(m.name)]
                tar.extractall(output_path, members=safe_members)  # nosec B202

            # Remove archive
            model_archive.unlink()

            logger.info(f"Model downloaded to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def _upload_file(self, local_path: str, remote_path: str):
        """Upload file to training server"""

        try:
            with open(local_path, "rb") as f:
                files = {"file": (remote_path, f, "application/octet-stream")}
                response = self.session.post(
                    f"{self.server_url}/training/upload", files=files, timeout=120
                )
                response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    def list_jobs(self) -> list[TrainingJobStatus]:
        """List all training jobs"""

        try:
            response = self.session.get(f"{self.server_url}/training/jobs", timeout=10)
            response.raise_for_status()

            jobs_data = response.json()
            return [TrainingJobStatus(**job) for job in jobs_data["jobs"]]

        except requests.RequestException as e:
            logger.error(f"Failed to list jobs: {e}")
            raise

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job"""

        try:
            response = self.session.post(
                f"{self.server_url}/training/cancel/{job_id}", timeout=10
            )
            response.raise_for_status()

            logger.info(f"Training job {job_id} cancelled")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False


# CLI commands for remote training
def remote_train_command(
    model_name: str = "microsoft/DialoGPT-small",
    train_dataset: str = "data/training/train_dataset.jsonl",
    val_dataset: Optional[str] = "data/training/validation_dataset.jsonl",
    epochs: int = 3,
    server_url: str = "http://localhost:8001",
    wait: bool = True,
) -> str:
    """Submit training job to remote server"""

    # Initialize remote trainer
    trainer = RemoteTrainer(server_url)

    # Submit job
    job_id = trainer.submit_training_job(
        model_name=model_name,
        train_dataset_path=train_dataset,
        validation_dataset_path=val_dataset,
        epochs=epochs,
    )

    print(f"Training job submitted: {job_id}")
    print(f"Track progress: dagnostics training remote-status {job_id}")

    if wait:
        # Wait for completion
        status = trainer.wait_for_completion(job_id)

        if status.status == "completed":
            # Download model
            model_path = trainer.download_model(job_id)
            print(f"Model downloaded to: {model_path}")
            return model_path
        else:
            print(f"Training failed: {status.error_message}")

    return job_id


def remote_status_command(job_id: str, server_url: str = "http://localhost:8001"):
    """Check status of remote training job"""

    trainer = RemoteTrainer(server_url)
    status = trainer.get_job_status(job_id)

    print(f"Job ID: {status.job_id}")
    print(f"Status: {status.status}")
    print(f"Progress: {status.progress:.1%}")

    if status.current_epoch:
        print(f"Epoch: {status.current_epoch}/{status.total_epochs}")
    if status.loss:
        print(f"Loss: {status.loss:.4f}")
    if status.error_message:
        print(f"Error: {status.error_message}")


def remote_download_command(
    job_id: str,
    output_dir: str = "models/fine_tuned",
    server_url: str = "http://localhost:8001",
):
    """Download trained model from remote server"""

    trainer = RemoteTrainer(server_url)
    model_path = trainer.download_model(job_id, output_dir)
    print(f"Model downloaded to: {model_path}")
    return model_path
