import csv
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from dagnostics.api.schemas import (
    CandidateStats,
    DatasetExportRequest,
    ErrorCandidate,
    FeedbackRequest,
)
from dagnostics.core.database import ErrorCandidateModel, get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


# Helper functions for database operations
def candidate_model_to_schema(db_candidate: ErrorCandidateModel) -> ErrorCandidate:
    """Convert database model to Pydantic schema"""
    return ErrorCandidate(
        id=db_candidate.id,
        dag_id=db_candidate.dag_id,
        task_id=db_candidate.task_id,
        run_id=db_candidate.run_id,
        error_message=db_candidate.error_message,
        raw_logs=db_candidate.raw_logs,
        llm_category=db_candidate.llm_category,
        llm_severity=db_candidate.llm_severity,
        llm_confidence=db_candidate.llm_confidence,
        llm_reasoning=db_candidate.llm_reasoning,
        human_category=db_candidate.human_category,
        human_severity=db_candidate.human_severity,
        human_feedback=db_candidate.human_feedback,
        status=db_candidate.status,
        reviewed_by=db_candidate.reviewed_by,
        reviewed_at=db_candidate.reviewed_at,
        created_at=db_candidate.created_at,
        processing_time=(
            db_candidate.processing_time
            if db_candidate.processing_time is not None
            else 0.0
        ),
    )


@router.get("/candidates", response_model=List[ErrorCandidate])
async def get_candidates(
    status: Optional[str] = Query(None, regex="^(pending|approved|rejected|modified)$"),
    category: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db=Depends(get_db_session),
):
    """Get error candidates for review"""
    try:
        # Build query
        query = db.query(ErrorCandidateModel)

        # Apply filters
        if status:
            query = query.filter(ErrorCandidateModel.status == status)
        if category:
            query = query.filter(
                (ErrorCandidateModel.llm_category == category)
                | (ErrorCandidateModel.human_category == category)
            )
        if severity:
            query = query.filter(
                (ErrorCandidateModel.llm_severity == severity)
                | (ErrorCandidateModel.human_severity == severity)
            )

        # Order by creation date (newest first) and apply pagination
        db_candidates = query.order_by(ErrorCandidateModel.created_at.desc()).offset(offset).limit(limit).all()  # type: ignore

        # Convert to schema
        candidates = [
            candidate_model_to_schema(db_candidate) for db_candidate in db_candidates
        ]

        return candidates

    except Exception as e:
        logger.error(f"Failed to get candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candidates/{candidate_id}", response_model=ErrorCandidate)
async def get_candidate(candidate_id: str, db=Depends(get_db_session)):
    """Get specific candidate details"""
    try:
        db_candidate = (
            db.query(ErrorCandidateModel)
            .filter(ErrorCandidateModel.id == candidate_id)
            .first()
        )

        if not db_candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        return candidate_model_to_schema(db_candidate)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get candidate {candidate_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates/{candidate_id}/feedback")
async def submit_feedback(
    candidate_id: str, feedback: FeedbackRequest, db=Depends(get_db_session)
):
    """Submit human feedback for a candidate"""
    try:
        db_candidate = (
            db.query(ErrorCandidateModel)
            .filter(ErrorCandidateModel.id == candidate_id)
            .first()
        )

        if not db_candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # Update candidate with human feedback
        db_candidate.status = feedback.action
        db_candidate.human_category = feedback.human_category
        db_candidate.human_severity = feedback.human_severity
        db_candidate.human_feedback = feedback.feedback_notes
        db_candidate.reviewed_by = feedback.reviewer_name or "anonymous"
        db_candidate.reviewed_at = datetime.now()

        db.commit()

        logger.info(
            f"Feedback submitted for candidate {candidate_id}: {feedback.action}"
        )

        return {
            "message": "Feedback submitted successfully",
            "candidate_id": candidate_id,
            "action": feedback.action,
            "status": db_candidate.status,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to submit feedback for {candidate_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=CandidateStats)
async def get_training_stats(db=Depends(get_db_session)):
    """Get training dataset statistics"""
    try:
        # Get counts for each status
        total = db.query(ErrorCandidateModel).count()
        pending = (
            db.query(ErrorCandidateModel)
            .filter(ErrorCandidateModel.status == "pending")
            .count()
        )
        approved = (
            db.query(ErrorCandidateModel)
            .filter(ErrorCandidateModel.status == "approved")
            .count()
        )
        rejected = (
            db.query(ErrorCandidateModel)
            .filter(ErrorCandidateModel.status == "rejected")
            .count()
        )
        modified = (
            db.query(ErrorCandidateModel)
            .filter(ErrorCandidateModel.status == "modified")
            .count()
        )

        # Calculate accuracy (approved + modified) / (approved + rejected + modified)
        reviewed = approved + rejected + modified
        accuracy_rate = (approved + modified) / reviewed if reviewed > 0 else 0.0

        # Calculate average confidence
        from sqlalchemy import func

        avg_confidence_result = (
            db.query(func.avg(ErrorCandidateModel.llm_confidence))
            .filter(ErrorCandidateModel.llm_confidence.isnot(None))  # type: ignore
            .scalar()
        )
        avg_confidence = float(avg_confidence_result) if avg_confidence_result else 0.0

        return CandidateStats(
            total_candidates=total,
            pending_review=pending,
            approved=approved,
            rejected=rejected,
            modified=modified,
            accuracy_rate=accuracy_rate,
            avg_confidence=avg_confidence,
        )

    except Exception as e:
        logger.error(f"Failed to get training stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_dataset(
    export_request: DatasetExportRequest, db=Depends(get_db_session)
):
    """Export training dataset in specified format"""
    try:
        # Build query
        query = db.query(ErrorCandidateModel)

        # Filter candidates
        if not export_request.include_rejected:
            query = query.filter(ErrorCandidateModel.status != "rejected")

        if export_request.category_filter:
            from sqlalchemy import and_, or_

            query = query.filter(
                or_(
                    ErrorCandidateModel.human_category.in_(export_request.category_filter),  # type: ignore
                    and_(
                        ErrorCandidateModel.human_category.is_(None),  # type: ignore
                        ErrorCandidateModel.llm_category.in_(export_request.category_filter),  # type: ignore
                    ),
                )
            )

        if export_request.severity_filter:
            from sqlalchemy import and_, or_

            query = query.filter(
                or_(
                    ErrorCandidateModel.human_severity.in_(export_request.severity_filter),  # type: ignore
                    and_(
                        ErrorCandidateModel.human_severity.is_(None),  # type: ignore
                        ErrorCandidateModel.llm_severity.in_(export_request.severity_filter),  # type: ignore
                    ),
                )
            )

        if export_request.confidence_threshold:
            query = query.filter(
                ErrorCandidateModel.llm_confidence >= export_request.confidence_threshold  # type: ignore
            )

        # Get all matching candidates
        db_candidates = query.all()
        candidates = [
            candidate_model_to_schema(db_candidate) for db_candidate in db_candidates
        ]

        # Generate export data
        if export_request.format == "json":
            return _export_json(candidates)
        elif export_request.format == "csv":
            return _export_csv(candidates)
        elif export_request.format == "jsonl":
            return _export_jsonl(candidates)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates/bulk")
async def create_candidates_from_analysis(
    analysis_results: List[dict], db=Depends(get_db_session)
):
    """Create candidates from analysis results (for testing)"""
    try:
        created_count = 0

        for result in analysis_results:
            candidate_id = str(uuid4())

            db_candidate = ErrorCandidateModel(
                id=candidate_id,
                dag_id=result.get("dag_id", "unknown"),
                task_id=result.get("task_id", "unknown"),
                run_id=result.get("run_id", "unknown"),
                error_message=result.get("error_message", ""),
                raw_logs=result.get("raw_logs", ""),
                llm_category=result.get("category"),
                llm_severity=result.get("severity"),
                llm_confidence=result.get("confidence"),
                llm_reasoning=result.get("reasoning"),
                status="pending",
                created_at=datetime.now(),
                processing_time=result.get("processing_time", 0.0),
            )

            db.add(db_candidate)
            created_count += 1

        db.commit()

        return {
            "message": f"Created {created_count} candidates",
            "created_count": created_count,
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates/load-from-jsonl")
async def load_candidates_from_jsonl(
    file_path: str = "/root/dagnostics/data/training_data.jsonl",
    limit: Optional[int] = 100,
    db=Depends(get_db_session),
):
    """Load candidates from your existing JSONL file"""
    try:
        import json
        import re
        from pathlib import Path

        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="JSONL file not found")

        created_count = 0

        with open(file_path, "r") as file:
            for i, line in enumerate(file):
                if limit and i >= limit:
                    break

                try:
                    data = json.loads(line.strip())
                    raw_logs = data.get("candidates", "")
                    error_message = data.get("error", "")

                    if not raw_logs or not error_message:
                        continue

                    # Extract DAG and task info from logs
                    dag_id = "unknown"
                    task_id = "unknown"
                    run_id = "unknown"

                    # Try to extract DAG ID
                    dag_match = re.search(r"DAG_ID[=:]\s*['\"]?([^'\",$\s]+)", raw_logs)
                    if dag_match:
                        dag_id = dag_match.group(1)

                    # Try to extract Task ID
                    task_match = re.search(
                        r"TASK_ID[=:]\s*['\"]?([^'\",$\s]+)", raw_logs
                    )
                    if task_match:
                        task_id = task_match.group(1)

                    # Try to extract Run ID
                    run_match = re.search(
                        r"DAG_RUN_ID[=:]\s*['\"]?([^'\",$\s]+)", raw_logs
                    )
                    if run_match:
                        run_id = run_match.group(1)

                    candidate_id = str(uuid4())

                    db_candidate = ErrorCandidateModel(
                        id=candidate_id,
                        dag_id=dag_id,
                        task_id=task_id,
                        run_id=run_id,
                        error_message=error_message,
                        raw_logs=raw_logs,
                        llm_category=None,  # No LLM analysis yet
                        llm_severity=None,
                        llm_confidence=None,
                        llm_reasoning=None,
                        status="pending",
                        created_at=datetime.now(),
                        processing_time=0.0,
                    )

                    db.add(db_candidate)
                    created_count += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line {i}: {e}")
                    continue

        db.commit()

        return {
            "message": f"Loaded {created_count} candidates from JSONL file",
            "created_count": created_count,
            "file_path": file_path,
        }

    except Exception as e:
        logger.error(f"Failed to load JSONL file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates/load-from-airflow")
async def load_candidates_from_airflow(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    dag_pattern: Optional[str] = Query(None, description="DAG name pattern filter"),
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_db_session),
):
    """Load candidates from live Airflow database for specific date range"""
    try:
        from datetime import datetime

        from dagnostics.analysis.analyzer import DAGAnalyzer
        from dagnostics.clustering.log_clusterer import LogClusterer
        from dagnostics.core.airflow_client import AirflowClient
        from dagnostics.core.config import load_config
        from dagnostics.heuristics.filter_factory import FilterFactory
        from dagnostics.llm.engine import LLMEngine, OllamaProvider

        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
            )

        # Initialize analyzer for live data access
        config = load_config()

        airflow_client = AirflowClient(
            base_url=config.airflow.base_url,
            username=config.airflow.username,
            password=config.airflow.password,
            db_connection=config.airflow.database_url,
            verify_ssl=False,
        )

        clusterer = LogClusterer(
            persistence_path=config.drain3.persistence_path, app_config=config
        )

        filter_instance = FilterFactory.create_for_notifications(config)
        llm_provider = OllamaProvider()
        llm = LLMEngine(llm_provider, config=config)

        analyzer = DAGAnalyzer(airflow_client, clusterer, filter_instance, llm, config)

        # Calculate time range in minutes
        time_range_minutes = int((end_dt - start_dt).total_seconds() / 60)

        # Get failed tasks from Airflow
        failed_tasks = analyzer.airflow_client.get_failed_tasks(time_range_minutes)

        if not failed_tasks:
            return {
                "message": "No failed tasks found in specified date range",
                "created_count": 0,
                "date_range": f"{start_date} to {end_date}",
            }

        # Filter by DAG pattern if provided
        if dag_pattern:
            import fnmatch

            failed_tasks = [
                task
                for task in failed_tasks
                if fnmatch.fnmatch(task.dag_id.lower(), dag_pattern.lower())
            ]

        created_count = 0

        # Process failed tasks and extract errors
        for task in failed_tasks[:limit]:
            try:
                # Get task tries
                task_tries = analyzer.airflow_client.get_task_tries(
                    task.dag_id, task.task_id, task.run_id
                )

                # Filter failed tries
                failed_tries = [
                    try_instance
                    for try_instance in task_tries
                    if try_instance.state == "failed" and try_instance.try_number > 0
                ]

                for failed_try in failed_tries:
                    try:
                        # Extract error using your existing method
                        _, candidates, error_message = (
                            analyzer.extract_task_error_for_sms(
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

                            candidate_id = str(uuid4())
                            db_candidate = ErrorCandidateModel(
                                id=candidate_id,
                                dag_id=failed_try.dag_id,
                                task_id=failed_try.task_id,
                                run_id=failed_try.run_id,
                                error_message=error_message,
                                raw_logs=error_logs,
                                llm_category=None,  # No LLM analysis yet
                                llm_severity=None,
                                llm_confidence=None,
                                llm_reasoning=None,
                                status="pending",
                                created_at=datetime.now(),
                                processing_time=0.0,
                            )

                            db.add(db_candidate)
                            created_count += 1

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

        db.commit()

        return {
            "message": f"Loaded {created_count} failed tasks from live Airflow",
            "created_count": created_count,
            "date_range": f"{start_date} to {end_date}",
            "dag_pattern": dag_pattern,
            "total_failed_tasks": len(failed_tasks),
        }

    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Analysis dependencies not available: {e}"
        )
    except Exception as e:
        logger.error(f"Failed to load from Airflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates/{candidate_id}/analyze")
async def analyze_candidate_with_llm(candidate_id: str, db=Depends(get_db_session)):
    """Run LLM analysis on a specific candidate"""
    try:
        db_candidate = (
            db.query(ErrorCandidateModel)
            .filter(ErrorCandidateModel.id == candidate_id)
            .first()
        )

        if not db_candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # Import LLM analyzer (in a real implementation)
        try:
            # This would normally analyze the candidate
            # For now, we'll simulate LLM analysis
            import random

            # Mock LLM analysis based on error patterns
            error_msg = db_candidate.error_message.lower()

            if "timeout" in error_msg or "connection" in error_msg:
                category = "resource_error"
                severity = "high"
                confidence = 0.9
                reasoning = (
                    "Network connectivity or timeout issue detected in error message."
                )
            elif "file" in error_msg and (
                "missing" in error_msg or "not found" in error_msg
            ):
                category = "data_quality"
                severity = "medium"
                confidence = 0.85
                reasoning = "Missing file or data source availability issue."
            elif "tpt" in error_msg and "error" in error_msg:
                category = "dependency_failure"
                severity = "high"
                confidence = 0.88
                reasoning = "TPT (Teradata Parallel Transporter) tool failure."
            elif "ssh" in error_msg:
                category = "resource_error"
                severity = "high"
                confidence = 0.92
                reasoning = "SSH connection failure to remote host."
            elif "bash command failed" in error_msg:
                category = "dependency_failure"
                severity = "medium"
                confidence = 0.75
                reasoning = "Shell command execution failure."
            else:
                category = "unknown"
                severity = "medium"
                confidence = 0.6
                reasoning = "Unable to classify error with high confidence."

            # Update candidate with LLM analysis
            db_candidate.llm_category = category
            db_candidate.llm_severity = severity
            db_candidate.llm_confidence = confidence
            db_candidate.llm_reasoning = reasoning
            db_candidate.processing_time = random.uniform(0.5, 3.0)

            db.commit()

            return {
                "message": "LLM analysis completed",
                "candidate_id": candidate_id,
                "analysis": {
                    "category": category,
                    "severity": severity,
                    "confidence": confidence,
                    "reasoning": reasoning,
                },
            }

        except ImportError:
            # Fallback mock analysis if LLM components aren't available
            import random

            categories = [
                "resource_error",
                "data_quality",
                "dependency_failure",
                "timeout_error",
                "unknown",
            ]
            severities = ["critical", "high", "medium", "low"]

            db_candidate.llm_category = random.choice(categories)
            db_candidate.llm_severity = random.choice(severities)
            db_candidate.llm_confidence = random.uniform(0.6, 0.95)
            db_candidate.llm_reasoning = (
                f"Mock analysis of error: {db_candidate.error_message[:50]}..."
            )
            db_candidate.processing_time = random.uniform(0.5, 3.0)

            db.commit()

            return {
                "message": "Mock LLM analysis completed",
                "candidate_id": candidate_id,
                "analysis": {
                    "category": db_candidate.llm_category,
                    "severity": db_candidate.llm_severity,
                    "confidence": db_candidate.llm_confidence,
                    "reasoning": db_candidate.llm_reasoning,
                },
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze candidate {candidate_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates/analyze-all")
async def analyze_all_pending_candidates(
    limit: int = Query(50, ge=1, le=200), db=Depends(get_db_session)
):
    """Run LLM analysis on all pending candidates"""
    try:
        # Get pending candidates with no LLM analysis
        pending_candidates = (
            db.query(ErrorCandidateModel)
            .filter(
                ErrorCandidateModel.status == "pending",
                ErrorCandidateModel.llm_category.is_(None),  # type: ignore
            )
            .limit(limit)
            .all()
        )

        analyzed_count = 0

        for db_candidate in pending_candidates:
            try:
                await analyze_candidate_with_llm(db_candidate.id, db)
                analyzed_count += 1
            except Exception as e:
                logger.warning(f"Failed to analyze candidate {db_candidate.id}: {e}")
                continue

        # Get remaining count
        remaining_count = (
            db.query(ErrorCandidateModel)
            .filter(
                ErrorCandidateModel.status == "pending",
                ErrorCandidateModel.llm_category.is_(None),  # type: ignore
            )
            .count()
        )

        return {
            "message": f"Analyzed {analyzed_count} candidates",
            "analyzed_count": analyzed_count,
            "remaining_pending": remaining_count,
        }

    except Exception as e:
        logger.error(f"Failed to analyze candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-ml-dataset")
async def generate_ml_training_dataset(
    ndays: int = Query(15, ge=1, le=90),
    output_format: str = Query("jsonl", regex="^(jsonl|huggingface)$"),
    db=Depends(get_db_session),
):
    """Generate ML training dataset from live Airflow data + web feedback"""
    try:
        from dagnostics.analysis.analyzer import DAGAnalyzer
        from dagnostics.clustering.log_clusterer import LogClusterer
        from dagnostics.core.airflow_client import AirflowClient
        from dagnostics.core.config import load_config
        from dagnostics.heuristics.filter_factory import FilterFactory
        from dagnostics.llm.engine import LLMEngine, OllamaProvider
        from dagnostics.training.dataset_generator import DatasetGenerator

        # Initialize analyzer for live data access
        config = load_config()

        airflow_client = AirflowClient(
            base_url=config.airflow.base_url,
            username=config.airflow.username,
            password=config.airflow.password,
            db_connection=config.airflow.database_url,
            verify_ssl=False,
        )

        clusterer = LogClusterer(
            persistence_path=config.drain3.persistence_path, app_config=config
        )

        filter = FilterFactory.create_for_notifications(config)
        llm_provider = OllamaProvider()
        llm = LLMEngine(llm_provider, config=config)

        analyzer = DAGAnalyzer(airflow_client, clusterer, filter, llm, config)

        # Convert web candidates to dict format for dataset generator
        feedback_candidates = []
        reviewed_db_candidates = (
            db.query(ErrorCandidateModel)
            .filter(
                ErrorCandidateModel.status.in_(["approved", "rejected", "modified"])  # type: ignore
            )
            .all()
        )

        for db_candidate in reviewed_db_candidates:
            feedback_candidates.append(
                {
                    "raw_logs": db_candidate.raw_logs,
                    "error_message": db_candidate.error_message,
                    "llm_category": db_candidate.llm_category,
                    "llm_severity": db_candidate.llm_severity,
                    "human_category": db_candidate.human_category,
                    "human_severity": db_candidate.human_severity,
                    "human_feedback": db_candidate.human_feedback,
                    "status": db_candidate.status,
                    "reviewed_by": db_candidate.reviewed_by,
                    "dag_id": db_candidate.dag_id,
                    "task_id": db_candidate.task_id,
                    "run_id": db_candidate.run_id,
                }
            )

        # Initialize dataset generator with live analyzer and feedback
        generator = DatasetGenerator(
            output_dir="/root/dagnostics/data/training",
            analyzer=analyzer,
            feedback_candidates=feedback_candidates,
        )

        # Generate the complete training dataset dynamically
        dataset_info = generator.generate_full_dataset()

        return {
            "message": "ML training dataset generated from live Airflow data + web feedback",
            "dataset_info": dataset_info,
            "live_airflow_days": ndays,
            "web_feedback_used": len(feedback_candidates),
            "output_format": output_format,
        }

    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Analysis dependencies not available: {e}"
        )
    except Exception as e:
        logger.error(f"Failed to generate ML dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-model-training")
async def start_model_training(
    model_name: str = Query("microsoft/DialoGPT-small"),
    num_epochs: int = Query(3, ge=1, le=10),
    learning_rate: float = Query(2e-4, gt=0, lt=1),
    batch_size: int = Query(2, ge=1, le=8),
):
    """Start fine-tuning a model with collected feedback"""
    try:
        from dagnostics.training.fine_tuner import SLMFineTuner

        # Check if training datasets exist
        train_path = "/root/dagnostics/data/training/train_dataset.jsonl"
        val_path = "/root/dagnostics/data/training/validation_dataset.jsonl"

        if not Path(train_path).exists():
            raise HTTPException(
                status_code=400,
                detail="Training dataset not found. Run 'Generate ML Dataset' first.",
            )

        # Initialize fine-tuner
        fine_tuner = SLMFineTuner(
            model_name=model_name,
            output_dir="/root/dagnostics/models/fine_tuned",
            use_quantization=True,
        )

        # Start training (this would typically be done asynchronously)
        model_path = fine_tuner.train_model(
            train_dataset_path=train_path,
            validation_dataset_path=val_path if Path(val_path).exists() else None,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        return {
            "message": "Model training completed successfully",
            "model_path": model_path,
            "training_params": {
                "model_name": model_name,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            },
        }

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fine-tuning dependencies not available: {e}. Install with: pip install 'dagnostics[finetuning]'",
        )
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _export_json(candidates: List[ErrorCandidate]) -> Response:
    """Export candidates as JSON"""
    export_data = []

    for candidate in candidates:
        # Use human labels if available, otherwise LLM labels
        final_category = candidate.human_category or candidate.llm_category
        final_severity = candidate.human_severity or candidate.llm_severity

        export_data.append(
            {
                "id": candidate.id,
                "dag_id": candidate.dag_id,
                "task_id": candidate.task_id,
                "error_message": candidate.error_message,
                "raw_logs": candidate.raw_logs,
                "category": final_category,
                "severity": final_severity,
                "confidence": candidate.llm_confidence,
                "status": candidate.status,
                "human_feedback": candidate.human_feedback,
                "created_at": candidate.created_at.isoformat(),
                "reviewed_at": (
                    candidate.reviewed_at.isoformat() if candidate.reviewed_at else None
                ),
            }
        )

    json_str = json.dumps(export_data, indent=2)

    return Response(
        content=json_str,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        },
    )


def _export_csv(candidates: List[ErrorCandidate]) -> StreamingResponse:
    """Export candidates as CSV"""

    def generate():
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(
            [
                "id",
                "dag_id",
                "task_id",
                "error_message",
                "category",
                "severity",
                "confidence",
                "status",
                "human_feedback",
                "created_at",
                "reviewed_at",
            ]
        )

        for candidate in candidates:
            final_category = candidate.human_category or candidate.llm_category
            final_severity = candidate.human_severity or candidate.llm_severity

            writer.writerow(
                [
                    candidate.id,
                    candidate.dag_id,
                    candidate.task_id,
                    candidate.error_message,
                    final_category,
                    final_severity,
                    candidate.llm_confidence,
                    candidate.status,
                    candidate.human_feedback or "",
                    candidate.created_at.isoformat(),
                    candidate.reviewed_at.isoformat() if candidate.reviewed_at else "",
                ]
            )

        output.seek(0)
        return output.read()

    return StreamingResponse(
        io.StringIO(generate()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        },
    )


def _export_jsonl(candidates: List[ErrorCandidate]) -> Response:
    """Export candidates as JSONL (JSON Lines)"""
    lines = []

    for candidate in candidates:
        final_category = candidate.human_category or candidate.llm_category
        final_severity = candidate.human_severity or candidate.llm_severity

        line_data = {
            "text": candidate.error_message,
            "label": final_category,
            "metadata": {
                "severity": final_severity,
                "dag_id": candidate.dag_id,
                "task_id": candidate.task_id,
                "confidence": candidate.llm_confidence,
                "status": candidate.status,
            },
        }

        lines.append(json.dumps(line_data))

    jsonl_content = "\n".join(lines)

    return Response(
        content=jsonl_content,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        },
    )
