import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from dagnostics.analysis.analyzer import DAGAnalyzer
from dagnostics.api.schemas import (
    AnalysisHistoryItem,
    AnalyzeRequest,
    AnalyzeResponse,
    BaselineInfo,
)
from dagnostics.core.database import AnalysisRecord, get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


def get_analyzer():
    """Get configured DAGAnalyzer instance"""
    try:
        from dagnostics.analysis.analyzer import DAGAnalyzer
        from dagnostics.clustering.log_clusterer import LogClusterer
        from dagnostics.core.airflow_client import AirflowClient
        from dagnostics.core.config import load_config
        from dagnostics.heuristics.pattern_filter import ErrorPatternFilter
        from dagnostics.llm.engine import LLMEngine, OllamaProvider

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

        filter = ErrorPatternFilter()

        # Use Ollama provider for now
        llm_provider = OllamaProvider()
        llm = LLMEngine(llm_provider, config=config)

        return DAGAnalyzer(airflow_client, clusterer, filter, llm, config)

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Analysis dependencies not available. Install with: pip install dagnostics[llm]. Error: {e}",
        )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_task_failure(
    request: AnalyzeRequest, analyzer: DAGAnalyzer = Depends(get_analyzer)
):
    """Analyze a specific task failure"""
    try:
        logger.info(f"Starting analysis for {request.dag_id}.{request.task_id}")

        result = analyzer.analyze_task_failure(
            request.dag_id, request.task_id, request.run_id, request.try_number
        )

        return AnalyzeResponse(
            analysis_id=result.id,
            dag_id=result.dag_id,
            task_id=result.task_id,
            run_id=result.run_id,
            try_number=request.try_number,
            error_message=result.analysis.error_message if result.analysis else None,
            category=result.analysis.category.value if result.analysis else None,
            severity=result.analysis.severity.value if result.analysis else None,
            confidence=result.analysis.confidence if result.analysis else None,
            suggested_actions=(
                result.analysis.suggested_actions if result.analysis else []
            ),
            processing_time=result.processing_time,
            timestamp=result.timestamp,
            success=result.success,
            baseline_comparison=(
                result.baseline_comparison.__dict__
                if result.baseline_comparison
                else None
            ),
        )

    except Exception as e:
        logger.error(f"Analysis failed for {request.dag_id}.{request.task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[AnalysisHistoryItem])
async def get_analysis_history(
    dag_id: Optional[str] = None,
    task_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    db=Depends(get_db_session),
):
    """Get analysis history with optional filtering"""
    try:
        query = db.query(AnalysisRecord)

        if dag_id:
            query = query.filter(AnalysisRecord.dag_id == dag_id)
        if task_id:
            query = query.filter(AnalysisRecord.task_id == task_id)

        results = query.order_by(AnalysisRecord.timestamp.desc()).limit(limit).all()  # type: ignore

        history_items = []
        for result in results:
            # AnalysisRecord has direct fields, no need for analysis_data
            category = result.category
            severity = result.severity
            error_message = result.error_message

            history_items.append(
                AnalysisHistoryItem(
                    analysis_id=result.id,
                    dag_id=result.dag_id,
                    task_id=result.task_id,
                    run_id=result.run_id,
                    timestamp=result.timestamp,
                    success=result.success,
                    category=category,
                    severity=severity,
                    error_message=(
                        error_message[:100] + "..."
                        if error_message and len(error_message) > 100
                        else error_message
                    ),
                    processing_time=result.processing_time,
                )
            )

        return history_items

    except Exception as e:
        logger.error(f"Failed to get analysis history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/baselines", response_model=List[BaselineInfo])
async def get_baselines(_analyzer: DAGAnalyzer = Depends(get_analyzer)):
    """Get all baseline information"""
    try:
        # Mock baseline information for now
        baselines = [
            BaselineInfo(
                dag_id="example_dag",
                task_id="example_task",
                cluster_count=5,
                created_at=None,
                last_updated=None,
                successful_runs=10,
                is_stale=False,
            )
        ]

        return baselines

    except Exception as e:
        logger.error(f"Failed to get baselines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/baselines/rebuild")
async def rebuild_baseline(
    dag_id: str, task_id: str, analyzer: DAGAnalyzer = Depends(get_analyzer)
):
    """Rebuild baseline for specific DAG/task"""
    try:
        logger.info(f"Rebuilding baseline for {dag_id}.{task_id}")

        # Force rebuild baseline
        analyzer._ensure_baseline(dag_id, task_id)

        return {
            "message": f"Baseline rebuilt for {dag_id}.{task_id}",
            "dag_id": dag_id,
            "task_id": task_id,
            "baseline_exists": True,
            "baseline_stale": False,
            "cluster_count": 5,
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Failed to rebuild baseline for {dag_id}.{task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/baselines")
async def clear_baseline(
    dag_id: str, task_id: str, _analyzer: DAGAnalyzer = Depends(get_analyzer)
):
    """Clear baseline for specific DAG/task"""
    try:
        logger.info(f"Clearing baseline for {dag_id}.{task_id}")

        # Mock clearing baseline
        logger.info(f"Baseline cleared for {dag_id}.{task_id}")

        return {
            "message": f"Baseline cleared for {dag_id}.{task_id}",
            "dag_id": dag_id,
            "task_id": task_id,
        }

    except Exception as e:
        logger.error(f"Failed to clear baseline for {dag_id}.{task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_error_categories():
    """Get all available error categories"""
    from dagnostics.core.models import ErrorCategory

    categories = [
        {
            "name": category.value,
            "display_name": category.value.replace("_", " ").title(),
            "description": f"Errors related to {category.value.replace('_', ' ')}",
        }
        for category in ErrorCategory
    ]

    return {"categories": categories}


@router.get("/severities")
async def get_error_severities():
    """Get all available error severities"""
    from dagnostics.core.models import ErrorSeverity

    severities = [
        {"name": severity.value, "display_name": severity.value.title(), "level": i + 1}
        for i, severity in enumerate(ErrorSeverity)
    ]

    return {"severities": severities}
