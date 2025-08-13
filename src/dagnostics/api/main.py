import logging
from datetime import datetime
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dagnostics import __version__
from dagnostics.analysis.analyzer import DAGAnalyzer
from dagnostics.core.models import (
    AnthropicLLMConfig,
    AppConfig,
    OllamaLLMConfig,
    OpenAILLMConfig,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DAGnostics API",
    description="Intelligent ETL Monitoring and Analysis API",
    version=__version__,
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AnalyzeRequest(BaseModel):
    dag_id: str
    task_id: str
    run_id: str
    try_number: int
    force_baseline_refresh: bool = False


class AnalyzeResponse(BaseModel):
    analysis_id: str
    dag_id: str
    task_id: str
    run_id: str
    error_message: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    suggested_actions: List[str] = []
    processing_time: float
    timestamp: datetime
    success: bool


class MonitorStatus(BaseModel):
    is_running: bool
    last_check: Optional[datetime] = None
    failed_tasks_count: int = 0
    processed_today: int = 0
    average_processing_time: float = 0.0


class DailySummary(BaseModel):
    date: str
    total_failures: int
    categories: dict
    top_failing_dags: List[dict]
    resolution_rate: float


# Dependency injection
def get_analyzer():
    """Get configured DAGAnalyzer instance"""
    from dagnostics.analysis.analyzer import DAGAnalyzer
    from dagnostics.clustering.log_clusterer import LogClusterer
    from dagnostics.core.airflow_client import AirflowClient
    from dagnostics.core.config import load_config
    from dagnostics.heuristics.pattern_filter import ErrorPatternFilter
    from dagnostics.llm.engine import LLMEngine, OllamaProvider

    config: AppConfig = load_config()

    airflow_client = AirflowClient(
        base_url=config.airflow.base_url,
        username=config.airflow.username,
        password=config.airflow.password,
        db_connection=config.airflow.database_url,
        verify_ssl=False,
    )

    # Accessing drain3.persistence_path directly via dot notation
    clusterer = LogClusterer(
        persistence_path=config.drain3.persistence_path, app_config=config
    )

    # If ErrorPatternFilter takes config_path from AppConfig:
    # filter = ErrorPatternFilter(config_path=config.pattern_filtering.config_path)
    # Otherwise, if it has sensible defaults, you can keep it as:
    filter = ErrorPatternFilter()

    # Determine which LLM provider to use based on config.llm.default_provider
    # This requires more robust handling than just assuming Ollama.
    # A proper implementation would involve a factory function or similar.

    llm_provider_config = config.llm.providers.get(config.llm.default_provider)
    if llm_provider_config is None:
        raise ValueError(
            f"LLM provider '{config.llm.default_provider}' not found in configuration."
        )

    if isinstance(llm_provider_config, OllamaLLMConfig):
        llm_provider = OllamaProvider(
            base_url=llm_provider_config.base_url,
            model=llm_provider_config.model,
        )
    elif isinstance(llm_provider_config, OpenAILLMConfig):
        # from dagnostics.llm.engine import OpenAIProvider
        # llm_provider = OpenAIProvider(
        #     api_key=llm_provider_config.api_key,
        #     model=llm_provider_config.model,
        #     base_url=llm_provider_config.base_url # This is why base_url was added as Optional
        # )
        raise NotImplementedError("OpenAIProvider not yet implemented or configured.")
    elif isinstance(llm_provider_config, AnthropicLLMConfig):
        # from dagnostics.llm.engine import AnthropicProvider
        # llm_provider = AnthropicProvider(
        #     api_key=llm_provider_config.api_key,
        #     model=llm_provider_config.model,
        #     base_url=llm_provider_config.base_url
        # )
        raise NotImplementedError(
            "AnthropicProvider not yet implemented or configured."
        )
    else:
        raise TypeError(
            f"Unknown LLM provider configuration type: {type(llm_provider_config)}"
        )

    llm = LLMEngine(llm_provider)

    return DAGAnalyzer(airflow_client, clusterer, filter, llm)


# API Endpoints
@app.get("/")
async def root():
    return {"message": "DAGnostics API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_task(
    request: AnalyzeRequest, analyzer: DAGAnalyzer = Depends(get_analyzer)
):
    """Analyze a specific task failure"""
    try:
        result = analyzer.analyze_task_failure(
            request.dag_id, request.task_id, request.run_id, request.try_number
        )

        return AnalyzeResponse(
            analysis_id=result.id,
            dag_id=result.dag_id,
            task_id=result.task_id,
            run_id=result.run_id,
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
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/monitor/status", response_model=MonitorStatus)
async def get_monitor_status():
    """Get monitoring service status"""
    # This would integrate with the actual monitor service
    return MonitorStatus(
        is_running=True,
        last_check=datetime.now(),
        failed_tasks_count=5,
        processed_today=23,
        average_processing_time=2.3,
    )


@app.post("/api/v1/monitor/start")
async def start_monitoring(background_tasks: BackgroundTasks):
    """Start monitoring service"""
    # background_tasks.add_task(start_monitor_service)
    return {"message": "Monitoring service started"}


@app.post("/api/v1/monitor/stop")
async def stop_monitoring():
    """Stop monitoring service"""
    return {"message": "Monitoring service stopped"}


@app.get("/api/v1/reports/daily", response_model=DailySummary)
async def get_daily_report(date: Optional[str] = None):
    """Get daily summary report"""
    target_date = (
        datetime.strptime(date, "%Y-%m-%d").date() if date else datetime.now().date()
    )

    # This would query the database for actual data
    return DailySummary(
        date=str(target_date),
        total_failures=15,
        categories={
            "resource_error": 5,
            "data_quality": 3,
            "dependency_failure": 4,
            "timeout_error": 2,
            "unknown": 1,
        },
        top_failing_dags=[
            {"dag_id": "etl_pipeline_1", "failures": 5},
            {"dag_id": "data_ingestion", "failures": 3},
            {"dag_id": "reporting_job", "failures": 2},
        ],
        resolution_rate=0.73,
    )


@app.get("/api/v1/baselines")
async def get_baselines():
    """Get all baseline information"""
    return {"baselines": {}, "message": "Baseline management endpoint"}


@app.post("/api/v1/baselines/rebuild")
async def rebuild_baseline(
    dag_id: str, task_id: str, analyzer: DAGAnalyzer = Depends(get_analyzer)
):
    """Rebuild baseline for specific DAG/task"""
    try:
        # Force rebuild baseline
        result = analyzer._ensure_baseline(dag_id, task_id)
        return {
            "message": f"Baseline rebuilt for {dag_id}.{task_id}",
            "result": result.__dict__,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
