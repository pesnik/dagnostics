from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# Request Models
class AnalyzeRequest(BaseModel):
    dag_id: str
    task_id: str
    run_id: str
    try_number: int = 1
    force_baseline_refresh: bool = False


# Response Models
class AnalyzeResponse(BaseModel):
    analysis_id: str
    dag_id: str
    task_id: str
    run_id: str
    try_number: int
    error_message: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    suggested_actions: List[str] = []
    processing_time: float
    timestamp: datetime
    success: bool
    baseline_comparison: Optional[Dict[str, Any]] = None


class MonitorStatus(BaseModel):
    is_running: bool
    last_check: Optional[datetime] = None
    failed_tasks_count: int = 0
    processed_today: int = 0
    average_processing_time: float = 0.0
    uptime: int = 0  # seconds
    next_check: Optional[datetime] = None


class DashboardStats(BaseModel):
    total_failures: int
    today_failures: int
    average_processing_time: float
    resolution_rate: float
    error_categories: Dict[str, int]
    top_failing_dags: List[Dict[str, Any]]
    period_days: int


class ErrorTrendData(BaseModel):
    date: str
    failures: int
    resolved: int = 0


class RecentFailureItem(BaseModel):
    dag_id: str
    task_id: str
    run_id: str
    timestamp: datetime
    error_message: str
    category: str
    severity: str


class FailureTimelineItem(BaseModel):
    timestamp: datetime
    dag_id: str
    task_id: str
    severity: str
    category: str


class AnalysisHistoryItem(BaseModel):
    analysis_id: str
    dag_id: str
    task_id: str
    run_id: str
    timestamp: datetime
    success: bool
    category: Optional[str] = None
    severity: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float


class BaselineInfo(BaseModel):
    dag_id: str
    task_id: str
    cluster_count: int
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    successful_runs: int = 0
    is_stale: bool = False


class DailySummary(BaseModel):
    date: str
    total_failures: int
    categories: Dict[str, int]
    top_failing_dags: List[Dict[str, Any]]
    resolution_rate: float


# WebSocket Models
class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()


class RealTimeUpdate(BaseModel):
    event_type: str  # "new_failure", "analysis_complete", "status_change"
    dag_id: Optional[str] = None
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    data: Dict[str, Any] = {}
    timestamp: datetime = datetime.now()


# Training Dataset Models
class ErrorCandidate(BaseModel):
    id: str
    dag_id: str
    task_id: str
    run_id: str
    error_message: str
    raw_logs: str
    llm_category: Optional[str] = None
    llm_severity: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_reasoning: Optional[str] = None
    human_category: Optional[str] = None
    human_severity: Optional[str] = None
    human_feedback: Optional[str] = None
    status: str = "pending"  # pending, approved, rejected, modified
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    processing_time: float


class FeedbackRequest(BaseModel):
    candidate_id: str
    action: str  # approve, reject, modify
    human_category: Optional[str] = None
    human_severity: Optional[str] = None
    feedback_notes: Optional[str] = None
    reviewer_name: Optional[str] = None


class CandidateStats(BaseModel):
    total_candidates: int
    pending_review: int
    approved: int
    rejected: int
    modified: int
    accuracy_rate: float
    avg_confidence: float


class DatasetExportRequest(BaseModel):
    format: str  # json, csv, jsonl
    include_rejected: bool = False
    category_filter: Optional[List[str]] = None
    severity_filter: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None
