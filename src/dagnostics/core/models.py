import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union  # Added Dict, Any for config types


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    RESOURCE_ERROR = "resource_error"
    DATA_QUALITY = "data_quality"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN = "unknown"


@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    message: str
    source: str
    dag_id: str
    task_id: str
    run_id: str
    line_number: Optional[int] = None
    raw_content: str = ""


@dataclass
class DrainCluster:
    cluster_id: str
    template: str
    log_ids: List[str]
    size: int
    created_at: datetime
    last_updated: datetime


@dataclass
class BaselineCluster:
    cluster_id: str
    template: str
    log_count: int
    last_updated: datetime
    dag_id: str
    task_id: str
    confidence_score: float = 0.0


@dataclass
class ErrorAnalysis:
    error_message: str
    confidence: float
    category: ErrorCategory
    severity: ErrorSeverity
    suggested_actions: List[str]
    related_logs: List[LogEntry]
    raw_error_lines: List[str] = field(default_factory=list)
    llm_reasoning: str = ""


@dataclass
class BaselineComparison:
    is_known_pattern: bool
    similar_clusters: List[str]
    novelty_score: float
    baseline_age_days: int


@dataclass
class AnalysisResult:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dag_id: str = ""
    task_id: str = ""
    run_id: str = ""
    analysis: Optional[ErrorAnalysis] = None
    baseline_comparison: Optional[BaselineComparison] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""


@dataclass
class TaskInstance:
    dag_id: str
    task_id: str
    run_id: str
    execution_date: datetime
    state: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    log_url: str = ""


@dataclass
class AirflowConfig:
    base_url: str
    username: str
    password: str
    database_url: str
    verify_ssl: bool
    timeout: int


@dataclass
class Drain3Config:
    depth: int
    sim_th: float
    max_children: int
    max_clusters: int
    extra_delimiters: List[str]
    persistence_path: str


@dataclass
class OllamaLLMConfig:
    base_url: str
    model: str
    temperature: float


@dataclass
class OpenAILLMConfig:
    api_key: str
    model: str
    temperature: float
    base_url: Optional[str] = None


@dataclass
class AnthropicLLMConfig:
    api_key: str
    model: str
    temperature: float
    base_url: Optional[str] = None


@dataclass
class LLMConfig:
    default_provider: str
    providers: Dict[str, Union[OllamaLLMConfig, OpenAILLMConfig, AnthropicLLMConfig]]


@dataclass
class MonitoringConfig:
    check_interval_minutes: int
    baseline_success_count: int
    max_log_lines: int
    failed_task_lookback_hours: int
    baseline_refresh_days: int


@dataclass
class LogProcessingConfig:
    max_log_size_mb: int
    chunk_size_lines: int
    timeout_seconds: int


@dataclass
class PatternFilteringConfig:
    config_path: str
    custom_patterns_enabled: bool


@dataclass
class SMSAlertConfig:
    enabled: bool
    provider: str
    account_sid: str
    auth_token: str
    from_number: str


@dataclass
class EmailAlertConfig:
    enabled: bool
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_address: str


@dataclass
class AlertsConfig:
    sms: SMSAlertConfig
    email: EmailAlertConfig


@dataclass
class ReportingConfig:
    output_dir: str
    daily_report_time: str
    retention_days: int
    formats: List[str]


@dataclass
class DatabaseConfig:
    url: str
    echo: bool
    pool_size: int
    max_overflow: int


@dataclass
class APIConfig:
    host: str
    port: int
    workers: int
    reload: bool
    log_level: str


@dataclass
class WebConfig:
    enabled: bool
    host: str
    port: int
    debug: bool


@dataclass
class AppConfig:
    """Main application configuration structure."""

    airflow: AirflowConfig
    drain3: Drain3Config
    llm: LLMConfig
    monitoring: "MonitoringConfig"
    log_processing: "LogProcessingConfig"
    pattern_filtering: "PatternFilteringConfig"
    alerts: "AlertsConfig"
    reporting: "ReportingConfig"
    database: "DatabaseConfig"
    api: "APIConfig"
    web: "WebConfig"
