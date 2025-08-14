import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, EmailStr, Field, HttpUrl


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


class BaselineUsage(str, Enum):
    STORED = "stored"
    REAL_TIME = "real_time"


class OutputFormat(str, Enum):
    json = "json"
    yaml = "yaml"
    text = "text"


class ReportFormat(str, Enum):
    html = "html"
    json = "json"
    pdf = "pdf"


class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    message: str
    source: str
    dag_id: str
    task_id: str
    run_id: str
    line_number: Optional[int] = None
    raw_content: str = ""


class DrainCluster(BaseModel):
    cluster_id: str
    template: str
    log_ids: List[str]
    size: int
    created_at: datetime
    last_updated: datetime


class BaselineCluster(BaseModel):
    cluster_id: str
    template: str
    log_count: int
    last_updated: datetime
    dag_id: str
    task_id: str
    confidence_score: float = 0.0


class ErrorAnalysis(BaseModel):
    error_message: str
    confidence: float
    category: ErrorCategory
    severity: ErrorSeverity
    suggested_actions: List[str]
    related_logs: List[LogEntry] = Field(default_factory=list)
    raw_error_lines: List[str] = Field(default_factory=list)
    llm_reasoning: str = ""


class BaselineComparison(BaseModel):
    is_known_pattern: bool
    similar_clusters: List[str]
    novelty_score: float
    baseline_age_days: int


class AnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dag_id: str = ""
    task_id: str = ""
    run_id: str = ""
    analysis: Optional[ErrorAnalysis] = None
    baseline_comparison: Optional[BaselineComparison] = None
    processing_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""


class TaskInstance(BaseModel):
    dag_id: str
    task_id: str
    run_id: str
    state: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    log_url: Union[HttpUrl, Literal[""]] = ""
    try_number: int


class AirflowConfig(BaseModel):
    base_url: HttpUrl
    username: str
    password: str
    database_url: str
    verify_ssl: bool = True
    timeout: int = Field(..., ge=1)
    db_timezone_offset: str = Field(
        default="+00:00",
        description="Database timezone offset (e.g., '+06:00', '-05:00')",
    )


class Drain3Config(BaseModel):
    depth: int = Field(..., ge=0)
    sim_th: float = Field(..., ge=0.0, le=1.0)
    max_children: int = Field(..., ge=1)
    max_clusters: int = Field(..., ge=1)
    extra_delimiters: List[str] = Field(default_factory=list)
    persistence_path: str
    config_path: str


class OllamaLLMConfig(BaseModel):
    base_url: HttpUrl
    model: str
    temperature: float = Field(..., ge=0.0, le=1.0)


class OpenAILLMConfig(BaseModel):
    api_key: str = Field(..., min_length=1)
    model: str
    temperature: float = Field(..., ge=0.0, le=1.0)
    base_url: Optional[HttpUrl] = None


class AnthropicLLMConfig(BaseModel):
    api_key: str = Field(..., min_length=1)
    model: str
    temperature: float = Field(..., ge=0.0, le=1.0)
    base_url: Optional[HttpUrl] = None


class GeminiLLMConfig(BaseModel):
    api_key: str = Field(..., min_length=1)
    model: str
    temperature: float = Field(..., ge=0.0, le=1.0)
    max_output_tokens: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)


class LLMConfig(BaseModel):
    default_provider: str
    providers: Dict[
        str,
        Union[OllamaLLMConfig, OpenAILLMConfig, AnthropicLLMConfig, GeminiLLMConfig],
    ]


class MonitoringConfig(BaseModel):
    check_interval_minutes: int = Field(..., ge=1)
    baseline_success_count: int = Field(..., ge=0)
    max_log_lines: int = Field(..., ge=1)
    failed_task_lookback_hours: int = Field(..., ge=0)
    baseline_refresh_days: int = Field(..., ge=1)
    baseline_usage: BaselineUsage = BaselineUsage.STORED


class LogProcessingConfig(BaseModel):
    max_log_size_mb: int = Field(..., ge=1)
    chunk_size_lines: int = Field(..., ge=1)
    timeout_seconds: int = Field(..., ge=1)


class PatternFilteringConfig(BaseModel):
    config_path: str
    custom_patterns_enabled: bool


class SMSAlertConfig(BaseModel):
    enabled: bool
    provider: str
    # Twilio-specific fields
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""
    # Custom gateway fields
    base_url: str = ""
    path: str = ""
    static_params: Dict[str, str] = Field(default_factory=dict)
    param_mapping: Dict[str, str] = Field(default_factory=dict)
    # Recipient configuration
    default_recipients: List[str] = Field(default_factory=list)
    task_recipients: Dict[str, List[str]] = Field(
        default_factory=dict
    )  # task_pattern -> phone_numbers


class EmailAlertConfig(BaseModel):
    enabled: bool
    smtp_server: str
    smtp_port: int = Field(..., gt=0)
    username: str
    password: str
    from_address: EmailStr


class AlertsConfig(BaseModel):
    sms: SMSAlertConfig
    email: EmailAlertConfig


class ReportingConfig(BaseModel):
    output_dir: Path
    daily_report_time: str
    retention_days: int = Field(..., ge=0)
    formats: List[str]


class DatabaseConfig(BaseModel):
    url: str
    echo: bool
    pool_size: int = Field(..., ge=1)
    max_overflow: int = Field(..., ge=0)


class APIConfig(BaseModel):
    host: str
    port: int = Field(..., gt=0, le=65535)
    workers: int = Field(..., ge=1)
    reload: bool
    log_level: str  # Consider Enum for allowed log levels (e.g., "INFO", "DEBUG")


class WebConfig(BaseModel):
    enabled: bool
    host: str
    port: int = Field(..., gt=0, le=65535)
    debug: bool


class FewShotExample(BaseModel):
    log_context: str
    extracted_response: str


class PromptsConfig(BaseModel):
    few_shot_examples: Dict[str, List[FewShotExample]] = Field(default_factory=dict)
    templates: Dict[str, str] = Field(default_factory=dict)


class FeedbackBackupConfig(BaseModel):
    enabled: bool = True
    interval_hours: int = 24
    backup_dir: str = "data/backups"
    max_backups: int = 7


class FeedbackStorageConfig(BaseModel):
    database_url: str = "sqlite:///data/feedback.db"
    jsonl_file: str = "data/feedback_data.jsonl"
    training_data_dir: str = "data/training"
    raw_logs_dir: str = "data/raw"
    retention_days: int = 365
    cleanup_enabled: bool = True
    backup: FeedbackBackupConfig = Field(default_factory=FeedbackBackupConfig)


class FeedbackCollectionConfig(BaseModel):
    min_quality_rating: int = 3
    auto_export_threshold: int = 50
    categories: List[str] = Field(
        default_factory=lambda: [
            "configuration_error",
            "timeout_error",
            "data_quality",
            "dependency_failure",
            "resource_error",
            "permission_error",
            "network_error",
            "unknown",
        ]
    )
    severities: List[str] = Field(
        default_factory=lambda: ["low", "medium", "high", "critical"]
    )


class FeedbackNotificationConfig(BaseModel):
    enabled: bool = False
    email_template: str = "feedback_request.html"
    slack_webhook: Optional[str] = None


class FeedbackIntegrationConfig(BaseModel):
    airflow_webhook: Optional[str] = None
    notifications: FeedbackNotificationConfig = Field(
        default_factory=FeedbackNotificationConfig
    )


class FeedbackWebInterfaceConfig(BaseModel):
    enabled: bool = True
    auto_redirect: bool = True
    session_timeout: int = 3600


class FeedbackConfig(BaseModel):
    storage: FeedbackStorageConfig = Field(default_factory=FeedbackStorageConfig)
    collection: FeedbackCollectionConfig = Field(
        default_factory=FeedbackCollectionConfig
    )
    web_interface: FeedbackWebInterfaceConfig = Field(
        default_factory=FeedbackWebInterfaceConfig
    )
    integration: FeedbackIntegrationConfig = Field(
        default_factory=FeedbackIntegrationConfig
    )


class AppConfig(BaseModel):
    """Main application configuration structure."""

    airflow: AirflowConfig
    drain3: Drain3Config
    llm: LLMConfig
    monitoring: MonitoringConfig
    log_processing: LogProcessingConfig
    pattern_filtering: PatternFilteringConfig
    alerts: AlertsConfig
    reporting: ReportingConfig
    database: DatabaseConfig
    api: APIConfig
    web: WebConfig
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    prompts: Optional[PromptsConfig] = None
