"""Pydantic models for DAGnostics configuration."""

from typing import List

from pydantic import BaseModel, Field


class AppMetadata(BaseModel):
    """Application metadata configuration."""

    name: str = "DAGnostics"
    version: str = "0.1.0"
    description: str = "DAG monitoring and diagnostics system"


class RetryConfig(BaseModel):
    """Retry configuration for monitoring operations."""

    max_attempts: int = Field(default=3, ge=1)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    timeout: int = Field(default=30, ge=1)


class MonitoringConfig(BaseModel):
    """Monitoring system configuration."""

    interval: int = Field(
        default=60, ge=1, description="Monitoring interval in seconds"
    )
    log_path: str = Field(
        default="/var/log/airflow", description="Path to log files to monitor"
    )
    patterns: List[str] = Field(default=["*.log", "dag_*.log", "scheduler.log"])
    error_patterns: List[str] = Field(
        default=["ERROR", "CRITICAL", "Failed", "Exception"]
    )
    retry: RetryConfig = Field(default_factory=RetryConfig)


class EmailConfig(BaseModel):
    """Email notification configuration."""

    enabled: bool = False
    smtp_server: str = "localhost"
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    recipients: List[str] = Field(default_factory=list)


class SlackConfig(BaseModel):
    """Slack notification configuration."""

    enabled: bool = False
    webhook_url: str = ""
    channel: str = "#alerts"


class ReportingConfig(BaseModel):
    """Reporting system configuration."""

    format: str = Field(default="json", pattern=r"^(json|html|pdf)$")
    output_dir: str = "./reports"
    frequency: str = Field(default="daily", pattern=r"^(hourly|daily|weekly|monthly)$")
    email: EmailConfig = Field(default_factory=EmailConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)


class DatabaseConfig(BaseModel):
    """Database configuration."""

    enabled: bool = False
    type: str = Field(default="sqlite", pattern=r"^(sqlite|postgresql|mysql)$")
    host: str = "localhost"
    port: int = 5432
    name: str = "dagnostics"
    username: str = ""
    password: str = ""


class APIConfig(BaseModel):
    """API server configuration."""

    enabled: bool = True
    host: str = "localhost"
    port: int = Field(default=8080, ge=1, le=65535)
    debug: bool = False


class AppConfig(BaseModel):
    """Main application configuration."""

    app: AppMetadata = Field(default_factory=AppMetadata)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    log_level: str = Field(
        default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment
