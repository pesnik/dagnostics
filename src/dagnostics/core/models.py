from pydantic import BaseModel, Field


class MonitoringConfig(BaseModel):
    interval: int = Field(gt=0, description="Monitoring interval in seconds")
    log_path: str = Field(..., description="Path to store logs")


class ReportingConfig(BaseModel):
    format: str = Field(default="markdown", description="Report format")
    output_dir: str = Field(
        default="reports", description="Output directory for reports"
    )


class AppConfig(BaseModel):
    monitoring: MonitoringConfig
    reporting: ReportingConfig
