import logging
from datetime import datetime
from typing import List

from dagnostics.core.models import (
    AnalysisResult,
    BaselineComparison,
    ErrorAnalysis,
    ErrorCategory,
    ErrorSeverity,
    LogEntry,
    TaskInstance,
)
from dagnostics.llm.engine import LLMEngine
from dagnostics.llm.log_clusterer import LogClusterer
from dagnostics.llm.pattern_filter import ErrorPatternFilter
from dagnostics.monitoring.airflow_client import AirflowClient

logger = logging.getLogger(__name__)


class DAGAnalyzer:
    """Main analysis orchestrator that combines all components"""

    def __init__(
        self,
        airflow_client: AirflowClient,
        clusterer: LogClusterer,
        filter: ErrorPatternFilter,
        llm: LLMEngine,
    ):
        self.airflow_client = airflow_client
        self.clusterer = clusterer
        self.filter = filter
        self.llm = llm

    def analyze_task_failure(
        self, dag_id: str, task_id: str, run_id: str, try_number: int
    ) -> AnalysisResult:
        """Complete analysis workflow for a single task failure"""
        start_time = datetime.now()

        try:
            logger.info(f"Starting analysis for {dag_id}.{task_id}.{run_id}")

            # Step 1: Ensure baseline exists
            baseline_comparison = self._ensure_baseline(dag_id, task_id)

            # Step 2: Collect failed task logs
            failed_logs = self._collect_failed_logs(dag_id, task_id, run_id, try_number)

            if not failed_logs:
                return AnalysisResult(
                    dag_id=dag_id,
                    task_id=task_id,
                    run_id=run_id,
                    success=False,
                    error_message="No logs found for failed task",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            # Step 3: Identify anomalous patterns using Drain3
            anomalous_logs = self.clusterer.identify_anomalous_patterns(
                failed_logs, dag_id, task_id
            )

            # Step 4: Filter known non-error patterns
            error_candidates = self.filter.filter_candidates(anomalous_logs)

            if not error_candidates:
                return AnalysisResult(
                    dag_id=dag_id,
                    task_id=task_id,
                    run_id=run_id,
                    analysis=ErrorAnalysis(
                        error_message="No error patterns identified",
                        confidence=0.1,
                        category=ErrorCategory.UNKNOWN,
                        severity=ErrorSeverity.LOW,
                        suggested_actions=["Review logs manually"],
                        related_logs=failed_logs,
                    ),
                    baseline_comparison=baseline_comparison,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            # Step 5: LLM analysis
            error_analysis = self.llm.extract_error_message(error_candidates)

            # Step 6: Generate resolution suggestions
            error_analysis.suggested_actions = self.llm.suggest_resolution(
                error_analysis
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            return AnalysisResult(
                dag_id=dag_id,
                task_id=task_id,
                run_id=run_id,
                analysis=error_analysis,
                baseline_comparison=baseline_comparison,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Analysis failed for {dag_id}.{task_id}.{run_id}: {e}")
            return AnalysisResult(
                dag_id=dag_id,
                task_id=task_id,
                run_id=run_id,
                success=False,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds(),
            )

    def extract_task_error_for_sms(
        self, dag_id: str, task_id: str, run_id: str, try_number: int
    ) -> str:
        """Extract error line for SMS notifications using LLM analysis"""
        try:
            logger.info(f"Extracting error for SMS: {dag_id}.{task_id}.{run_id}")

            # Collect failed task logs
            failed_logs = self._collect_failed_logs(dag_id, task_id, run_id, try_number)

            if not failed_logs:
                return f"{dag_id}.{task_id}: No logs found"

            # Use LLM-based error extraction for SMS notifications
            error_line = self.llm.extract_error_line(failed_logs)

            return f"{dag_id}.{task_id}: {error_line}"

        except Exception as e:
            logger.error(
                f"Error extraction failed for {dag_id}.{task_id}.{run_id}: {e}"
            )
            return f"{dag_id}.{task_id}: Analysis failed - {str(e)}"

    def _ensure_baseline(self, dag_id: str, task_id: str) -> BaselineComparison:
        """Ensure baseline exists for the given dag/task"""
        baseline_key = f"{dag_id}.{task_id}"

        # Check if baseline exists and is recent
        if baseline_key in self.clusterer.baseline_clusters:
            # For simplicity, assume baseline is always current
            # In production, you'd check timestamp and refresh if needed
            return BaselineComparison(
                is_known_pattern=False,  # Will be updated during analysis
                similar_clusters=[],
                novelty_score=0.0,
                baseline_age_days=1,  # Placeholder
            )

        # Build new baseline
        logger.info(f"Building baseline for {dag_id}.{task_id}")
        successful_tasks = self.airflow_client.get_successful_tasks(
            dag_id, task_id, limit=3
        )

        if not successful_tasks:
            logger.warning(f"No successful tasks found for {dag_id}.{task_id}")
            return BaselineComparison(
                is_known_pattern=False,
                similar_clusters=[],
                novelty_score=1.0,  # High novelty when no baseline
                baseline_age_days=0,
            )

        # Collect logs from successful tasks
        baseline_logs = []
        for task in successful_tasks:
            try:
                logs_content = self.airflow_client.get_task_logs(
                    task.dag_id, task.task_id, task.run_id
                )
                parsed_logs = self._parse_logs(logs_content, task)
                baseline_logs.extend(parsed_logs)
            except Exception as e:
                logger.warning(
                    f"Failed to collect baseline logs for {task.run_id}: {e}"
                )

        # Build clusters from baseline logs
        if baseline_logs:
            self.clusterer.build_baseline_clusters(baseline_logs, dag_id, task_id)

        return BaselineComparison(
            is_known_pattern=False,
            similar_clusters=[],
            novelty_score=0.5,
            baseline_age_days=0,
        )

    def _collect_failed_logs(
        self, dag_id: str, task_id: str, run_id: str, try_number: int
    ) -> List[LogEntry]:
        """Collect and parse logs from failed task"""
        try:
            logs_content = self.airflow_client.get_task_logs(
                dag_id, task_id, run_id, try_number
            )
            return self._parse_logs(
                logs_content,
                TaskInstance(
                    dag_id=dag_id,
                    task_id=task_id,
                    run_id=run_id,
                    state="failed",
                    try_number=try_number,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to collect logs for {dag_id}.{task_id}.{run_id}: {e}")
            return []

    def _parse_logs(self, logs_content: str, task: TaskInstance) -> List[LogEntry]:
        """Parse raw log content into LogEntry objects"""
        if not logs_content:
            return []

        log_entries = []
        lines = logs_content.split("\n")

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            # Try to extract timestamp and level
            timestamp = datetime.now()  # Fallback
            level = "INFO"  # Fallback
            message = line.strip()

            # Simple regex for common log formats
            import re

            timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if timestamp_match:
                try:
                    timestamp = datetime.strptime(
                        timestamp_match.group(1), "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    pass

            level_match = re.search(
                r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL)\b", line, re.IGNORECASE
            )
            if level_match:
                level = level_match.group(1).upper()

            log_entries.append(
                LogEntry(
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    source="airflow",
                    dag_id=task.dag_id,
                    task_id=task.task_id,
                    run_id=task.run_id,
                    line_number=i + 1,
                    raw_content=line,
                )
            )

        return log_entries
