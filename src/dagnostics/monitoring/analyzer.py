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
        config=None,
    ):
        self.airflow_client = airflow_client
        self.clusterer = clusterer
        self.filter = filter
        self.llm = llm
        self.config = config

    def _analyze_task_core(
        self, dag_id: str, task_id: str, run_id: str, try_number: int
    ) -> tuple[List[LogEntry], BaselineComparison]:
        """Core analysis workflow - common logic for both full analysis and SMS extraction"""
        logger.info(f"Starting core analysis for {dag_id}.{task_id}.{run_id}")

        # Step 1: Ensure baseline exists (stored or real-time)
        baseline_comparison = self._ensure_baseline(dag_id, task_id)

        # Step 2: Collect failed task logs
        failed_logs = self._collect_failed_logs(dag_id, task_id, run_id, try_number)

        if not failed_logs:
            return [], baseline_comparison

        # Step 3: Identify anomalous patterns using Drain3
        anomalous_logs = self.clusterer.identify_anomalous_patterns(
            failed_logs, dag_id, task_id
        )

        # Step 4: Filter known non-error patterns
        error_candidates = self.filter.filter_candidates(anomalous_logs)

        return error_candidates, baseline_comparison

    def analyze_task_failure(
        self, dag_id: str, task_id: str, run_id: str, try_number: int
    ) -> AnalysisResult:
        """Complete analysis workflow for a single task failure"""
        start_time = datetime.now()

        try:
            # Use core analysis logic
            error_candidates, baseline_comparison = self._analyze_task_core(
                dag_id, task_id, run_id, try_number
            )

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
                        related_logs=[],
                    ),
                    baseline_comparison=baseline_comparison,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                )

            # Step 5: Full LLM analysis with categorization and resolution
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
        """Extract error line for SMS notifications using Drain3 clustering and LLM analysis"""
        try:
            # Use core analysis logic
            error_candidates, _ = self._analyze_task_core(
                dag_id, task_id, run_id, try_number
            )

            if not error_candidates:
                return f"{dag_id}.{task_id}: No error patterns identified"

            error_line = self.llm.extract_error_line(error_candidates)

            return f"{dag_id}.{task_id}: {error_line}"

        except Exception as e:
            logger.error(
                f"Error extraction failed for {dag_id}.{task_id}.{run_id}: {e}"
            )
            return f"{dag_id}.{task_id}: Analysis failed - {str(e)}"

    def _ensure_baseline(self, dag_id: str, task_id: str) -> BaselineComparison:
        """Ensure baseline exists for the given dag/task based on configuration"""
        baseline_key = f"{dag_id}.{task_id}"

        # Check configuration for baseline usage
        use_stored_baseline = True  # Default to stored
        refresh_days = 7  # Default refresh days
        if self.config and hasattr(self.config, "monitoring"):
            use_stored_baseline = (
                self.config.monitoring.baseline_usage.value == "stored"
            )
            refresh_days = self.config.monitoring.baseline_refresh_days

        # Check if baseline exists and is recent (only if using stored baseline)
        if use_stored_baseline and baseline_key in self.clusterer.baseline_clusters:
            # Check if baseline is stale and needs refresh
            if self.clusterer.is_baseline_stale(dag_id, task_id, refresh_days):
                logger.info(
                    f"Baseline for {dag_id}.{task_id} is stale ({self.clusterer.get_baseline_age_days(dag_id, task_id)} days old), refreshing..."
                )
                # Fall through to build new baseline
            else:
                # Baseline is current, use it
                baseline_age_days = self.clusterer.get_baseline_age_days(
                    dag_id, task_id
                )
                logger.debug(
                    f"Using existing baseline for {dag_id}.{task_id} (age: {baseline_age_days} days)"
                )
                return BaselineComparison(
                    is_known_pattern=False,  # Will be updated during analysis
                    similar_clusters=[],
                    novelty_score=0.0,
                    baseline_age_days=baseline_age_days,
                )

        # Build new baseline (either real-time, stale stored, or because stored doesn't exist)
        baseline_type = "real-time" if not use_stored_baseline else "stored"
        logger.info(f"Building {baseline_type} baseline for {dag_id}.{task_id}")

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

        baseline_age_days = self.clusterer.get_baseline_age_days(dag_id, task_id)
        return BaselineComparison(
            is_known_pattern=False,
            similar_clusters=[],
            novelty_score=0.5,
            baseline_age_days=baseline_age_days,
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
