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

        # Step 1: Ensure baseline exists and get comparison info
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

        # Update baseline comparison with analysis results
        baseline_comparison.is_known_pattern = len(error_candidates) == 0
        if len(failed_logs) > 0:
            baseline_comparison.novelty_score = len(anomalous_logs) / len(failed_logs)
        else:
            baseline_comparison.novelty_score = 0.0

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

            return f"DAG: {dag_id} Task: {task_id}: {error_line}"

        except Exception as e:
            logger.error(
                f"Error extraction failed for {dag_id}.{task_id}.{run_id}: {e}"
            )
            return f"{dag_id}.{task_id}: Analysis failed - {str(e)}"

    def _ensure_baseline(self, dag_id: str, task_id: str) -> BaselineComparison:
        """Ensure baseline exists for the given dag/task based on configuration"""
        # Get configuration parameters
        refresh_days = self._get_baseline_refresh_days()

        # Check if baseline refresh is needed
        if (
            self.clusterer.is_baseline_stale(dag_id, task_id, refresh_days)
            or self.clusterer.baseline_cluster_size(dag_id, task_id) < 1
        ):
            logger.info(
                f"Baseline for {dag_id}.{task_id} is stale or missing "
                f"({self.clusterer.get_baseline_age_days(dag_id, task_id)} days old), building new baseline..."
            )

            # Collect successful task data for baseline
            successful_logs = self._collect_baseline_logs(dag_id, task_id)

            if successful_logs:
                # Build new baseline
                self.clusterer.build_baseline_clusters(successful_logs, dag_id, task_id)
                logger.info(
                    f"Built new baseline for {dag_id}.{task_id} with {len(successful_logs)} log entries"
                )
            else:
                logger.warning(
                    f"No successful logs found to build baseline for {dag_id}.{task_id}"
                )
        else:
            baseline_age_days = self.clusterer.get_baseline_age_days(dag_id, task_id)
            logger.debug(
                f"Using existing baseline for {dag_id}.{task_id} (age: {baseline_age_days} days)"
            )

        # Return baseline comparison structure
        baseline_age_days = self.clusterer.get_baseline_age_days(dag_id, task_id)
        return BaselineComparison(
            is_known_pattern=False,  # Will be updated during analysis
            similar_clusters=[],  # Could be enhanced to find similar patterns
            novelty_score=0.0,  # Will be updated during analysis
            baseline_age_days=baseline_age_days,
        )

    def _collect_baseline_logs(self, dag_id: str, task_id: str) -> List[LogEntry]:
        """Collect logs from successful task runs for baseline creation"""
        baseline_logs = []

        try:
            # Get recent successful tasks
            successful_tasks = self.airflow_client.get_successful_tasks(
                dag_id, task_id, limit=self._get_baseline_success_count()
            )

            if not successful_tasks:
                logger.warning(f"No successful tasks found for {dag_id}.{task_id}")
                return []

            # Collect logs from successful tasks
            for task in successful_tasks:
                try:
                    logs_content = self.airflow_client.get_task_logs(
                        task.dag_id, task.task_id, task.run_id
                    )
                    parsed_logs = self._parse_logs(logs_content, task)
                    baseline_logs.extend(parsed_logs)
                    logger.debug(
                        f"Collected {len(parsed_logs)} logs from successful run {task.run_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to collect baseline logs for {task.run_id}: {e}"
                    )

            logger.info(
                f"Collected total of {len(baseline_logs)} baseline logs for {dag_id}.{task_id}"
            )
            return baseline_logs

        except Exception as e:
            logger.error(f"Failed to collect baseline logs for {dag_id}.{task_id}: {e}")
            return []

    def _collect_failed_logs(
        self, dag_id: str, task_id: str, run_id: str, try_number: int
    ) -> List[LogEntry]:
        """Collect and parse logs from failed task"""
        try:
            logs_content = self.airflow_client.get_task_logs(
                dag_id, task_id, run_id, try_number
            )

            task_instance = TaskInstance(
                dag_id=dag_id,
                task_id=task_id,
                run_id=run_id,
                state="failed",
                try_number=try_number,
            )

            parsed_logs = self._parse_logs(logs_content, task_instance)
            logger.info(
                f"Collected {len(parsed_logs)} failed logs for {dag_id}.{task_id}.{run_id}"
            )
            return parsed_logs

        except Exception as e:
            logger.error(f"Failed to collect logs for {dag_id}.{task_id}.{run_id}: {e}")
            return []

    def _parse_logs(self, logs_content: str, task: TaskInstance) -> List[LogEntry]:
        """Parse raw log content into LogEntry objects"""
        if not logs_content:
            return []

        log_entries = []
        lines = logs_content.split("\n")

        # Apply max log lines limit if configured
        max_log_lines = self._get_max_log_lines()
        if len(lines) > max_log_lines:
            logger.warning(f"Truncating log from {len(lines)} to {max_log_lines} lines")
            lines = lines[
                -max_log_lines:
            ]  # Keep the last N lines (usually most relevant)

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

    def _get_baseline_refresh_days(self) -> int:
        """Get baseline refresh days from config"""
        if self.config and hasattr(self.config, "monitoring"):
            return self.config.monitoring.baseline_refresh_days
        return 7  # Default

    def _get_baseline_success_count(self) -> int:
        """Get number of successful tasks to use for baseline"""
        if self.config and hasattr(self.config, "monitoring"):
            return self.config.monitoring.baseline_success_count
        return 3  # Default

    def _get_max_log_lines(self) -> int:
        """Get maximum log lines to process"""
        if self.config and hasattr(self.config, "monitoring"):
            return self.config.monitoring.max_log_lines
        return 1000  # Default

    def cleanup_old_baselines(self, retention_days: int = 30):
        """Cleanup old baseline files - delegates to clusterer"""
        try:
            self.clusterer.cleanup_old_baselines(retention_days)
            logger.info(f"Cleaned up baselines older than {retention_days} days")
        except Exception as e:
            logger.error(f"Failed to cleanup old baselines: {e}")

    def get_baseline_stats(
        self, dag_id: str | None = None, task_id: str | None = None
    ) -> dict:
        """Get statistics about current baselines"""
        stats = {
            "total_baselines": 0,
            "average_age_days": 0,
            "oldest_baseline_days": 0,
            "newest_baseline_days": 0,
            "baselines_by_age": {"fresh": 0, "moderate": 0, "stale": 0},
        }

        try:
            if dag_id and task_id:
                # Stats for specific dag/task
                age_days = self.clusterer.get_baseline_age_days(dag_id, task_id)
                stats.update(
                    {
                        "total_baselines": 1 if age_days > 0 else 0,
                        "average_age_days": age_days,
                        "oldest_baseline_days": age_days,
                        "newest_baseline_days": age_days,
                    }
                )
            else:
                # Global stats (would need to iterate through all baselines)
                # This would require extending the clusterer to expose baseline timestamps
                logger.info("Global baseline stats not yet implemented")

        except Exception as e:
            logger.error(f"Failed to get baseline stats: {e}")

        return stats
