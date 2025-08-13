"""
DAGnostics Python Package API

Provides a clean interface for programmatic usage of DAGnostics functionality.
"""

from typing import List, Optional, Tuple

from dagnostics.analysis.analyzer import DAGAnalyzer
from dagnostics.cli.utils import initialize_components
from dagnostics.core.models import AnalysisResult, AppConfig, LogEntry


class DAGnostics:
    """
    Main API client for DAGnostics functionality.

    Example usage:
        from dagnostics import DAGnostics

        client = DAGnostics()
        error_msg, candidates, error_line = client.get_error_message(
            "my_dag", "my_task", "2023-01-01", 1
        )
    """

    def __init__(self, config_file: Optional[str] = None, llm_provider: str = "ollama"):
        """
        Initialize DAGnostics client.

        Args:
            config_file: Path to configuration file (default: searches standard locations)
            llm_provider: LLM provider to use (ollama, openai, anthropic, gemini)
        """
        self.config_file = config_file
        self.llm_provider = llm_provider
        self._config: Optional[AppConfig] = None
        self._analyzer: Optional[DAGAnalyzer] = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of components"""
        if self._analyzer is None:
            self._config, self._analyzer = initialize_components(
                self.config_file, self.llm_provider
            )

        # Ensure analyzer is initialized (should never happen)
        if self._analyzer is None:
            raise RuntimeError("Failed to initialize analyzer")

    def get_error_message(
        self, dag_id: str, task_id: str, run_id: str, try_number: int
    ) -> Tuple[str, Optional[List[LogEntry]], str]:
        """
        Extract error message from a failed Airflow task.

        Args:
            dag_id: ID of the DAG
            task_id: ID of the task
            run_id: Run ID of the task instance
            try_number: Attempt number of the task

        Returns:
            Tuple containing:
            - error_message: Processed error summary
            - error_candidates: List of potential error log entries
            - error_line: Raw error line from logs

        Raises:
            Exception: If error extraction fails
        """
        self._ensure_initialized()
        analyzer = self._analyzer
        if analyzer is None:
            raise RuntimeError("Analyzer not initialized")
        return analyzer.extract_task_error_for_sms(dag_id, task_id, run_id, try_number)

    def analyze_task_failure(
        self, dag_id: str, task_id: str, run_id: str, try_number: int
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis of a task failure.

        Args:
            dag_id: ID of the DAG
            task_id: ID of the task
            run_id: Run ID of the task instance
            try_number: Attempt number of the task

        Returns:
            AnalysisResult object containing detailed analysis

        Raises:
            Exception: If analysis fails
        """
        self._ensure_initialized()
        analyzer = self._analyzer
        if analyzer is None:
            raise RuntimeError("Analyzer not initialized")
        return analyzer.analyze_task_failure(dag_id, task_id, run_id, try_number)

    def get_failed_tasks(self, since_minutes: int = 60) -> List:
        """
        Get list of failed tasks from Airflow within the specified time window.

        Args:
            since_minutes: Look back window in minutes (default: 60)

        Returns:
            List of failed task instances

        Raises:
            Exception: If fetching failed tasks fails
        """
        self._ensure_initialized()
        analyzer = self._analyzer
        if analyzer is None:
            raise RuntimeError("Analyzer not initialized")
        return analyzer.airflow_client.get_failed_tasks(since_minutes)


# Convenience functions for direct import
def get_error_message(
    dag_id: str,
    task_id: str,
    run_id: str,
    try_number: int,
    config_file: Optional[str] = None,
    llm_provider: str = "ollama",
) -> Tuple[str, Optional[List[LogEntry]], str]:
    """
    Extract error message from a failed Airflow task.

    Convenience function that doesn't require creating a client instance.

    Args:
        dag_id: ID of the DAG
        task_id: ID of the task
        run_id: Run ID of the task instance
        try_number: Attempt number of the task
        config_file: Path to configuration file (optional)
        llm_provider: LLM provider to use (default: ollama)

    Returns:
        Tuple containing (error_message, error_candidates, error_line)
    """
    from dagnostics.cli.utils import get_error_message as _get_error_message

    return _get_error_message(
        dag_id, task_id, run_id, try_number, config_file, llm_provider
    )


def analyze_task_failure(
    dag_id: str,
    task_id: str,
    run_id: str,
    try_number: int,
    config_file: Optional[str] = None,
    llm_provider: str = "ollama",
) -> AnalysisResult:
    """
    Perform comprehensive analysis of a task failure.

    Convenience function that doesn't require creating a client instance.

    Args:
        dag_id: ID of the DAG
        task_id: ID of the task
        run_id: Run ID of the task instance
        try_number: Attempt number of the task
        config_file: Path to configuration file (optional)
        llm_provider: LLM provider to use (default: ollama)

    Returns:
        AnalysisResult object containing detailed analysis
    """
    client = DAGnostics(config_file, llm_provider)
    return client.analyze_task_failure(dag_id, task_id, run_id, try_number)
