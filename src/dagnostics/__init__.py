"""
DAGnostics - Intelligent Airflow Task Failure Analysis

A comprehensive tool for analyzing, clustering, and diagnosing Airflow DAG task failures
using advanced heuristics and LLM-powered analysis.

Main API:
    from dagnostics import DAGnostics, get_error_message, analyze_task_failure

    # Using the client class
    client = DAGnostics()
    error_msg, candidates, error_line = client.get_error_message("dag_id", "task_id", "run_id", 1)

    # Using convenience functions
    error_msg, candidates, error_line = get_error_message("dag_id", "task_id", "run_id", 1)
    analysis_result = analyze_task_failure("dag_id", "task_id", "run_id", 1)
"""

from dagnostics.api_client import DAGnostics, analyze_task_failure, get_error_message

try:
    from importlib.metadata import version

    __version__ = version("dagnostics")
except ImportError:
    # Fallback for older Python versions
    import pkg_resources  # type: ignore

    __version__ = pkg_resources.get_distribution("dagnostics").version
__all__ = ["DAGnostics", "get_error_message", "analyze_task_failure"]
