import logging
from typing import List

import requests
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine, text

from dagnostics.core.models import TaskInstance

logger = logging.getLogger(__name__)


class AirflowAPIClient:
    """Client for Airflow REST API interactions"""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.auth = HTTPBasicAuth(username, password)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.verify = False  # For self-signed certificates

    def get_task_logs(
        self, dag_id: str, task_id: str, run_id: str, try_number: int = 1
    ) -> str:
        """Fetch task logs from Airflow API"""
        url = f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}"

        try:
            response = self.session.get(url)
            response.raise_for_status()

            # Extract log content from response
            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
                return data.get("content", "")
            else:
                return response.text

        except requests.RequestException as e:
            logger.error(f"Failed to fetch logs for {dag_id}.{task_id}: {e}")
            raise


class AirflowDBClient:
    """Client for Airflow MetaDB interactions"""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def get_failed_tasks(self, minutes_back: int = 60) -> List[TaskInstance]:
        """Get failed tasks from the last N minutes"""
        query = """
        SELECT
            dag_id,
            task_id,
            run_id,
            execution_date,
            state,
            start_date,
            end_date
        FROM task_instance
        WHERE state = 'failed'
        AND start_date >= NOW() - INTERVAL %s MINUTE
        ORDER BY start_date DESC
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(query), (minutes_back,))  # type: ignore
            return [
                TaskInstance(
                    dag_id=row.dag_id,
                    task_id=row.task_id,
                    run_id=row.run_id,
                    execution_date=row.execution_date,
                    state=row.state,
                    start_date=row.start_date,
                    end_date=row.end_date,
                )
                for row in result
            ]

    def get_successful_tasks(
        self, dag_id: str, task_id: str, limit: int = 3
    ) -> List[TaskInstance]:
        """Get last N successful runs of a specific task"""
        query = """
        SELECT
            dag_id,
            task_id,
            run_id,
            execution_date,
            state,
            start_date,
            end_date
        FROM task_instance
        WHERE dag_id = %s
        AND task_id = %s
        AND state = 'success'
        ORDER BY end_date DESC
        LIMIT %s
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(query), (dag_id, task_id, limit))  # type: ignore
            return [
                TaskInstance(
                    dag_id=row.dag_id,
                    task_id=row.task_id,
                    run_id=row.run_id,
                    execution_date=row.execution_date,
                    state=row.state,
                    start_date=row.start_date,
                    end_date=row.end_date,
                )
                for row in result
            ]


class AirflowClient:
    """Combined client for Airflow API and MetaDB"""

    def __init__(self, base_url: str, username: str, password: str, db_connection: str):
        self.api_client = AirflowAPIClient(base_url, username, password)
        self.db_client = AirflowDBClient(db_connection)

    def get_task_logs(self, dag_id: str, task_id: str, run_id: str) -> str:
        return self.api_client.get_task_logs(dag_id, task_id, run_id)

    def get_failed_tasks(self, minutes_back: int = 60) -> List[TaskInstance]:
        return self.db_client.get_failed_tasks(minutes_back)

    def get_successful_tasks(
        self, dag_id: str, task_id: str, limit: int = 3
    ) -> List[TaskInstance]:
        return self.db_client.get_successful_tasks(dag_id, task_id, limit)
