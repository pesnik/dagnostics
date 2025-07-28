import logging
from typing import List

import requests
import urllib3
from pydantic import HttpUrl
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine, text

from dagnostics.core.models import TaskInstance

logger = logging.getLogger(__name__)


class AirflowAPIClient:
    """Client for Airflow REST API interactions"""

    def __init__(
        self, base_url: HttpUrl, username: str, password: str, verify_ssl: bool = True
    ):
        self.base_url = str(base_url).rstrip("/")
        self.auth = HTTPBasicAuth(username, password)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.verify = verify_ssl
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def get_task_logs(
        self, dag_id: str, task_id: str, run_id: str, try_number: int = 1
    ) -> str:
        """Fetch task logs from Airflow API"""
        url = f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}"
        logger.debug(f"Fetching logs from URL: {url}")

        response = None

        try:
            response = self.session.get(url)
            response.raise_for_status()

            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
                return data.get("content", "")
            else:
                return response.text

        except requests.exceptions.HTTPError as http_err:
            error_details = (
                f" - Response: {response.text}" if response is not None else ""
            )
            logger.error(
                f"HTTP error fetching logs for {dag_id}.{task_id} (run_id: {run_id}, try: {try_number}) from {url}: {http_err}{error_details}"
            )
            raise
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(
                f"Connection error fetching logs for {dag_id}.{task_id} (run_id: {run_id}, try: {try_number}) from {url}: {conn_err}"
            )
            raise
        except requests.exceptions.Timeout as timeout_err:
            logger.error(
                f"Timeout fetching logs for {dag_id}.{task_id} (run_id: {run_id}, try: {try_number}) from {url}: {timeout_err}"
            )
            raise
        except requests.RequestException as e:
            error_details = (
                f" - Response: {response.text}" if response is not None else ""
            )
            logger.error(
                f"""An unknown request error occurred fetching logs for
                    {dag_id}.{task_id} (run_id: {run_id}, try: {try_number}) from {url}: {e}{error_details}"""
            )
            raise


class AirflowDBClient:
    """Client for Airflow MetaDB interactions"""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def _execute_query(self, query: str, params: dict) -> List[TaskInstance]:
        """Helper to execute SQL queries and map results to TaskInstance."""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [
                TaskInstance(
                    dag_id=row.dag_id,
                    task_id=row.task_id,
                    run_id=row.run_id,
                    state=row.state,
                    start_date=row.start_date,
                    end_date=row.end_date,
                    try_number=row.try_number,
                )
                for row in result
            ]

    def get_failed_tasks(self, minutes_back: int = 60) -> List[TaskInstance]:
        """Get failed tasks from the last N minutes (PostgreSQL syntax assumed)."""
        query = """
        SELECT
            dag_id,
            task_id,
            run_id,
            execution_date,
            state,
            start_date,
            end_date,
            try_number
        FROM task_fail
        WHERE start_date >= NOW() - INTERVAL :minutes_back MINUTE
        ORDER BY start_date DESC
        """
        return self._execute_query(query, {"minutes_back": minutes_back})

    def get_successful_tasks(
        self, dag_id: str, task_id: str, limit: int = 3
    ) -> List[TaskInstance]:
        """Get last N successful runs of a specific task."""
        query = """
        SELECT
            dag_id,
            task_id,
            run_id,
            state,
            start_date,
            end_date,
            try_number
        FROM task_instance
        WHERE dag_id = :dag_id
        AND task_id = :task_id
        AND state = 'success'
        ORDER BY end_date DESC
        LIMIT :limit
        """
        return self._execute_query(
            query, {"dag_id": dag_id, "task_id": task_id, "limit": limit}
        )


class AirflowClient:
    """Combined client for Airflow API and MetaDB"""

    def __init__(
        self,
        base_url: HttpUrl,
        username: str,
        password: str,
        db_connection: str,
        verify_ssl: bool = True,
    ):
        logger.info(
            f"AirflowClient initialized. Base URL: {base_url}, Username: {username}, DB: {db_connection}"
        )
        self.api_client = AirflowAPIClient(
            base_url, username, password, verify_ssl=verify_ssl
        )
        self.db_client = AirflowDBClient(db_connection)

    def get_task_logs(
        self, dag_id: str, task_id: str, run_id: str, try_number: int = 1
    ) -> str:
        return self.api_client.get_task_logs(dag_id, task_id, run_id, try_number)

    def get_failed_tasks(self, minutes_back: int = 60) -> List[TaskInstance]:
        return self.db_client.get_failed_tasks(minutes_back)

    def get_successful_tasks(
        self, dag_id: str, task_id: str, limit: int = 3
    ) -> List[TaskInstance]:
        return self.db_client.get_successful_tasks(dag_id, task_id, limit)
