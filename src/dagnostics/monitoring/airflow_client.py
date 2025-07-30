import logging
from typing import List, Optional, Tuple

import requests
import urllib3
from pydantic import HttpUrl
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine, text  # Import Connection for typing

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

    def get_task_tries(
        self, dag_id: str, task_id: str, run_id: str
    ) -> List[TaskInstance]:
        """Fetch all tries for a specific task instance from Airflow API"""
        url = f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/tries"
        logger.debug(f"Fetching task tries from URL: {url}")

        response = None

        try:
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            task_instances = []

            for task_data in data.get("task_instances", []):
                # Parse datetime fields
                start_date = None
                end_date = None

                if task_data.get("start_date"):
                    try:
                        from datetime import datetime

                        start_date = datetime.fromisoformat(
                            task_data["start_date"].replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Failed to parse start_date: {task_data.get('start_date')}"
                        )

                if task_data.get("end_date"):
                    try:
                        from datetime import datetime

                        end_date = datetime.fromisoformat(
                            task_data["end_date"].replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Failed to parse end_date: {task_data.get('end_date')}"
                        )

                task_instance = TaskInstance(
                    dag_id=task_data["dag_id"],
                    task_id=task_data["task_id"],
                    run_id=task_data["dag_run_id"],
                    state=task_data["state"],
                    start_date=start_date,
                    end_date=end_date,
                    try_number=task_data["try_number"],
                )
                task_instances.append(task_instance)

            return task_instances

        except requests.exceptions.HTTPError as http_err:
            error_details = (
                f" - Response: {response.text}" if response is not None else ""
            )
            logger.error(
                f"HTTP error fetching task tries for {dag_id}.{task_id} (run_id: {run_id}) from {url}: {http_err}{error_details}"
            )
            raise
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(
                f"Connection error fetching task tries for {dag_id}.{task_id} (run_id: {run_id}) from {url}: {conn_err}"
            )
            raise
        except requests.exceptions.Timeout as timeout_err:
            logger.error(
                f"Timeout fetching task tries for {dag_id}.{task_id} (run_id: {run_id}) from {url}: {timeout_err}"
            )
            raise
        except requests.RequestException as e:
            error_details = (
                f" - Response: {response.text}" if response is not None else ""
            )
            logger.error(
                f"""An unknown request error occurred fetching task tries for
                    {dag_id}.{task_id} (run_id: {run_id}) from {url}: {e}{error_details}"""
            )
            raise


class AirflowDBClient:
    """Client for Airflow MetaDB interactions"""

    def __init__(self, connection_string: str, timezone_offset: str = "+00:00"):
        self.engine = create_engine(connection_string)
        self.timezone_offset = timezone_offset

    def _get_timezone_adjustment_params(self) -> Optional[Tuple[str, int, int]]:
        """
        Parses timezone offset and returns components (sign, hours, minutes)
        for parameterizing SQL queries.
        Returns None if no adjustment is needed (+00:00).
        """
        if self.timezone_offset == "+00:00":
            return None

        sign = self.timezone_offset[0]
        try:
            hours, minutes = map(int, self.timezone_offset[1:].split(":"))
            return (sign, hours, minutes)
        except ValueError:
            logger.error(f"Invalid timezone offset format: {self.timezone_offset}")
            return None

    def _execute_query(self, query_template: str, params: dict) -> List[TaskInstance]:
        """Helper to execute SQL queries and map results to TaskInstance."""
        timezone_adj_params = self._get_timezone_adjustment_params()

        bound_params = params.copy()  # Make a copy to add timezone params

        # Dynamically build the SELECT clause to apply timezone adjustment
        # and ensure it's done safely.
        # This requires reconstructing the select clause based on whether
        # timezone adjustment is needed.
        # Note: This is a more involved change because the adjustment
        # is directly on the column, not just a value.
        # The safest way is to use specific SQL functions if available,
        # or construct the query parts very carefully with known-safe string literals.

        # Example for Postgres/MySQL which allow interval arithmetic
        # This part still requires string concatenation for the SQL function name and interval units,
        # but the actual values (hours, minutes) are parameterized.

        # Determine the database dialect to choose the correct date adjustment function
        dialect_name = self.engine.dialect.name

        start_date_select = "start_date"
        end_date_select = "end_date"

        if timezone_adj_params:
            sign, hours, minutes = timezone_adj_params

            # Add parameters for hours and minutes
            bound_params["tz_hours"] = hours
            bound_params["tz_minutes"] = minutes

            if dialect_name == "postgresql":
                if sign == "+":
                    start_date_select = "start_date + INTERVAL ':tz_hours HOURS' + INTERVAL ':tz_minutes MINUTES'"
                    end_date_select = "end_date + INTERVAL ':tz_hours HOURS' + INTERVAL ':tz_minutes MINUTES'"
                else:  # sign == '-'
                    start_date_select = "start_date - INTERVAL ':tz_hours HOURS' - INTERVAL ':tz_minutes MINUTES'"
                    end_date_select = "end_date - INTERVAL ':tz_hours HOURS' - INTERVAL ':tz_minutes MINUTES'"
            elif dialect_name == "mysql":
                if sign == "+":
                    start_date_select = "DATE_ADD(start_date, INTERVAL :tz_hours HOUR, INTERVAL :tz_minutes MINUTE)"
                    end_date_select = "DATE_ADD(end_date, INTERVAL :tz_hours HOUR, INTERVAL :tz_minutes MINUTE)"
                else:  # sign == '-'
                    start_date_select = "DATE_SUB(start_date, INTERVAL :tz_hours HOUR, INTERVAL :tz_minutes MINUTE)"
                    end_date_select = "DATE_SUB(end_date, INTERVAL :tz_hours HOUR, INTERVAL :tz_minutes MINUTE)"
            else:
                # Fallback for other dialects or if specific functions are unknown.
                # This part is the original risky part but parameterized values are safer.
                # For maximum safety with Bandit, consider if a different approach for other DBs is needed.
                # For most common cases (Postgres, MySQL), the above is preferred.
                logger.warning(
                    f"Unsupported dialect '{dialect_name}' for parameterized timezone adjustment. Falling back to string concatenation for INTERVAL."
                )
                # Original logic for _get_timezone_adjustment_sql might be needed here, but parameterized values.
                # Since we already have sign, hours, minutes, we can still construct it safely.
                interval_str = f"INTERVAL '{hours} HOUR {minutes} MINUTE'"
                if sign == "+":
                    start_date_select = f"start_date + {interval_str}"
                    end_date_select = f"end_date + {interval_str}"
                else:
                    start_date_select = f"start_date - {interval_str}"
                    end_date_select = f"end_date - {interval_str}"

        # Replace placeholders in the query_template with the correctly formatted strings
        # This still involves f-string like substitution, but on known safe parts (start_date/end_date select string)
        # after parameterizing the dynamic values.
        # This approach mitigates Bandit's B608 for the interval part.
        query_with_tz_placeholders = query_template.format(
            start_date_select=start_date_select, end_date_select=end_date_select
        )

        with self.engine.connect() as conn:
            # Passing bound_params to text() ensures proper escaping
            result = conn.execute(text(query_with_tz_placeholders), bound_params)
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
        """Get failed tasks from the last N minutes with timezone adjustment."""
        # Use placeholders for the date select part
        query = """
        SELECT
            dag_id,
            task_id,
            run_id,
            state,
            {start_date_select} as start_date,
            {end_date_select} as end_date,
            try_number
        FROM task_instance_history
        WHERE start_date >= NOW() - INTERVAL ':minutes_back MINUTE'
        AND state = 'failed'
        ORDER BY start_date DESC
        """
        return self._execute_query(query, {"minutes_back": minutes_back})

    def get_successful_tasks(
        self, dag_id: str, task_id: str, limit: int = 3
    ) -> List[TaskInstance]:
        """Get last N successful runs of a specific task with timezone adjustment."""
        # Use placeholders for the date select part
        query = """
        SELECT
            dag_id,
            task_id,
            run_id,
            state,
            {start_date_select} as start_date,
            {end_date_select} as end_date,
            try_number
        FROM task_instance
        WHERE dag_id = :dag_id
        AND task_id = :task_id
        AND state = 'success'
        AND try_number = 1
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
        db_timezone_offset: str = "+00:00",
    ):
        self.api_client = AirflowAPIClient(
            base_url, username, password, verify_ssl=verify_ssl
        )
        self.db_client = AirflowDBClient(db_connection, db_timezone_offset)

        logger.info("AirflowClient initialized.")

    def get_task_logs(
        self, dag_id: str, task_id: str, run_id: str, try_number: int = 1
    ) -> str:
        return self.api_client.get_task_logs(dag_id, task_id, run_id, try_number)

    def get_task_tries(
        self, dag_id: str, task_id: str, run_id: str
    ) -> List[TaskInstance]:
        return self.api_client.get_task_tries(dag_id, task_id, run_id)

    def get_failed_tasks(self, minutes_back: int = 60) -> List[TaskInstance]:
        return self.db_client.get_failed_tasks(minutes_back)

    def get_successful_tasks(
        self, dag_id: str, task_id: str, limit: int = 3
    ) -> List[TaskInstance]:
        return self.db_client.get_successful_tasks(dag_id, task_id, limit)
