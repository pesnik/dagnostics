import logging
import os
import pickle  # nosec B403 # Data is from trusted internal sources only
from datetime import datetime
from typing import Dict, List, Optional, Set

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from dagnostics.core.models import DrainCluster, LogEntry

logger = logging.getLogger(__name__)


class LogClusterer:
    """Drain3-based log clustering for baseline creation and anomaly detection"""

    def __init__(
        self, config_path: Optional[str] = None, persistence_path: Optional[str] = None
    ):
        self.config = TemplateMinerConfig()
        if config_path:
            self.config.load(config_path)

        persistence_handler = None
        if persistence_path:
            persistence_handler = FilePersistence(persistence_path)

        # Create TemplateMiner with proper config and persistence
        # Pass persistence_handler only if it's not None
        if persistence_handler:
            self.drain = TemplateMiner(
                persistence_handler=persistence_handler, config=self.config
            )
        else:
            self.drain = TemplateMiner(config=self.config)

        self.persistence_path = persistence_path
        self.baseline_clusters: Dict[str, Set[str]] = {}
        self.baseline_timestamps: Dict[str, datetime] = (
            {}
        )  # Track baseline creation times
        self.load_baseline_state()

    def build_baseline_clusters(
        self, successful_logs: List[LogEntry], dag_id: str, task_id: str
    ) -> Dict[str, DrainCluster]:
        """Build baseline clusters from successful task logs"""
        logger.info(
            f"Building baseline for {dag_id}.{task_id} with {len(successful_logs)} logs"
        )

        # Create a separate drain instance for this baseline
        baseline_persistence_handler = None
        if self.persistence_path:
            baseline_path = f"{self.persistence_path}.baseline.{dag_id}.{task_id}"
            baseline_persistence_handler = FilePersistence(baseline_path)

        # Pass baseline_persistence_handler only if it's not None
        if baseline_persistence_handler:
            baseline_drain = TemplateMiner(
                persistence_handler=baseline_persistence_handler, config=self.config
            )
        else:
            baseline_drain = TemplateMiner(config=self.config)

        baseline_key = f"{dag_id}.{task_id}"
        cluster_templates = set()

        for log_entry in successful_logs:
            result = baseline_drain.add_log_message(log_entry.message)
            if result["change_type"] != "none":
                cluster_templates.add(result["template_mined"])

        # Store baseline templates
        self.baseline_clusters[baseline_key] = cluster_templates
        self.baseline_timestamps[baseline_key] = datetime.now()

        # Convert to DrainCluster objects
        clusters = {}
        for i, cluster in enumerate(baseline_drain.drain.clusters):
            cluster_id = f"{baseline_key}_cluster_{i}"
            clusters[cluster_id] = DrainCluster(
                cluster_id=cluster_id,
                template=cluster.get_template(),
                log_ids=[],  # We don't store individual log IDs for baselines
                size=cluster.size,
                created_at=datetime.now(),
                last_updated=datetime.now(),
            )

        self.save_baseline_state()
        logger.info(f"Created {len(clusters)} baseline clusters for {baseline_key}")
        return clusters

    def is_baseline_stale(self, dag_id: str, task_id: str, refresh_days: int) -> bool:
        """Check if baseline is stale and needs refresh"""
        baseline_key = f"{dag_id}.{task_id}"

        if baseline_key not in self.baseline_timestamps:
            return True  # No baseline exists, needs creation

        baseline_age = datetime.now() - self.baseline_timestamps[baseline_key]
        baseline_age_days = baseline_age.days

        return baseline_age_days >= refresh_days

    def get_baseline_age_days(self, dag_id: str, task_id: str) -> int:
        """Get the age of baseline in days"""
        baseline_key = f"{dag_id}.{task_id}"

        if baseline_key not in self.baseline_timestamps:
            return 0  # No baseline exists

        baseline_age = datetime.now() - self.baseline_timestamps[baseline_key]
        return baseline_age.days

    def refresh_baseline_if_needed(
        self,
        dag_id: str,
        task_id: str,
        refresh_days: int,
        airflow_client,
        successful_logs: List[LogEntry],
    ) -> bool:
        """Refresh baseline if it's stale"""
        if self.is_baseline_stale(dag_id, task_id, refresh_days):
            logger.info(f"Refreshing stale baseline for {dag_id}.{task_id}")
            self.build_baseline_clusters(successful_logs, dag_id, task_id)
            return True
        return False

    def identify_anomalous_patterns(
        self, failed_logs: List[LogEntry], dag_id: str, task_id: str
    ) -> List[LogEntry]:
        """Identify log entries that don't match baseline patterns"""
        baseline_key = f"{dag_id}.{task_id}"
        baseline_templates = self.baseline_clusters.get(baseline_key, set())

        if not baseline_templates:
            logger.warning(f"No baseline found for {baseline_key}, returning all logs")
            return failed_logs

        anomalous_logs = []

        for log_entry in failed_logs:
            # Test against baseline patterns
            result = self.drain.add_log_message(log_entry.message)
            # template = result["template_mined"]

            # Check if this template is similar to any baseline template
            is_baseline_pattern = result["change_type"] == "cluster_created"
            # is_baseline_pattern = any(
            #     self._calculate_template_similarity(template, baseline_template) > 0.8
            #     for baseline_template in baseline_templates
            # )

            if is_baseline_pattern:
                anomalous_logs.append(log_entry)

        logger.info(
            f"Found {len(anomalous_logs)} anomalous patterns out of {len(failed_logs)} logs"
        )
        return anomalous_logs

    def _calculate_template_similarity(self, template1: str, template2: str) -> float:
        """Calculate similarity between two templates"""
        # Handle None values
        if template1 is None:
            template1 = ""
        if template2 is None:
            template2 = ""

        tokens1 = set(template1.split())
        tokens2 = set(template2.split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    def save_baseline_state(self):
        """Persist baseline clusters state"""
        if self.persistence_path:
            baseline_state_path = f"{self.persistence_path}.baseline_clusters"
            os.makedirs(os.path.dirname(baseline_state_path), exist_ok=True)
            try:
                state = {
                    "baseline_clusters": {
                        k: list(v) for k, v in self.baseline_clusters.items()
                    },
                    "baseline_timestamps": {
                        k: v.isoformat() for k, v in self.baseline_timestamps.items()
                    },
                }
                with open(baseline_state_path, "wb") as f:
                    pickle.dump(state, f)
                logger.debug("Saved baseline clusters state")
            except Exception as e:
                logger.error(f"Failed to save baseline clusters state: {e}")

    def load_baseline_state(self):
        """Load baseline clusters state"""
        if self.persistence_path:
            baseline_state_path = f"{self.persistence_path}.baseline_clusters"
            if os.path.exists(baseline_state_path):
                try:
                    with open(baseline_state_path, "rb") as f:
                        state = pickle.load(
                            f
                        )  # nosec B301 # Data is from trusted internal sources only

                    if "baseline_clusters" in state:
                        self.baseline_clusters = {
                            k: set(v) for k, v in state["baseline_clusters"].items()
                        }
                    if "baseline_timestamps" in state:
                        self.baseline_timestamps = {
                            k: datetime.fromisoformat(v)
                            for k, v in state["baseline_timestamps"].items()
                        }
                        logger.info("Loaded baseline clusters state from persistence")
                except Exception as e:
                    logger.error(f"Failed to load baseline clusters state: {e}")
