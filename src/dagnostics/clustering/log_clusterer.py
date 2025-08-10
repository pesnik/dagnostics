import logging
import os
import pickle  # nosec B403 # Data is from trusted internal sources only
from datetime import datetime
from typing import Dict, List, Optional

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from dagnostics.core.models import AppConfig, DrainCluster, LogEntry

logger = logging.getLogger(__name__)


class LogClusterer:
    """Drain3-based log clustering for baseline creation and anomaly detection"""

    def __init__(
        self,
        app_config: AppConfig,
        config_path: Optional[str] = None,
        persistence_path: Optional[str] = None,
    ):
        self.app_config = app_config
        self.config = TemplateMinerConfig()
        if config_path:
            self.config.load(config_path)

        self.persistence_path = persistence_path
        self.baseline_timestamps: Dict[str, datetime] = (
            {}
        )  # Track baseline creation times
        self.baseline_drains_cache: Dict[str, TemplateMiner] = (
            {}
        )  # In-memory cache for real-time mode
        self.load_baseline_state()

    def build_baseline_clusters(
        self, successful_logs: List[LogEntry], dag_id: str, task_id: str
    ) -> Dict[str, DrainCluster]:
        """Build baseline clusters from successful task logs"""
        logger.info(
            f"Building baseline for {dag_id}.{task_id} with {len(successful_logs)} logs"
        )

        baseline_key = f"{dag_id}.{task_id}"

        # Create baseline drain instance
        baseline_drain = self._create_baseline_drain_instance(dag_id, task_id)
        if not baseline_drain:
            return {}

        # Ingest all successful logs into baseline
        for log_entry in successful_logs:
            _ = baseline_drain.add_log_message(log_entry.message)
            # print(result)

        # Update baseline timestamp
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

    def identify_anomalous_patterns(
        self, failed_logs: List[LogEntry], dag_id: str, task_id: str
    ) -> List[LogEntry]:
        """
        Identify log entries that represent new/anomalous patterns.
        Uses temporary drain instances to avoid polluting the baseline.
        """
        baseline_key = f"{dag_id}.{task_id}"

        if baseline_key not in self.baseline_timestamps:
            logger.warning(
                f"No baseline found for {baseline_key}. All {len(failed_logs)} logs will be considered anomalous."
            )
            return failed_logs

        # Create temporary drain instance for analysis
        temp_drain = self._create_temp_drain_instance()
        if not temp_drain:
            logger.error("Failed to create temporary drain instance")
            return failed_logs

        # Load baseline drain instance (read-only)
        baseline_drain = self._load_baseline_drain_instance(dag_id, task_id)
        if not baseline_drain:
            logger.warning(
                f"Could not load baseline drain for {baseline_key}. Treating all logs as anomalous."
            )
            return failed_logs

        # First, train temporary drain with baseline patterns
        # This preserves the baseline without modifying it
        for cluster in baseline_drain.drain.clusters:
            # Add one representative message from each baseline cluster to temp drain
            if cluster.log_template_tokens:
                # Reconstruct a message from the template (approximate)
                template_message = " ".join(cluster.log_template_tokens)
                temp_drain.add_log_message(template_message)

        anomalous_logs = []

        # Now test failed logs against the temp drain (which has baseline knowledge)
        for log_entry in failed_logs:
            result = temp_drain.add_log_message(log_entry.message)

            # Check if this created a new cluster (anomalous pattern)
            if result["change_type"] == "cluster_created":
                anomalous_logs.append(log_entry)
                logger.debug(
                    f"Anomalous pattern detected: '{result['template_mined']}'"
                )
            else:
                logger.debug(f"Log matches known pattern: '{result['template_mined']}'")

        logger.info(
            f"Found {len(anomalous_logs)} anomalous patterns out of {len(failed_logs)} logs for {baseline_key}"
        )
        return anomalous_logs

    def _create_baseline_drain_instance(
        self, dag_id: str, task_id: str
    ) -> Optional[TemplateMiner]:
        """Create a drain instance for baseline storage"""
        baseline_key = f"{dag_id}.{task_id}"

        if (
            self.app_config.monitoring.baseline_usage == "stored"
            and self.persistence_path
        ):
            baseline_drain_path = f"{self.persistence_path}.baseline.{dag_id}.{task_id}"
            os.makedirs(os.path.dirname(baseline_drain_path), exist_ok=True)

            try:
                persistence_handler = FilePersistence(baseline_drain_path)
                baseline_drain = TemplateMiner(
                    persistence_handler=persistence_handler, config=self.config
                )
                return baseline_drain
            except Exception as e:
                logger.error(f"Failed to create persistent baseline drain: {e}")
                return None
        else:
            # Real-time mode - create and cache in memory
            try:
                baseline_drain = TemplateMiner(config=self.config)
                self.baseline_drains_cache[baseline_key] = baseline_drain
                return baseline_drain
            except Exception as e:
                logger.error(f"Failed to create in-memory baseline drain: {e}")
                return None

    def _load_baseline_drain_instance(
        self, dag_id: str, task_id: str
    ) -> Optional[TemplateMiner]:
        """Load existing baseline drain instance for read-only operations"""
        baseline_key = f"{dag_id}.{task_id}"

        if (
            self.app_config.monitoring.baseline_usage == "stored"
            and self.persistence_path
        ):
            baseline_drain_path = f"{self.persistence_path}.baseline.{dag_id}.{task_id}"

            if os.path.exists(baseline_drain_path):
                try:
                    persistence_handler = FilePersistence(baseline_drain_path)
                    return TemplateMiner(
                        persistence_handler=persistence_handler, config=self.config
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load baseline drain from {baseline_drain_path}: {e}"
                    )
                    return None
            else:
                logger.warning(f"Baseline drain file not found: {baseline_drain_path}")
                return None
        else:
            # Real-time mode - load from memory cache
            if baseline_key in self.baseline_drains_cache:
                return self.baseline_drains_cache[baseline_key]
            else:
                logger.warning(f"No baseline drain found in cache for {baseline_key}")
                return None

    def _create_temp_drain_instance(self) -> Optional[TemplateMiner]:
        """Create a temporary drain instance for analysis (no persistence)"""
        try:
            return TemplateMiner(config=self.config)
        except Exception as e:
            logger.error(f"Failed to create temporary drain instance: {e}")
            return None

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

    def baseline_cluster_size(self, dag_id: str, task_id: str) -> int:
        """Get the number of clusters in the baseline for a given dag/task"""
        baseline_key = f"{dag_id}.{task_id}"

        try:
            if (
                self.app_config.monitoring.baseline_usage == "stored"
                and self.persistence_path
            ):
                # For stored mode, load the baseline drain instance
                baseline_drain = self._load_baseline_drain_instance(dag_id, task_id)
                if baseline_drain and baseline_drain.drain.clusters:
                    cluster_count = len(baseline_drain.drain.clusters)
                    logger.debug(
                        f"Baseline cluster count for {baseline_key}: {cluster_count}"
                    )
                    return cluster_count
                else:
                    logger.debug(
                        f"No baseline clusters found for {baseline_key} in stored mode"
                    )
                    return 0
            else:
                # For real-time mode, check the in-memory cache
                if baseline_key in self.baseline_drains_cache:
                    baseline_drain = self.baseline_drains_cache[baseline_key]
                    if baseline_drain and baseline_drain.drain.clusters:
                        cluster_count = len(baseline_drain.drain.clusters)
                        logger.debug(
                            f"Baseline cluster count for {baseline_key}: {cluster_count}"
                        )
                        return cluster_count
                    else:
                        logger.debug(
                            f"No baseline clusters found for {baseline_key} in cache"
                        )
                        return 0
                else:
                    logger.debug(f"No baseline found in cache for {baseline_key}")
                    return 0

        except Exception as e:
            logger.error(f"Failed to get baseline cluster size for {baseline_key}: {e}")
            return 0

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

    def save_baseline_state(self):
        """Persist baseline timestamps (clusters are persisted via Drain3 FilePersistence)"""
        if self.persistence_path:
            baseline_state_path = f"{self.persistence_path}.baseline_clusters_state.pkl"
            os.makedirs(os.path.dirname(baseline_state_path), exist_ok=True)
            try:
                state = {
                    "baseline_timestamps": {
                        k: v.isoformat() for k, v in self.baseline_timestamps.items()
                    },
                }
                with open(baseline_state_path, "wb") as f:
                    pickle.dump(state, f)
                logger.debug(f"Saved baseline state to {baseline_state_path}")
            except Exception as e:
                logger.error(f"Failed to save baseline state: {e}")

    def load_baseline_state(self):
        """Load baseline timestamps"""
        if self.persistence_path:
            baseline_state_path = f"{self.persistence_path}.baseline_clusters_state.pkl"
            if os.path.exists(baseline_state_path):
                try:
                    with open(baseline_state_path, "rb") as f:
                        state = pickle.load(
                            f
                        )  # nosec B301 # Data is from trusted internal sources only

                    if "baseline_timestamps" in state:
                        self.baseline_timestamps = {
                            k: datetime.fromisoformat(v)
                            for k, v in state["baseline_timestamps"].items()
                        }
                    logger.info(f"Loaded baseline state from {baseline_state_path}")
                except Exception as e:
                    logger.error(f"Failed to load baseline state: {e}")

    def cleanup_old_baselines(self, retention_days: int = 30):
        """Clean up old baseline files to manage disk space"""
        if not self.persistence_path:
            return

        try:
            base_dir = os.path.dirname(self.persistence_path)
            current_time = datetime.now()

            for filename in os.listdir(base_dir):
                if filename.startswith(
                    os.path.basename(self.persistence_path) + ".baseline."
                ):
                    filepath = os.path.join(base_dir, filename)

                    # Check file age
                    file_age = current_time - datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    )

                    if file_age.days > retention_days:
                        os.remove(filepath)
                        logger.info(f"Removed old baseline file: {filename}")

        except Exception as e:
            logger.error(f"Failed to cleanup old baselines: {e}")

    def clear_memory_cache(self):
        """Clear in-memory baseline cache (useful for testing or memory management)"""
        self.baseline_drains_cache.clear()
        logger.info("Cleared baseline drains memory cache")

    def get_cache_stats(self) -> dict:
        """Get statistics about the in-memory cache"""
        return {
            "cached_baselines": len(self.baseline_drains_cache),
            "baseline_keys": list(self.baseline_drains_cache.keys()),
            "cache_mode": (
                "real-time"
                if self.app_config.monitoring.baseline_usage != "stored"
                else "stored"
            ),
        }
