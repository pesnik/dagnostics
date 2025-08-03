import logging
from typing import Optional

from dagnostics.core.config import load_config
from dagnostics.core.models import AppConfig, LogEntry
from dagnostics.heuristics.pattern_filter import ErrorPatternFilter, FilterRuleType

logger = logging.getLogger(__name__)


class IntegratedErrorPatternFilter(ErrorPatternFilter):
    """
    Extended ErrorPatternFilter that integrates with your existing AppConfig structure
    """

    def __init__(
        self, app_config: Optional[AppConfig] = None, config_path: Optional[str] = None
    ):
        """
        Initialize filter with app configuration

        Args:
            app_config: The main application configuration
            config_path: Override path to filter patterns config
        """
        self.app_config = app_config

        # Determine config path from app config if not provided
        if config_path is None and app_config:
            config_path = app_config.pattern_filtering.config_path

        # Initialize parent class
        super().__init__(config_path)

        # Register additional custom functions based on app config
        self._register_app_specific_functions()

    def _register_app_specific_functions(self):
        """Register custom functions based on your application's needs"""
        if not self.app_config:
            return

        # Type check to satisfy Pylance
        app_config = self.app_config
        if app_config is None:
            return

        # Add timezone-aware filtering based on your DB offset
        def is_timezone_info_line(log_entry: LogEntry) -> bool:
            """Filter out timezone conversion messages"""
            if app_config is None:
                return False
            tz_patterns = [
                f"timezone offset {app_config.airflow.db_timezone_offset}",
                "Converting timezone",
                "UTC offset applied",
            ]
            return any(pattern in log_entry.message for pattern in tz_patterns)

        # Add Airflow-specific filtering based on your config
        def is_airflow_config_message(log_entry: LogEntry) -> bool:
            """Filter out Airflow configuration messages"""
            if app_config is None:
                return False
            config_patterns = [
                f"Connecting to {app_config.airflow.base_url}",
                f"Using timeout {app_config.airflow.timeout}",
                (
                    "SSL verification disabled"
                    if not app_config.airflow.verify_ssl
                    else "SSL verification enabled"
                ),
                "Configuration loaded successfully",
            ]
            return any(pattern in log_entry.message for pattern in config_patterns)

        # Add LLM provider specific filtering
        def is_llm_info_message(log_entry: LogEntry) -> bool:
            """Filter out LLM provider info messages"""
            if app_config is None:
                return False
            llm_patterns = [
                f"Using {app_config.llm.default_provider} as default provider",
                "Model loaded successfully",
                "LLM engine initialized",
                "Temperature set to",
            ]
            return any(pattern in log_entry.message for pattern in llm_patterns)

        # Add drain3 clustering info
        def is_clustering_info_message(log_entry: LogEntry) -> bool:
            """Filter out clustering algorithm messages"""
            if app_config is None:
                return False
            clustering_patterns = [
                f"Drain3 depth: {app_config.drain3.depth}",
                f"Similarity threshold: {app_config.drain3.sim_th}",
                "Cluster created",
                "Template matched",
                "Persistence saved",
            ]
            return any(pattern in log_entry.message for pattern in clustering_patterns)

        # Register all custom functions
        self.add_custom_function(
            "timezone_info_line", is_timezone_info_line, "Timezone conversion messages"
        )
        self.add_custom_function(
            "airflow_config_message",
            is_airflow_config_message,
            "Airflow configuration messages",
        )
        self.add_custom_function(
            "llm_info_message", is_llm_info_message, "LLM provider information messages"
        )
        self.add_custom_function(
            "clustering_info_message",
            is_clustering_info_message,
            "Drain3 clustering information",
        )

    def add_runtime_filters_from_config(self):
        """Add runtime filters based on current configuration state"""
        if not self.app_config:
            return

        # Type check to satisfy Pylance
        app_config = self.app_config
        if app_config is None:
            return

        # Add filters based on enabled features
        if not app_config.alerts.sms.enabled:
            self.add_filter_rule(
                "SMS alert",
                FilterRuleType.SUBSTRING,
                "SMS alerts disabled",
                negate=True,
            )

        if not app_config.alerts.email.enabled:
            self.add_filter_rule(
                "Email alert",
                FilterRuleType.SUBSTRING,
                "Email alerts disabled",
                negate=True,
            )

        # Add debug-level filtering based on API log level
        if app_config.api.log_level.upper() != "DEBUG":
            self.add_filter_rule(
                ".*DEBUG.*",
                FilterRuleType.REGEX,
                "Debug messages (API not in debug mode)",
            )

    @classmethod
    def from_app_config(
        cls, config_path: str = "config/config.yaml"
    ) -> "IntegratedErrorPatternFilter":
        """
        Factory method to create filter from app config file

        Args:
            config_path: Path to main application config

        Returns:
            Configured IntegratedErrorPatternFilter instance
        """
        app_config = load_config(config_path)
        return cls(app_config)


def create_filter_for_notify_failures(
    config: AppConfig,
) -> IntegratedErrorPatternFilter:
    """
    Create a properly configured filter for the notify_failures function
    """
    # Create integrated filter
    filter_instance = IntegratedErrorPatternFilter(config)

    # Add runtime filters based on current state
    filter_instance.add_runtime_filters_from_config()

    # Add notification-specific filters
    def is_notification_spam(log_entry: LogEntry) -> bool:
        """Filter out messages that would create notification spam"""
        spam_patterns = [
            "Notification sent successfully",
            "SMS delivered",
            "Email queued",
            "Alert triggered for the same error",
            "Rate limit applied",
        ]
        return any(pattern in log_entry.message for pattern in spam_patterns)

    filter_instance.add_custom_function(
        "notification_spam",
        is_notification_spam,
        "Messages that would create notification spam",
    )

    logger.info(f"Created filter with {len(filter_instance.filter_rules)} rules")
    logger.info(f"Filter statistics: {filter_instance.get_filter_stats()}")

    return filter_instance


def get_enhanced_filter(config: AppConfig) -> IntegratedErrorPatternFilter:
    """
    Replace the simple ErrorPatternFilter() call in notify_failures with this
    """
    return create_filter_for_notify_failures(config)
