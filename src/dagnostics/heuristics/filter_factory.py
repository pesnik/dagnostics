"""
Filter factory for creating configured ErrorPatternFilter instances.
Centralizes filter creation logic for different use cases.
"""

import logging
import re

from dagnostics.core.models import AppConfig, LogEntry
from dagnostics.heuristics.pattern_filter import ErrorPatternFilter

logger = logging.getLogger(__name__)


class FilterFactory:
    """Factory for creating configured filter instances"""

    @staticmethod
    def create_for_notifications(config: AppConfig) -> ErrorPatternFilter:
        """
        Create a filter optimized for notification workflows

        Args:
            config: Application configuration

        Returns:
            Configured ErrorPatternFilter for notifications
        """
        # Use the pattern filtering config path
        filter_instance = ErrorPatternFilter(config.pattern_filtering.config_path)

        # Add notification-specific custom functions
        def is_notification_spam(log_entry: LogEntry) -> bool:
            """Filter out messages that would create notification spam"""
            spam_patterns = [
                "Notification sent successfully",
                "SMS delivered",
                "Email queued",
                "Alert triggered for the same error",
                "Rate limit applied",
                "Duplicate alert suppressed",
            ]
            return any(pattern in log_entry.message for pattern in spam_patterns)

        def is_config_noise(log_entry: LogEntry) -> bool:
            """Filter out configuration-related noise"""
            config_patterns = [
                f"timezone offset {config.airflow.db_timezone_offset}",
                f"Using {config.llm.default_provider} provider",
                "Configuration loaded",
                "SSL verification disabled" if not config.airflow.verify_ssl else None,
            ]
            return any(
                pattern and pattern in log_entry.message for pattern in config_patterns
            )

        def standard_error_log_except_start_with_caused_by(log_entry: LogEntry) -> bool:
            message = log_entry.message

            if "Caused by" in message:
                return False

            return not (
                message.startswith(("[", "$")) or bool(re.search(r"TPT\d+", message))
            )

        # Register custom functions
        filter_instance.add_custom_function(
            "notification_spam",
            is_notification_spam,
            "Messages that create notification spam",
        )
        filter_instance.add_custom_function(
            "config_noise", is_config_noise, "Configuration-related noise"
        )
        filter_instance.add_custom_function(
            "standard_error_log",
            standard_error_log_except_start_with_caused_by,
            "Standard Error Log",
        )

        # Add runtime filters for notification context
        FilterFactory._add_notification_runtime_filters(filter_instance, config)

        logger.info(
            f"Created notification filter with {len(filter_instance.filter_rules)} rules"
        )
        return filter_instance

    @staticmethod
    def create_for_analysis(config: AppConfig) -> ErrorPatternFilter:
        """
        Create a filter optimized for detailed analysis workflows

        Args:
            config: Application configuration

        Returns:
            Configured ErrorPatternFilter for analysis
        """
        filter_instance = ErrorPatternFilter(config.pattern_filtering.config_path)

        # Analysis might want to see more details, so less aggressive filtering
        # Add analysis-specific customizations here

        logger.info(
            f"Created analysis filter with {len(filter_instance.filter_rules)} rules"
        )
        return filter_instance

    @staticmethod
    def create_for_monitoring(config: AppConfig) -> ErrorPatternFilter:
        """
        Create a filter optimized for continuous monitoring

        Args:
            config: Application configuration

        Returns:
            Configured ErrorPatternFilter for monitoring
        """
        filter_instance = ErrorPatternFilter(config.pattern_filtering.config_path)

        # Monitoring might want different sensitivity
        # Add monitoring-specific customizations here

        logger.info(
            f"Created monitoring filter with {len(filter_instance.filter_rules)} rules"
        )
        return filter_instance

    @staticmethod
    def _add_notification_runtime_filters(
        filter_instance: ErrorPatternFilter, config: AppConfig
    ):
        """Add runtime filters specific to notification context"""
        from dagnostics.heuristics.pattern_filter import FilterRuleType

        # More aggressive filtering for notifications to reduce noise
        if config.api.log_level.upper() != "DEBUG":
            filter_instance.add_filter_rule(
                ".*\\b(DEBUG|debug)\\b.*",
                FilterRuleType.REGEX,
                "Debug messages (not in debug mode)",
            )

        # Filter out feature-disabled messages
        if not config.alerts.sms.enabled:
            filter_instance.add_filter_rule(
                "SMS.*disabled", FilterRuleType.REGEX, "SMS disabled messages"
            )
