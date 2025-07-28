import logging
import re
from re import Pattern
from typing import List, Optional

import yaml

from dagnostics.core.models import LogEntry

logger = logging.getLogger(__name__)


class ErrorPatternFilter:
    """Filter out known non-error patterns using regex and rules"""

    def __init__(self, config_path: Optional[str] = None):
        self.regex_filters: List[Pattern[str]] = []
        self.load_patterns(config_path)

    def load_patterns(self, config_path: Optional[str] = None):
        """Load filtering patterns from configuration"""
        default_patterns = [
            r".*INFO.*",
            r".*DEBUG.*",
            r".*Starting.*",
            r".*Finished.*",
            r".*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*",  # Timestamp lines
            r".*={3,}.*",  # Separator lines
            r".*-{3,}.*",  # Separator lines
            # Airflow specific non-error patterns
            r".*Marking task as SUCCESS.*",
            r".*Task exited with return code 0.*",
            r".*Dependencies all met for.*",
            r".*Airflow task starting.*",
            # Common non-critical warnings
            r".*UserWarning.*",
            r".*DeprecationWarning.*",
            r".*PendingDeprecationWarning.*",
        ]

        custom_patterns: List[str] = []
        if config_path:
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    custom_patterns = config.get("filter_patterns", []) or []
                    if not isinstance(custom_patterns, list):
                        logger.warning(
                            f"Expected list for 'filter_patterns' in {config_path}, but got {type(custom_patterns)}. Ignoring."
                        )
                        custom_patterns = []

            except Exception as e:
                logger.warning(f"Could not load custom patterns: {e}")

        all_patterns = default_patterns + custom_patterns
        self.regex_filters = [
            re.compile(pattern, re.IGNORECASE) for pattern in all_patterns
        ]

        logger.info(f"Loaded {len(self.regex_filters)} filter patterns")

    def filter_candidates(self, anomalous_logs: List[LogEntry]) -> List[LogEntry]:
        """Filter out non-error log entries"""
        filtered_logs: List[LogEntry] = []

        for log_entry in anomalous_logs:
            if self._is_error_candidate(log_entry):
                filtered_logs.append(log_entry)

        logger.info(
            f"Filtered {len(anomalous_logs)} logs down to {len(filtered_logs)} candidates"
        )
        return filtered_logs

    def _is_error_candidate(self, log_entry: LogEntry) -> bool:
        """Check if log entry is likely an error"""
        message = log_entry.message

        # Skip empty or very short messages
        if not message or len(message.strip()) < 10:
            return False

        # Skip if matches any filter pattern
        for pattern in self.regex_filters:
            if pattern.search(message):
                return False

        # Positive indicators for errors
        error_indicators = [
            "error",
            "exception",
            "failed",
            "failure",
            "fatal",
            "critical",
            "traceback",
            "stack trace",
            "connection refused",
            "timeout",
            "permission denied",
            "not found",
            "invalid",
            "corrupt",
        ]

        message_lower = message.lower()
        has_error_indicator = any(
            indicator in message_lower for indicator in error_indicators
        )

        # Include if it has error indicators or is a WARNING/ERROR level
        return has_error_indicator or log_entry.level.upper() in [
            "ERROR",
            "CRITICAL",
            "FATAL",
        ]

    def add_custom_filter(self, pattern: str, pattern_type: str = "regex"):
        """Add a custom filter pattern"""
        if pattern_type == "regex":
            try:
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
                self.regex_filters.append(compiled_pattern)
                logger.info(f"Added custom filter pattern: {pattern}")
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {e}")
        else:
            logger.warning(f"Unsupported pattern type: {pattern_type}")
