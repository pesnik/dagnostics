"""
Error candidate classifier for identifying potential error log entries.
Uses heuristic rules to classify log entries as error candidates.
"""

import logging
from typing import Any, Dict, List, Optional

from dagnostics.core.models import LogEntry

logger = logging.getLogger(__name__)


class ErrorCandidateClassifier:
    """Classifies log entries as error candidates based on positive indicators"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        _config_dict: Dict[str, Any] = config or {}

        # Default error indicators
        self.error_indicators: List[str] = _config_dict.get(
            "error_indicators",
            [
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
            ],
        )

        # Log levels that are considered errors
        self.error_levels: List[str] = _config_dict.get(
            "error_levels", ["ERROR", "CRITICAL", "FATAL"]
        )

        # Minimum confidence score to be considered an error candidate
        self.min_confidence_score: float = _config_dict.get("min_confidence_score", 0.3)

        # Weights for different indicators
        self.level_weight: float = _config_dict.get("level_weight", 0.8)
        self.indicator_weight: float = _config_dict.get("indicator_weight", 0.2)
        self.stack_trace_weight: float = _config_dict.get("stack_trace_weight", 0.3)

    def is_error_candidate(self, log_entry: LogEntry) -> tuple[bool, float]:
        """
        Check if log entry is an error candidate
        Returns: (is_candidate, confidence_score)
        """
        confidence_score = 0.0

        # Check log level
        if log_entry.level.upper() in self.error_levels:
            confidence_score += self.level_weight

        # Check for error indicators
        message_lower = log_entry.message.lower()
        indicator_matches = sum(
            1 for indicator in self.error_indicators if indicator in message_lower
        )

        if indicator_matches > 0:
            confidence_score += min(0.6, indicator_matches * self.indicator_weight)

        # Check for stack traces or multiline error patterns
        if any(
            pattern in log_entry.message
            for pattern in ["\n", "Traceback", "Exception in thread"]
        ):
            confidence_score += self.stack_trace_weight

        is_candidate = confidence_score >= self.min_confidence_score
        return is_candidate, confidence_score

    def classify_batch(
        self, log_entries: List[LogEntry]
    ) -> List[tuple[LogEntry, bool, float]]:
        """
        Classify a batch of log entries
        Returns: List of (log_entry, is_candidate, confidence_score)
        """
        results = []
        for log_entry in log_entries:
            is_candidate, confidence = self.is_error_candidate(log_entry)
            results.append((log_entry, is_candidate, confidence))
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier configuration statistics"""
        return {
            "error_indicators_count": len(self.error_indicators),
            "error_levels_count": len(self.error_levels),
            "min_confidence_threshold": self.min_confidence_score,
            "weights": {
                "level": self.level_weight,
                "indicator": self.indicator_weight,
                "stack_trace": self.stack_trace_weight,
            },
        }

    def update_config(self, config: Dict[str, Any]):
        """Update classifier configuration dynamically"""
        if "error_indicators" in config:
            self.error_indicators = config["error_indicators"]
        if "error_levels" in config:
            self.error_levels = config["error_levels"]
        if "min_confidence_score" in config:
            self.min_confidence_score = config["min_confidence_score"]

        logger.info("Updated error classifier configuration")
