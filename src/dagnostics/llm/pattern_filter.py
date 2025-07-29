import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from re import Pattern
from typing import Any, Callable, Dict, List, Optional

import yaml

from dagnostics.core.models import LogEntry

logger = logging.getLogger(__name__)


class FilterRuleType(Enum):
    """Types of filter rules available"""

    REGEX = "regex"
    SUBSTRING = "substring"
    CUSTOM_FUNCTION = "custom_function"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    LENGTH_THRESHOLD = "length_threshold"


@dataclass
class FilterRule:
    """Represents a single filter rule"""

    pattern: str
    rule_type: FilterRuleType
    description: str = ""
    case_sensitive: bool = False
    negate: bool = False  # If True, rule passes when pattern DOESN'T match


class FilterRuleEngine(ABC):
    """Abstract base class for filter rule engines"""

    @abstractmethod
    def matches(self, log_entry: LogEntry, rule: FilterRule) -> bool:
        """Check if log entry matches the given rule"""
        pass


class RegexFilterEngine(FilterRuleEngine):
    """Engine for regex-based filtering"""

    def __init__(self):
        self._compiled_patterns: Dict[str, Pattern[str]] = {}

    def matches(self, log_entry: LogEntry, rule: FilterRule) -> bool:
        if rule.pattern not in self._compiled_patterns:
            flags = 0 if rule.case_sensitive else re.IGNORECASE
            try:
                self._compiled_patterns[rule.pattern] = re.compile(rule.pattern, flags)
            except re.error as e:
                logger.error(f"Invalid regex pattern '{rule.pattern}': {e}")
                return False

        pattern = self._compiled_patterns[rule.pattern]
        result = bool(pattern.search(log_entry.message))
        return not result if rule.negate else result


class SubstringFilterEngine(FilterRuleEngine):
    """Engine for substring-based filtering"""

    def matches(self, log_entry: LogEntry, rule: FilterRule) -> bool:
        message = (
            log_entry.message if rule.case_sensitive else log_entry.message.lower()
        )
        pattern = rule.pattern if rule.case_sensitive else rule.pattern.lower()
        result = pattern in message
        return not result if rule.negate else result


class CustomFunctionFilterEngine(FilterRuleEngine):
    """Engine for custom function-based filtering"""

    def __init__(self):
        self._functions: Dict[str, Callable[[LogEntry], bool]] = {}

    def register_function(self, name: str, func: Callable[[LogEntry], bool]):
        """Register a custom filter function"""
        self._functions[name] = func

    def matches(self, log_entry: LogEntry, rule: FilterRule) -> bool:
        if rule.pattern not in self._functions:
            logger.warning(f"Custom function '{rule.pattern}' not registered")
            return False

        try:
            result = self._functions[rule.pattern](log_entry)
            return not result if rule.negate else result
        except Exception as e:
            logger.error(f"Error executing custom function '{rule.pattern}': {e}")
            return False


class StartsWithFilterEngine(FilterRuleEngine):
    """Engine for starts-with filtering"""

    def matches(self, log_entry: LogEntry, rule: FilterRule) -> bool:
        message = (
            log_entry.message if rule.case_sensitive else log_entry.message.lower()
        )
        pattern = rule.pattern if rule.case_sensitive else rule.pattern.lower()
        result = message.startswith(pattern)
        return not result if rule.negate else result


class EndsWithFilterEngine(FilterRuleEngine):
    """Engine for ends-with filtering"""

    def matches(self, log_entry: LogEntry, rule: FilterRule) -> bool:
        message = (
            log_entry.message if rule.case_sensitive else log_entry.message.lower()
        )
        pattern = rule.pattern if rule.case_sensitive else rule.pattern.lower()
        result = message.endswith(pattern)
        return not result if rule.negate else result


class LengthThresholdFilterEngine(FilterRuleEngine):
    """Engine for length-based filtering"""

    def matches(self, log_entry: LogEntry, rule: FilterRule) -> bool:
        try:
            threshold = int(rule.pattern)
            result = len(log_entry.message.strip()) < threshold
            return not result if rule.negate else result
        except ValueError:
            logger.error(f"Invalid length threshold: {rule.pattern}")
            return False


class ErrorCandidateClassifier:
    """Classifies log entries as error candidates based on positive indicators"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        _config_dict: Dict[str, Any] = config or {}  # Explicitly type it for mypy

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

    def is_error_candidate(self, log_entry: LogEntry) -> tuple[bool, float]:
        """
        Check if log entry is an error candidate
        Returns: (is_candidate, confidence_score)
        """
        confidence_score = 0.0

        # Check log level
        if log_entry.level.upper() in self.error_levels:
            confidence_score += 0.8

        # Check for error indicators
        message_lower = log_entry.message.lower()
        indicator_matches = sum(
            1 for indicator in self.error_indicators if indicator in message_lower
        )

        if indicator_matches > 0:
            confidence_score += min(0.6, indicator_matches * 0.2)

        # Check for stack traces or multiline error patterns
        if any(
            pattern in log_entry.message
            for pattern in ["\n", "Traceback", "Exception in thread"]
        ):
            confidence_score += 0.3

        is_candidate = confidence_score >= self.min_confidence_score
        return is_candidate, confidence_score


class ErrorPatternFilter:
    """Enhanced filter with pluggable rule engines and decoupled configuration"""

    def __init__(self, config_path: Optional[str] = None):
        # Initialize rule engines
        self.engines: Dict[FilterRuleType, FilterRuleEngine] = {
            FilterRuleType.REGEX: RegexFilterEngine(),
            FilterRuleType.SUBSTRING: SubstringFilterEngine(),
            FilterRuleType.CUSTOM_FUNCTION: CustomFunctionFilterEngine(),
            FilterRuleType.STARTS_WITH: StartsWithFilterEngine(),
            FilterRuleType.ENDS_WITH: EndsWithFilterEngine(),
            FilterRuleType.LENGTH_THRESHOLD: LengthThresholdFilterEngine(),
        }

        # Initialize components
        self.filter_rules: List[FilterRule] = []
        self.classifier: (
            ErrorCandidateClassifier  # Will be initialized in load_configuration
        )

        # Load configuration
        self.load_configuration(config_path)

    def load_configuration(self, config_path: Optional[str] = None):
        """Load filtering configuration from file"""
        # Default configuration
        default_config: Dict[str, Any] = {  # Explicitly type default_config
            "filter_rules": [
                {
                    "pattern": r".*DEBUG.*",
                    "type": "regex",
                    "description": "Debug level logs",
                },
                {
                    "pattern": r".*Starting.*",
                    "type": "regex",
                    "description": "Startup messages",
                },
                {
                    "pattern": r".*Finished.*",
                    "type": "regex",
                    "description": "Completion messages",
                },
                {
                    "pattern": "Marking task as SUCCESS",
                    "type": "substring",
                    "description": "Airflow success",
                },
                {
                    "pattern": "Task exited with return code 0",
                    "type": "substring",
                    "description": "Successful exit",
                },
                {
                    "pattern": "Dependencies all met for",
                    "type": "substring",
                    "description": "Airflow dependencies",
                },
                {
                    "pattern": "UserWarning",
                    "type": "substring",
                    "description": "Non-critical warning",
                },
                {
                    "pattern": "DeprecationWarning",
                    "type": "substring",
                    "description": "Deprecation warning",
                },
                {
                    "pattern": "10",
                    "type": "length_threshold",
                    "description": "Very short messages",
                },
                {
                    "pattern": "tpt_info_line",
                    "type": "custom_function",
                    "description": "TPT informational messages",
                },
            ],
            "classifier": {
                "error_indicators": [
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
                "error_levels": ["ERROR", "CRITICAL", "FATAL"],
                "min_confidence_score": 0.3,
            },
        }

        # Load custom configuration if provided
        config: Dict[str, Any] = (
            default_config  # Initialize config with default_config and its type
        )
        if config_path:
            try:
                with open(config_path, "r") as f:
                    custom_config = yaml.safe_load(f)
                    if isinstance(
                        custom_config, dict
                    ):  # Ensure custom_config is a dict before updating
                        config.update(custom_config)
                    else:
                        logger.warning(
                            f"Custom configuration from {config_path} is not a dictionary. Skipping update."
                        )
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load configuration from {config_path}: {e}")

        # Parse filter rules
        self.filter_rules = []
        for rule_config in config.get("filter_rules", []):
            try:
                # Ensure rule_config is a dict as well
                if not isinstance(rule_config, dict):
                    logger.warning(
                        f"Invalid filter rule entry (not a dict): {rule_config}"
                    )
                    continue

                rule_type = FilterRuleType(rule_config["type"])
                rule = FilterRule(
                    pattern=rule_config["pattern"],
                    rule_type=rule_type,
                    description=rule_config.get("description", ""),
                    case_sensitive=rule_config.get("case_sensitive", False),
                    negate=rule_config.get("negate", False),
                )
                self.filter_rules.append(rule)
            except (KeyError, ValueError) as e:
                logger.warning(
                    f"Invalid filter rule configuration: {rule_config}, error: {e}"
                )

        # Initialize classifier with config
        classifier_config_raw = config.get("classifier", {})
        # Explicitly ensure classifier_config is Dict[str, Any]
        classifier_config: Dict[str, Any] = {}
        if isinstance(classifier_config_raw, dict):
            classifier_config = classifier_config_raw
        else:
            logger.warning(
                f"Classifier configuration is not a dictionary, using empty config: {classifier_config_raw}"
            )

        self.classifier = ErrorCandidateClassifier(classifier_config)

        logger.info(f"Loaded {len(self.filter_rules)} filter rules")

    def filter_candidates(self, anomalous_logs: List[LogEntry]) -> List[LogEntry]:
        """Filter out non-error log entries using the rule engine"""
        filtered_logs: List[LogEntry] = []

        for log_entry in anomalous_logs:
            if self._is_error_candidate(log_entry):
                filtered_logs.append(log_entry)

        logger.info(
            f"Filtered {len(anomalous_logs)} logs down to {len(filtered_logs)} candidates"
        )
        return filtered_logs

    def _is_error_candidate(self, log_entry: LogEntry) -> bool:
        """Check if log entry is likely an error candidate"""
        # First, check if it should be filtered out
        if self._should_filter_out(log_entry):
            return False

        # Then, check if it's a positive error candidate
        is_candidate, confidence = self.classifier.is_error_candidate(log_entry)
        return is_candidate

    def _should_filter_out(self, log_entry: LogEntry) -> bool:
        """Check if log entry should be filtered out based on rules"""
        for rule in self.filter_rules:
            engine = self.engines.get(rule.rule_type)
            if engine and engine.matches(log_entry, rule):
                logger.debug(f"Filtered out log entry due to rule: {rule.description}")
                return True
        return False

    def add_filter_rule(
        self,
        pattern: str,
        rule_type: FilterRuleType,
        description: str = "",
        case_sensitive: bool = False,
        negate: bool = False,
    ):
        """Add a new filter rule dynamically"""
        rule = FilterRule(
            pattern=pattern,
            rule_type=rule_type,
            description=description,
            case_sensitive=case_sensitive,
            negate=negate,
        )
        self.filter_rules.append(rule)
        logger.info(f"Added filter rule: {description or pattern}")

    def add_custom_function(
        self, name: str, func: Callable[[LogEntry], bool], description: str = ""
    ):
        """Add a custom filter function"""
        custom_engine = self.engines[FilterRuleType.CUSTOM_FUNCTION]
        if isinstance(custom_engine, CustomFunctionFilterEngine):
            custom_engine.register_function(name, func)

            # Also add it as a rule
            self.add_filter_rule(name, FilterRuleType.CUSTOM_FUNCTION, description)
        else:
            logger.error("Custom function engine not available")

    def get_filter_stats(self) -> Dict[str, int]:
        """Get statistics about loaded filter rules"""
        stats = {}
        for rule_type in FilterRuleType:
            count = sum(1 for rule in self.filter_rules if rule.rule_type == rule_type)
            stats[rule_type.value] = count
        return stats

    # Maintain backward compatibility
    def add_custom_filter(self, pattern: str, pattern_type: str = "regex"):
        """Backward compatibility method"""
        try:
            rule_type = FilterRuleType(pattern_type)
            self.add_filter_rule(pattern, rule_type, f"Custom {pattern_type} filter")
        except ValueError:
            logger.warning(f"Unsupported pattern type: {pattern_type}")
