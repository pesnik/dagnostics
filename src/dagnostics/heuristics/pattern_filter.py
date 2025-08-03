"""
Enhanced pattern filter with pluggable rule engines and decoupled configuration.
Main orchestrator for heuristic-based log filtering.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import yaml

from dagnostics.core.models import LogEntry
from dagnostics.heuristics.engines.base import (
    FilterRule,
    FilterRuleEngine,
    FilterRuleType,
)
from dagnostics.heuristics.engines.custom_function_engine import (
    CustomFunctionFilterEngine,
)
from dagnostics.heuristics.engines.engine_factory import EngineFactory
from dagnostics.heuristics.error_classifier import ErrorCandidateClassifier

# Import protocols only for type checking to avoid runtime issues
# if TYPE_CHECKING:
#     from dagnostics.heuristics.engines.base import CacheableEngine, FunctionRegistryEngine

logger = logging.getLogger(__name__)


class ErrorPatternFilter:
    """Enhanced filter with pluggable rule engines and decoupled configuration"""

    def __init__(self, config_path: Optional[str] = None):
        # Initialize rule engines using factory
        self.engines: Dict[FilterRuleType, FilterRuleEngine] = (
            EngineFactory.create_all_engines()
        )

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
        default_config: Dict[str, Any] = {
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
        config: Dict[str, Any] = default_config
        if config_path:
            try:
                with open(config_path, "r") as f:
                    custom_config = yaml.safe_load(f)
                    if isinstance(custom_config, dict):
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
        return not self._should_filter_out(log_entry)

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
        custom_engine = self.engines.get(FilterRuleType.CUSTOM_FUNCTION)
        if isinstance(custom_engine, CustomFunctionFilterEngine):
            custom_engine.register_function(name, func)
            # Also add it as a rule
            self.add_filter_rule(name, FilterRuleType.CUSTOM_FUNCTION, description)
        else:
            logger.error("Custom function engine not available")

    def remove_filter_rule(self, pattern: str, rule_type: FilterRuleType) -> bool:
        """Remove a filter rule"""
        for i, rule in enumerate(self.filter_rules):
            if rule.pattern == pattern and rule.rule_type == rule_type:
                removed_rule = self.filter_rules.pop(i)
                logger.info(
                    f"Removed filter rule: {removed_rule.description or pattern}"
                )
                return True
        return False

    def get_filter_stats(self) -> Dict[str, int]:
        """Get statistics about loaded filter rules"""
        stats = {}
        for rule_type in FilterRuleType:
            count = sum(1 for rule in self.filter_rules if rule.rule_type == rule_type)
            stats[rule_type.value] = count
        return stats

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about engines"""
        from dagnostics.heuristics.engines.base import (
            CacheableEngine,
            FunctionRegistryEngine,
        )

        stats = {}
        for rule_type, engine in self.engines.items():
            engine_stats = {"type": type(engine).__name__}

            # Add engine-specific stats using protocols for type safety
            if isinstance(engine, CacheableEngine):
                try:
                    engine_stats["cache_size"] = str(engine.get_cache_size())
                except Exception as e:
                    logger.warning(
                        f"Failed to get cache size for {rule_type.value}: {e}"
                    )

            if isinstance(engine, FunctionRegistryEngine):
                try:
                    engine_stats["registered_functions"] = ", ".join(
                        engine.get_registered_functions()
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to get registered functions for {rule_type.value}: {e}"
                    )

            stats[rule_type.value] = engine_stats
        return stats

    def clear_engine_caches(self):
        """Clear caches in engines that support it"""
        from dagnostics.heuristics.engines.base import CacheableEngine

        cleared_count = 0
        for rule_type, engine in self.engines.items():
            if isinstance(engine, CacheableEngine):
                try:
                    engine.clear_cache()
                    cleared_count += 1
                    logger.debug(f"Cleared cache for {rule_type.value} engine")
                except Exception as e:
                    logger.warning(f"Failed to clear cache for {rule_type.value}: {e}")

        if cleared_count > 0:
            logger.info(f"Cleared caches for {cleared_count} engines")
        else:
            logger.debug("No engine caches to clear")

    # Maintain backward compatibility
    def add_custom_filter(self, pattern: str, pattern_type: str = "regex"):
        """Backward compatibility method"""
        try:
            rule_type = FilterRuleType(pattern_type)
            self.add_filter_rule(pattern, rule_type, f"Custom {pattern_type} filter")
        except ValueError:
            logger.warning(f"Unsupported pattern type: {pattern_type}")
