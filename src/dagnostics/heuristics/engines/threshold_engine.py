"""
Length threshold filter engine for filtering by message length.
"""

import logging

from dagnostics.core.models import LogEntry
from dagnostics.heuristics.engines.base import FilterRule, FilterRuleEngine

logger = logging.getLogger(__name__)


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
