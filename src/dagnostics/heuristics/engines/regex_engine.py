"""
Regex-based filter engine for pattern matching.
"""

import logging
import re
from re import Pattern
from typing import Dict

from dagnostics.core.models import LogEntry
from dagnostics.heuristics.engines.base import FilterRule, FilterRuleEngine

logger = logging.getLogger(__name__)


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

    def clear_cache(self):
        """Clear compiled pattern cache"""
        self._compiled_patterns.clear()

    def get_cache_size(self) -> int:
        """Get number of compiled patterns in cache"""
        return len(self._compiled_patterns)
