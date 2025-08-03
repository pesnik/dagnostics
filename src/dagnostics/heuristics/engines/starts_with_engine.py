"""
Starts-with filter engine for prefix matching.
"""

from dagnostics.core.models import LogEntry
from dagnostics.heuristics.engines.base import FilterRule, FilterRuleEngine


class StartsWithFilterEngine(FilterRuleEngine):
    """Engine for starts-with filtering"""

    def matches(self, log_entry: LogEntry, rule: FilterRule) -> bool:
        message = (
            log_entry.message if rule.case_sensitive else log_entry.message.lower()
        )
        pattern = rule.pattern if rule.case_sensitive else rule.pattern.lower()
        result = message.startswith(pattern)
        return not result if rule.negate else result
