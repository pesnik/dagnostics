"""
Custom function-based filter engine for programmable filtering logic.
"""

import logging
from typing import Callable, Dict

from dagnostics.core.models import LogEntry
from dagnostics.heuristics.engines.base import FilterRule, FilterRuleEngine

logger = logging.getLogger(__name__)


class CustomFunctionFilterEngine(FilterRuleEngine):
    """Engine for custom function-based filtering"""

    def __init__(self):
        self._functions: Dict[str, Callable[[LogEntry], bool]] = {}

    def register_function(self, name: str, func: Callable[[LogEntry], bool]):
        """Register a custom filter function"""
        self._functions[name] = func
        logger.debug(f"Registered custom function: {name}")

    def unregister_function(self, name: str) -> bool:
        """Unregister a custom filter function"""
        if name in self._functions:
            del self._functions[name]
            logger.debug(f"Unregistered custom function: {name}")
            return True
        return False

    def get_registered_functions(self) -> list[str]:
        """Get list of registered function names"""
        return list(self._functions.keys())

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
