"""
Base classes and protocols for filter rule engines.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class CacheableEngine(Protocol):
    """Protocol for engines that support caching"""

    def get_cache_size(self) -> int:
        """Get the size of the cache"""
        ...

    def clear_cache(self) -> None:
        """Clear the cache"""
        ...


@runtime_checkable
class FunctionRegistryEngine(Protocol):
    """Protocol for engines that support function registration"""

    def get_registered_functions(self) -> list[str]:
        """Get list of registered function names"""
        ...

    def register_function(self, name: str, func) -> None:
        """Register a function"""
        ...

    def unregister_function(self, name: str) -> bool:
        """Unregister a function"""
        ...
