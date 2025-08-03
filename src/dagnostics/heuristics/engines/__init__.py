"""
Filter engines for different types of pattern matching.
Each engine implements a specific matching strategy.
"""

# Import base classes and protocols
from dagnostics.heuristics.engines.base import (
    CacheableEngine,
    FilterRule,
    FilterRuleEngine,
    FilterRuleType,
    FunctionRegistryEngine,
)
from dagnostics.heuristics.engines.custom_function_engine import (
    CustomFunctionFilterEngine,
)
from dagnostics.heuristics.engines.ends_with_engine import EndsWithFilterEngine
from dagnostics.heuristics.engines.engine_factory import EngineFactory

# Import concrete engines
from dagnostics.heuristics.engines.regex_engine import RegexFilterEngine
from dagnostics.heuristics.engines.starts_with_engine import StartsWithFilterEngine
from dagnostics.heuristics.engines.substring_engine import SubstringFilterEngine
from dagnostics.heuristics.engines.threshold_engine import LengthThresholdFilterEngine

__all__ = [
    # Base classes and protocols
    "FilterRuleEngine",
    "FilterRuleType",
    "FilterRule",
    "CacheableEngine",
    "FunctionRegistryEngine",
    # Concrete engines
    "RegexFilterEngine",
    "SubstringFilterEngine",
    "CustomFunctionFilterEngine",
    "StartsWithFilterEngine",
    "EndsWithFilterEngine",
    "LengthThresholdFilterEngine",
    "EngineFactory",
]
