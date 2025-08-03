"""
Heuristics module for rule-based filtering and pattern matching.
Provides deterministic, non-ML approaches for log analysis.
"""

from dagnostics.heuristics.engines import (
    CacheableEngine,
    FilterRule,
    FilterRuleType,
    FunctionRegistryEngine,
)
from dagnostics.heuristics.error_classifier import ErrorCandidateClassifier
from dagnostics.heuristics.filter_factory import FilterFactory
from dagnostics.heuristics.filter_integration import IntegratedErrorPatternFilter
from dagnostics.heuristics.pattern_filter import ErrorPatternFilter

__all__ = [
    "ErrorPatternFilter",
    "FilterRule",
    "FilterRuleType",
    "FilterFactory",
    "IntegratedErrorPatternFilter",
    "ErrorCandidateClassifier",
    "CacheableEngine",
    "FunctionRegistryEngine",
]
