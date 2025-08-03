"""
Factory for creating and managing filter rule engines.
"""

import logging
from typing import Dict, Type

from dagnostics.heuristics.engines.base import FilterRuleEngine, FilterRuleType
from dagnostics.heuristics.engines.custom_function_engine import (
    CustomFunctionFilterEngine,
)
from dagnostics.heuristics.engines.ends_with_engine import EndsWithFilterEngine
from dagnostics.heuristics.engines.regex_engine import RegexFilterEngine
from dagnostics.heuristics.engines.starts_with_engine import StartsWithFilterEngine
from dagnostics.heuristics.engines.substring_engine import SubstringFilterEngine
from dagnostics.heuristics.engines.threshold_engine import LengthThresholdFilterEngine

logger = logging.getLogger(__name__)


class EngineFactory:
    """Factory for creating filter rule engines"""

    # Registry of engine classes
    _engine_registry: Dict[FilterRuleType, Type[FilterRuleEngine]] = {
        FilterRuleType.REGEX: RegexFilterEngine,
        FilterRuleType.SUBSTRING: SubstringFilterEngine,
        FilterRuleType.CUSTOM_FUNCTION: CustomFunctionFilterEngine,
        FilterRuleType.STARTS_WITH: StartsWithFilterEngine,
        FilterRuleType.ENDS_WITH: EndsWithFilterEngine,
        FilterRuleType.LENGTH_THRESHOLD: LengthThresholdFilterEngine,
    }

    @classmethod
    def create_engine(cls, rule_type: FilterRuleType) -> FilterRuleEngine:
        """Create a single engine instance"""
        if rule_type not in cls._engine_registry:
            raise ValueError(f"Unknown rule type: {rule_type}")

        engine_class = cls._engine_registry[rule_type]
        return engine_class()

    @classmethod
    def create_all_engines(cls) -> Dict[FilterRuleType, FilterRuleEngine]:
        """Create all available engines"""
        engines = {}
        for rule_type, engine_class in cls._engine_registry.items():
            try:
                engines[rule_type] = engine_class()
                logger.debug(f"Created engine for {rule_type.value}")
            except Exception as e:
                logger.error(f"Failed to create engine for {rule_type.value}: {e}")

        logger.info(f"Created {len(engines)} filter engines")
        return engines

    @classmethod
    def register_engine(
        cls, rule_type: FilterRuleType, engine_class: Type[FilterRuleEngine]
    ):
        """Register a custom engine class"""
        if not issubclass(engine_class, FilterRuleEngine):
            raise TypeError("Engine class must inherit from FilterRuleEngine")

        cls._engine_registry[rule_type] = engine_class
        logger.info(f"Registered custom engine for {rule_type.value}")

    @classmethod
    def get_supported_types(cls) -> list[FilterRuleType]:
        """Get list of supported rule types"""
        return list(cls._engine_registry.keys())
