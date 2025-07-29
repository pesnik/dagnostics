#!/usr/bin/env python3
"""Test script to verify filter configuration"""

import os
import sys
from datetime import datetime

from dagnostics.core.config import load_config
from dagnostics.core.models import LogEntry
from dagnostics.llm.filter_integration import get_enhanced_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


def test_filter():
    config = load_config()
    filter_instance = get_enhanced_filter(config)

    # Test log entries
    test_logs = [
        LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            message="Task exited with return code 0",
            source="test",
            dag_id="test_dag",
            task_id="test_task",
            run_id="test_run",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level="ERROR",
            message="Connection refused to database",
            source="test",
            dag_id="test_dag",
            task_id="test_task",
            run_id="test_run",
        ),
    ]

    filtered = filter_instance.filter_candidates(test_logs)
    print(f"Original logs: {len(test_logs)}")
    print(f"After filtering: {len(filtered)}")
    print(f"Filter stats: {filter_instance.get_filter_stats()}")


if __name__ == "__main__":
    test_filter()
