import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast  # Import necessary types

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from dagnostics.analysis.analyzer import DAGAnalyzer
from dagnostics.core.models import AnalysisResult, ErrorSeverity, TaskInstance

logger = logging.getLogger(__name__)


class DAGMonitor:
    """Continuous monitoring service for DAG failures"""

    def __init__(
        self,
        analyzer: DAGAnalyzer,
        alert_manager: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.analyzer = analyzer
        self.alert_manager = alert_manager
        self.config = config or {}
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        self.stats: Dict[str, Union[int, float, datetime, None]] = {
            "processed_today": 0,
            "failed_tasks_found": 0,
            "last_check": None,
            "total_processing_time": 0.0,
        }

    async def start_monitoring(self, interval_minutes: int = 5):
        """Start the monitoring service"""
        if self.is_running:
            logger.warning("Monitor is already running")
            return

        logger.info(f"Starting DAGnostics monitor with {interval_minutes}m interval")

        # Schedule periodic checks
        self.scheduler.add_job(
            self.check_failed_tasks,
            IntervalTrigger(minutes=interval_minutes),
            id="check_failed_tasks",
            max_instances=1,
        )

        # Schedule daily cleanup
        self.scheduler.add_job(
            self.daily_cleanup,
            trigger="cron",
            hour=2,
            id="daily_cleanup",  # 2 AM daily
        )

        self.scheduler.start()
        self.is_running = True

        # Run initial check
        await self.check_failed_tasks()

    async def stop_monitoring(self):
        """Stop the monitoring service"""
        if not self.is_running:
            return

        logger.info("Stopping DAGnostics monitor")
        self.scheduler.shutdown()
        self.is_running = False

    async def check_failed_tasks(self):
        """Check for new failed tasks and analyze them"""
        try:
            logger.info("Checking for failed tasks...")
            self.stats["last_check"] = datetime.now()

            # Get failed tasks from last check interval
            interval_minutes = self.config.get("check_interval_minutes", 5)
            failed_tasks = self.analyzer.airflow_client.get_failed_tasks(
                interval_minutes
            )

            self.stats["failed_tasks_found"] = len(failed_tasks)

            if not failed_tasks:
                logger.info("No failed tasks found")
                return

            logger.info(f"Found {len(failed_tasks)} failed tasks to analyze")

            # Process each failed task
            analysis_results = []
            for task in failed_tasks:
                try:
                    result = await self.process_failure(task)
                    self.stats["processed_today"] = (
                        cast(int, self.stats["processed_today"]) + 1
                    )
                    if result.processing_time is not None:
                        self.stats["total_processing_time"] = (
                            cast(float, self.stats["total_processing_time"])
                            + result.processing_time
                        )
                    else:
                        logger.warning(
                            f"Processing time for {result.id} is None, skipping addition to total."
                        )

                    analysis_results.append(result)

                except Exception as e:
                    logger.error(f"Failed to process {task.dag_id}.{task.task_id}: {e}")

            # Send alerts for critical failures
            await self.send_alerts(analysis_results)

            # Store results for reporting
            await self.store_results(analysis_results)

        except Exception as e:
            logger.error(f"Monitor check failed: {e}")

    async def process_failure(self, task_instance: TaskInstance) -> AnalysisResult:
        """Process a single task failure"""
        logger.info(
            f"Processing failure: {task_instance.dag_id}.{task_instance.task_id}"
        )

        # Run analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.analyzer.analyze_task_failure,
            task_instance.dag_id,
            task_instance.task_id,
            task_instance.run_id,
            task_instance.try_number,
        )

        return result

    async def send_alerts(self, results: List[AnalysisResult]):
        """Send alerts for critical failures"""
        if not self.alert_manager:
            return

        critical_failures = [
            r
            for r in results
            if r.analysis
            and r.analysis.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        ]

        for result in critical_failures:
            try:
                await self.alert_manager.send_alert(result)
            except Exception as e:
                logger.error(
                    f"Failed to send alert for {result.dag_id}.{result.task_id}: {e}"
                )

    async def store_results(self, results: List[AnalysisResult]):
        """Store analysis results for reporting"""
        # This would integrate with your database
        logger.info(f"Storing {len(results)} analysis results")

        # For now, just log the results
        for result in results:
            if result.success and result.analysis:
                logger.info(
                    f"Stored: {result.dag_id}.{result.task_id} - "
                    f"{result.analysis.category.value} - {result.analysis.severity.value}"
                )

    async def daily_cleanup(self):
        """Daily cleanup tasks"""
        logger.info("Running daily cleanup...")

        # Reset daily stats
        self.stats["processed_today"] = 0
        self.stats["total_processing_time"] = 0.0

        # Clean up old cluster data
        # Clean up old logs
        # Generate daily reports

        logger.info("Daily cleanup completed")

    def get_stats(self) -> dict:
        """Get monitoring statistics"""
        avg_processing_time = 0.0
        processed_today = cast(int, self.stats["processed_today"])
        total_processing_time = cast(float, self.stats["total_processing_time"])

        if processed_today > 0:
            avg_processing_time = total_processing_time / processed_today

        return {
            **self.stats,
            "is_running": self.is_running,
            "average_processing_time": avg_processing_time,
        }
