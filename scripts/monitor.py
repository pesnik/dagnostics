# #!/usr/bin/env python3
# """
# Standalone monitoring script that can be run independently
# """
# import asyncio
# import sys
# import os
# import signal
# from pathlib import Path
# from typing import Optional

# # Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# from dagnostics.core.config import load_config
# from dagnostics.monitoring.airflow_client import AirflowClient
# from dagnostics.llm.log_clusterer import LogClusterer
# from dagnostics.llm.pattern_filter import ErrorPatternFilter
# from dagnostics.llm.engine import LLMEngine, OllamaProvider
# from dagnostics.monitoring.analyzer import DAGAnalyzer
# from dagnostics.monitoring.monitor import DAGMonitor
# from dagnostics.monitoring.alert import AlertManager
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('logs/monitor.log'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# logger = logging.getLogger(__name__)

# class MonitorService:
#     """Wrapper class for the monitoring service with proper lifecycle management"""

#     def __init__(self):
#         self.monitor: Optional[DAGMonitor] = None
#         self.running = False
#         self._setup_signal_handlers()

#     def _setup_signal_handlers(self):
#         """Setup signal handlers for graceful shutdown"""
#         def signal_handler(signum, frame):
#             logger.info(f"Received signal {signum}, initiating graceful shutdown...")
#             self.running = False

#         signal.signal(signal.SIGINT, signal_handler)
#         signal.signal(signal.SIGTERM, signal_handler)

#     async def initialize(self, config_path: Optional[str] = None):
#         """Initialize all monitoring components"""
#         try:
#             logger.info("Initializing DAGnostics monitoring service...")

#             # Load configuration
#             config = load_config(config_path)
#             logger.info("Configuration loaded successfully")

#             # Initialize Airflow client
#             airflow_client = AirflowClient(
#                 base_url=config['airflow']['base_url'],
#                 username=config['airflow']['username'],
#                 password=config['airflow']['password'],
#                 db_connection=config['airflow']['database_url']
#             )
#             logger.info("Airflow client initialized")

#             # Test Airflow connection
#             await airflow_client.test_connection()
#             logger.info("Airflow connection verified")

#             # Initialize log clustering
#             persistence_path = config.get('drain3', {}).get('persistence_path', 'data/clusters/drain3_state.pkl')
#             clusterer = LogClusterer(persistence_path=persistence_path)
#             logger.info("Log clusterer initialized")

#             # Initialize error pattern filter
#             pattern_filter = ErrorPatternFilter()
#             logger.info("Error pattern filter initialized")

#             # Initialize LLM provider
#             llm_config = config['llm']['providers']['ollama']
#             llm_provider = OllamaProvider(
#                 base_url=llm_config['base_url'],
#                 model=llm_config['model']
#             )

#             # Test LLM connection
#             await llm_provider.test_connection()
#             logger.info("LLM provider connection verified")

#             llm_engine = LLMEngine(llm_provider)
#             logger.info("LLM engine initialized")

#             # Initialize analyzer
#             analyzer = DAGAnalyzer(airflow_client, clusterer, pattern_filter, llm_engine)
#             logger.info("DAG analyzer initialized")

#             # Initialize alert manager
#             alert_manager = AlertManager(config)
#             logger.info("Alert manager initialized")

#             # Initialize monitor
#             self.monitor = DAGMonitor(analyzer, alert_manager, config)
#             logger.info("DAG monitor initialized successfully")

#         except Exception as e:
#             logger.error(f"Failed to initialize monitoring service: {e}")
#             raise

#     async def start(self, interval: int = 5):
#         """Start the monitoring service"""
#         if not self.monitor:
#             raise RuntimeError("Monitor not initialized. Call initialize() first.")

#         self.running = True
#         logger.info(f"Starting DAGnostics monitor with {interval} minute interval")

#         try:
#             # Start the monitor
#             await self.monitor.start_monitoring(interval)

#             # Main monitoring loop
#             while self.running:
#                 await asyncio.sleep(10)  # Check every 10 seconds for shutdown signal

#         except asyncio.CancelledError:
#             logger.info("Monitor cancelled")
#         except Exception as e:
#             logger.error(f"Monitor encountered an error: {e}")
#             raise
#         finally:
#             await self.stop()

#     async def stop(self):
#         """Stop the monitoring service gracefully"""
#         logger.info("Stopping monitoring service...")

#         if self.monitor:
#             try:
#                 await self.monitor.stop_monitoring()
#                 logger.info("Monitor stopped successfully")
#             except Exception as e:
#                 logger.error(f"Error stopping monitor: {e}")

#         self.running = False
#         logger.info("Monitoring service stopped")

# async def main():
#     """Main entry point"""
#     import argparse

#     parser = argparse.ArgumentParser(description='DAGnostics Monitoring Service')
#     parser.add_argument('--interval', '-i', type=int, default=5,
#                        help='Monitoring interval in minutes (default: 5)')
#     parser.add_argument('--config', '-c', type=str,
#                        help='Path to configuration file')
#     parser.add_argument('--daemon', '-d', action='store_true',
#                        help='Run as daemon (no interactive output)')
#     parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
#                        default='INFO', help='Logging level')

#     args = parser.parse_args()

#     # Set log level
#     logging.getLogger().setLevel(getattr(logging, args.log_level))

#     # Initialize service
#     service = MonitorService()

#     try:
#         await service.initialize(args.config)

#         if not args.daemon:
#             print("=" * 60)
#             print("🚀 DAGnostics Monitor Started")
#             print("=" * 60)
#             print(f"Monitoring interval: {args.interval} minutes")
#             print(f"Log level: {args.log_level}")
#             print("Press Ctrl+C to stop...")
#             print("=" * 60)

#         await service.start(args.interval)

#     except KeyboardInterrupt:
#         logger.info("Received keyboard interrupt")
#     except Exception as e:
#         logger.error(f"Monitor failed: {e}")
#         sys.exit(1)
#     finally:
#         if not args.daemon:
#             print("\n👋 DAGnostics Monitor stopped")

# def run_sync():
#     """Synchronous wrapper for running the async main function"""
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         pass
#     except Exception as e:
#         print(f"Fatal error: {e}", file=sys.stderr)
#         sys.exit(1)

# if __name__ == "__main__":
#     # Ensure logs directory exists
#     os.makedirs('logs', exist_ok=True)

#     run_sync()
