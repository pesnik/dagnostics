import asyncio
import json
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dagnostics.cli.utils import initialize_components
from dagnostics.monitoring.monitor import DAGMonitor

logger = logging.getLogger(__name__)


class DaemonService:
    """DAGnostics daemon service for background monitoring"""

    def __init__(self, config_file: Optional[str] = None, llm_provider: str = "ollama"):
        self.config_file = config_file
        self.llm_provider = llm_provider
        self.monitor: Optional[DAGMonitor] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.pid_file = self._get_pid_file()

    def _get_pid_file(self) -> Path:
        """Get the PID file path"""
        home_dir = Path.home()
        pid_dir = home_dir / ".dagnostics"
        pid_dir.mkdir(exist_ok=True)
        return pid_dir / "dagnostics.pid"

    def _get_status_file(self) -> Path:
        """Get the status file path"""
        home_dir = Path.home()
        status_dir = home_dir / ".dagnostics"
        status_dir.mkdir(exist_ok=True)
        return status_dir / "status.json"

    def _write_pid_file(self):
        """Write the PID file"""
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))

    def _remove_pid_file(self):
        """Remove the PID file"""
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _update_status(self, status: str, **kwargs):
        """Update the daemon status file"""
        status_data = {
            "status": status,
            "pid": os.getpid(),
            "timestamp": datetime.now().isoformat(),
            "config_file": self.config_file,
            "llm_provider": self.llm_provider,
            **kwargs,
        }

        with open(self._get_status_file(), "w") as f:
            json.dump(status_data, f, indent=2)

    def get_status(self) -> Dict[str, Any]:
        """Get the current daemon status"""
        status_file = self._get_status_file()

        if not status_file.exists():
            return {"status": "stopped", "message": "No status file found"}

        try:
            with open(status_file, "r") as f:
                status_data = json.load(f)

            # Check if the process is actually running
            pid = status_data.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)  # Check if process exists
                except (OSError, ProcessLookupError):
                    # Process doesn't exist
                    status_data["status"] = "stopped"
                    status_data["message"] = "Process not running"

            return status_data

        except (json.JSONDecodeError, IOError) as e:
            return {"status": "unknown", "error": str(e)}

    def is_daemon_running(self) -> bool:
        """Check if daemon is already running"""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, "r") as f:
                pid = int(f.read().strip())

            # Check if process is running
            os.kill(pid, 0)
            return True

        except (OSError, ProcessLookupError, ValueError):
            # Process doesn't exist or PID file is corrupted
            self._remove_pid_file()
            return False

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, _):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.is_running = False
            if self.monitor and hasattr(self.monitor, "stop_monitoring"):
                # Stop monitor gracefully without calling stop() again
                pass

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _run_monitor_loop(self, interval_minutes: int):
        """Run the monitoring loop in a separate thread"""

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            try:
                self.loop.run_until_complete(self._async_monitor_loop(interval_minutes))
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self._update_status("error", error=str(e))
            finally:
                if self.loop and not self.loop.is_closed():
                    self.loop.close()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    async def _async_monitor_loop(self, interval_minutes: int):
        """Async monitoring loop"""
        try:
            # Initialize components
            _, analyzer = initialize_components(self.config_file, self.llm_provider)

            # Create monitor instance
            self.monitor = DAGMonitor(
                analyzer, config={"check_interval_minutes": interval_minutes}
            )

            # Start monitoring
            await self.monitor.start_monitoring(interval_minutes)

            # Update status to running
            self._update_status("running", interval_minutes=interval_minutes)

            # Keep the loop running
            while self.is_running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            self._update_status("error", error=str(e))
            raise
        finally:
            if self.monitor:
                await self.monitor.stop_monitoring()

    def start(self, interval_minutes: int = 5, detach: bool = True) -> bool:
        """Start the daemon service"""
        if self.is_daemon_running():
            logger.warning("Daemon is already running")
            return False

        logger.info(f"Starting DAGnostics daemon (interval: {interval_minutes}m)")

        if detach:
            # Fork to become a daemon
            pid = os.fork()
            if pid > 0:
                # Parent process
                return True

            # Child process continues as daemon
            os.setsid()  # Create new session

            # Fork again to prevent zombie processes
            pid = os.fork()
            if pid > 0:
                os._exit(0)

            # Redirect standard file descriptors
            sys.stdin.close()
            sys.stdout.close()
            sys.stderr.close()

            # Open new file descriptors
            sys.stdin = open(os.devnull, "r")
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

        try:
            self.is_running = True
            self._write_pid_file()
            self._setup_signal_handlers()
            self._update_status("starting")

            # Start the monitoring loop
            self._run_monitor_loop(interval_minutes)

            if detach:
                # Keep the daemon process alive
                while self.is_running:
                    import time

                    time.sleep(1)
            else:
                # Wait for the monitoring thread
                if self.thread:
                    self.thread.join()

            return True

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            self._update_status("failed", error=str(e))
            self._remove_pid_file()
            return False

    def stop(self) -> bool:
        """Stop the daemon service"""
        if not self.is_daemon_running():
            logger.warning("Daemon is not running")
            return False

        try:
            with open(self.pid_file, "r") as f:
                pid = int(f.read().strip())

            # Send SIGTERM to the daemon process
            os.kill(pid, signal.SIGTERM)

            # Wait for process to terminate
            import time

            for _ in range(30):  # Wait up to 30 seconds
                try:
                    os.kill(pid, 0)
                    time.sleep(1)
                except (OSError, ProcessLookupError):
                    break
            else:
                # Force kill if still running
                try:
                    os.kill(pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass

            self._remove_pid_file()
            self._update_status("stopped")

            # Stop local components if running in the same process
            if self.monitor and self.loop:
                if self.loop.is_running():

                    async def stop_monitor():
                        if self.monitor:
                            await self.monitor.stop_monitoring()

                    self.loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(stop_monitor())
                    )

            self.is_running = False
            logger.info("Daemon stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop daemon: {e}")
            return False

    def restart(self, interval_minutes: int = 5) -> bool:
        """Restart the daemon service"""
        logger.info("Restarting daemon...")

        if self.is_daemon_running():
            if not self.stop():
                return False

            # Wait a moment for cleanup
            import time

            time.sleep(2)

        return self.start(interval_minutes)
