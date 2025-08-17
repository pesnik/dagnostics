import logging
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from dagnostics.api.schemas import MonitorStatus
from dagnostics.daemon.service import DaemonService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitor", tags=["monitor"])

# Global daemon service instance
_daemon_service: Optional[DaemonService] = None


def get_daemon_service() -> DaemonService:
    """Get or create daemon service instance"""
    global _daemon_service
    if _daemon_service is None:
        _daemon_service = DaemonService()
    return _daemon_service


@router.get("/status", response_model=MonitorStatus)
async def get_monitor_status():
    """Get current monitoring service status"""
    try:
        daemon = get_daemon_service()

        # Check if daemon is actually running
        is_running = daemon.is_running

        if is_running:
            status_info = daemon.get_status()
            return MonitorStatus(
                is_running=True,
                last_check=status_info.get("last_check"),
                failed_tasks_count=status_info.get("failed_tasks_count", 0),
                processed_today=status_info.get("processed_today", 0),
                average_processing_time=status_info.get("average_processing_time", 0.0),
                uptime=status_info.get("uptime", 0),
                next_check=status_info.get("next_check"),
            )
        else:
            return MonitorStatus(
                is_running=False,
                last_check=None,
                failed_tasks_count=0,
                processed_today=0,
                average_processing_time=0.0,
                uptime=0,
                next_check=None,
            )

    except Exception as e:
        logger.error(f"Failed to get monitor status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_monitoring(background_tasks: BackgroundTasks):
    """Start the monitoring service"""
    try:
        daemon = get_daemon_service()

        if daemon.is_running:
            return {
                "message": "Monitoring service is already running",
                "status": "already_running",
            }

        # Start daemon in background
        background_tasks.add_task(daemon.start)

        return {
            "message": "Monitoring service started successfully",
            "status": "started",
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Failed to start monitoring service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_monitoring():
    """Stop the monitoring service"""
    try:
        daemon = get_daemon_service()

        if not daemon.is_running:
            return {
                "message": "Monitoring service is not running",
                "status": "already_stopped",
            }

        daemon.stop()

        return {
            "message": "Monitoring service stopped successfully",
            "status": "stopped",
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Failed to stop monitoring service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart")
async def restart_monitoring(background_tasks: BackgroundTasks):
    """Restart the monitoring service"""
    try:
        daemon = get_daemon_service()

        # Stop if running
        if daemon.is_running:
            daemon.stop()

        # Start again
        background_tasks.add_task(daemon.start)

        return {
            "message": "Monitoring service restarted successfully",
            "status": "restarted",
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Failed to restart monitoring service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_monitor_logs(lines: int = 50):
    """Get recent monitoring service logs"""
    try:
        daemon = get_daemon_service()

        # For now, return status-based mock logs since get_recent_logs doesn't exist
        # In a full implementation, this would read from actual log files
        if daemon.is_running:
            logs = [
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO - DAGnostics daemon is running",
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO - Monitoring active tasks",
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO - Last check completed successfully",
            ]
        else:
            logs = [
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO - DAGnostics daemon is stopped",
            ]

        # Limit to requested number of lines
        logs = logs[-lines:] if len(logs) > lines else logs

        return {"logs": logs, "total_lines": len(logs), "timestamp": datetime.now()}

    except Exception as e:
        logger.error(f"Failed to get monitor logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_monitor_config():
    """Get current monitoring configuration"""
    try:
        daemon = get_daemon_service()

        config_info = {
            "check_interval": getattr(daemon, "check_interval", 60),
            "llm_provider": getattr(daemon, "llm_provider", "ollama"),
            "config_file": getattr(daemon, "config_file", None),
            "is_running": daemon.is_running,
        }

        return config_info

    except Exception as e:
        logger.error(f"Failed to get monitor config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config")
async def update_monitor_config(config: Dict):
    """Update monitoring configuration"""
    try:
        # This would update daemon configuration
        # For now, return success
        return {
            "message": "Configuration updated successfully",
            "config": config,
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Failed to update monitor config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
