import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Set

from fastapi import WebSocket

from dagnostics.api.schemas import RealTimeUpdate

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and real-time updates"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_info: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, client_info: Optional[Dict] = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_info[websocket] = client_info or {}

        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

        # Send welcome message
        await self.send_personal_message(
            websocket,
            {
                "type": "connection_established",
                "data": {
                    "message": "Connected to DAGnostics real-time updates",
                    "timestamp": datetime.now().isoformat(),
                    "connection_count": len(self.active_connections),
                },
            },
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if websocket in self.connection_info:
            del self.connection_info[websocket]

        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_personal_message(self, websocket: WebSocket, message: Dict):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict, exclude: Optional[Set[WebSocket]] = None):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return

        exclude = exclude or set()
        connections_to_remove = set()

        for connection in self.active_connections:
            if connection in exclude:
                continue

            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send broadcast message: {e}")
                connections_to_remove.add(connection)

        # Remove failed connections
        for connection in connections_to_remove:
            self.disconnect(connection)

    async def broadcast_update(self, update: RealTimeUpdate):
        """Broadcast a real-time update to all clients"""
        message = {
            "type": update.event_type,
            "data": update.data,
            "timestamp": update.timestamp.isoformat(),
            "dag_id": update.dag_id,
            "task_id": update.task_id,
            "run_id": update.run_id,
        }

        await self.broadcast(message)
        logger.debug(
            f"Broadcasted update: {update.event_type} for {update.dag_id}.{update.task_id}"
        )

    async def broadcast_analysis_complete(self, analysis_result):
        """Broadcast when an analysis is completed"""
        update = RealTimeUpdate(
            event_type="analysis_complete",
            dag_id=analysis_result.dag_id,
            task_id=analysis_result.task_id,
            run_id=analysis_result.run_id,
            data={
                "analysis_id": analysis_result.id,
                "success": analysis_result.success,
                "processing_time": analysis_result.processing_time,
                "error_message": (
                    analysis_result.analysis.error_message
                    if analysis_result.analysis
                    else None
                ),
                "category": (
                    analysis_result.analysis.category.value
                    if analysis_result.analysis
                    else None
                ),
                "severity": (
                    analysis_result.analysis.severity.value
                    if analysis_result.analysis
                    else None
                ),
            },
        )

        await self.broadcast_update(update)

    async def broadcast_new_failure(
        self, dag_id: str, task_id: str, run_id: str, error_info: Dict
    ):
        """Broadcast when a new failure is detected"""
        update = RealTimeUpdate(
            event_type="new_failure",
            dag_id=dag_id,
            task_id=task_id,
            run_id=run_id,
            data={
                "error_message": error_info.get("error_message"),
                "timestamp": error_info.get("timestamp", datetime.now().isoformat()),
                "severity": error_info.get("severity", "unknown"),
            },
        )

        await self.broadcast_update(update)

    async def broadcast_monitor_status(self, status: Dict):
        """Broadcast monitor status changes"""
        update = RealTimeUpdate(
            event_type="status_change",
            data={
                "monitor_status": status,
                "timestamp": datetime.now().isoformat(),
            },
        )

        await self.broadcast_update(update)

    async def send_heartbeat(self):
        """Send heartbeat to all connections to keep them alive"""
        if not self.active_connections:
            return

        heartbeat_message = {
            "type": "heartbeat",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "connection_count": len(self.active_connections),
            },
        }

        await self.broadcast(heartbeat_message)

    def get_connection_stats(self) -> Dict:
        """Get statistics about active connections"""
        return {
            "active_connections": len(self.active_connections),
            "connection_details": [
                {
                    "client_info": self.connection_info.get(conn, {}),
                    "connected_at": self.connection_info.get(conn, {}).get(
                        "connected_at"
                    ),
                }
                for conn in self.active_connections
            ],
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


async def start_heartbeat_task():
    """Start background task to send periodic heartbeats"""
    while True:
        try:
            await websocket_manager.send_heartbeat()
            await asyncio.sleep(
                60
            )  # Send heartbeat every 60 seconds (reduced frequency)
        except Exception as e:
            logger.error(f"Heartbeat task error: {e}")
            await asyncio.sleep(60)
