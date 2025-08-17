import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dagnostics import __version__
from dagnostics.api.routes import analysis, dashboard, monitor, training
from dagnostics.api.schemas import RealTimeUpdate
from dagnostics.api.websocket_manager import start_heartbeat_task, websocket_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    import asyncio

    # Initialize database
    from dagnostics.core.database import init_database

    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Start heartbeat task
    asyncio.create_task(start_heartbeat_task())
    logger.info("DAGnostics API started with real-time capabilities")

    yield

    # Shutdown
    logger.info("DAGnostics API shutting down")


app = FastAPI(
    title="DAGnostics API",
    description="Intelligent ETL Monitoring and Analysis API",
    version=__version__,
    lifespan=lifespan,
)

# Include routers
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")
app.include_router(monitor.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")

# Mount static files
try:
    app.mount(
        "/static", StaticFiles(directory="src/dagnostics/web/static"), name="static"
    )
except Exception:
    pass  # Static directory might not exist

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/")
async def root():
    return {"message": "DAGnostics API", "version": __version__, "status": "running"}


@app.get("/dashboard")
async def serve_dashboard():
    """Serve the training dataset creator"""
    try:
        return FileResponse("src/dagnostics/web/templates/training.html")
    except Exception:
        return {
            "message": "Dashboard not available",
            "error": "Template file not found",
        }


@app.get("/dashboard/monitoring")
async def serve_monitoring_dashboard():
    """Serve the original monitoring dashboard"""
    try:
        return FileResponse("src/dagnostics/web/templates/index.html")
    except Exception:
        return {
            "message": "Dashboard not available",
            "error": "Template file not found",
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


# Legacy analyze endpoint for backward compatibility
@app.post("/api/v1/analyze")
async def analyze_task_legacy(request: dict):
    """Legacy analyze endpoint - redirects to new analysis route"""
    try:
        from dagnostics.api.routes.analysis import get_analyzer
        from dagnostics.api.schemas import AnalyzeRequest

        analyzer = get_analyzer()
        analyze_request = AnalyzeRequest(**request)
        result = await analysis.analyze_task_failure(analyze_request, analyzer)

        # Broadcast real-time update
        await websocket_manager.broadcast_analysis_complete(result)

        return result
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Analysis dependencies not available. Install with: pip install dagnostics[llm]",
        )
    except Exception as e:
        logger.error(f"Legacy analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(
        websocket,
        {
            "connected_at": datetime.now().isoformat(),
            "user_agent": websocket.headers.get("user-agent", "unknown"),
        },
    )

    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()

            try:
                message = (
                    json.loads(data)
                    if data.startswith("{")
                    else {"type": "message", "data": data}
                )

                # Handle different message types
                if message.get("type") == "ping":
                    await websocket_manager.send_personal_message(
                        websocket,
                        {"type": "pong", "timestamp": datetime.now().isoformat()},
                    )
                else:
                    # Echo other messages
                    await websocket_manager.send_personal_message(
                        websocket,
                        {
                            "type": "echo",
                            "data": message,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

            except json.JSONDecodeError:
                await websocket_manager.send_personal_message(
                    websocket,
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat(),
                    },
                )

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.get("/api/v1/websocket/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return websocket_manager.get_connection_stats()


# Utility function to broadcast real-time updates
async def broadcast_update(update: RealTimeUpdate):
    """Broadcast real-time update to all connected clients"""
    await websocket_manager.broadcast_update(update)
