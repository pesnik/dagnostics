#!/usr/bin/env python3
"""
DAGnostics Web Dashboard Application

This module provides a complete web dashboard for DAGnostics with:
- Real-time monitoring
- Interactive analysis
- WebSocket updates
- RESTful API
"""

import logging

import uvicorn

from dagnostics.core.config import load_config

logger = logging.getLogger(__name__)


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
):
    """Run the DAGnostics web dashboard"""

    # Load configuration
    try:
        load_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load configuration: {e}")
        logger.info("Using default configuration")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting DAGnostics Web Dashboard on {host}:{port}")
    logger.info("Dashboard features:")
    logger.info("   Real-time monitoring")
    logger.info("   Interactive analysis")
    logger.info("   WebSocket updates")
    logger.info("   RESTful API")
    logger.info("   Error trends and analytics")

    # Run the server
    uvicorn.run(
        "dagnostics.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DAGnostics Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    run_dashboard(
        host=args.host, port=args.port, reload=args.reload, log_level=args.log_level
    )
