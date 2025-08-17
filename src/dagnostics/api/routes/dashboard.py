import logging
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from dagnostics.api.schemas import (
    DashboardStats,
    ErrorTrendData,
    FailureTimelineItem,
    RecentFailureItem,
)
from dagnostics.core.database import get_db_session


# Mock models since we don't have the actual database models yet
class MockAnalysisResult:
    pass


class MockErrorCategory:
    pass


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    days: int = Query(7, ge=1, le=30), _db=Depends(get_db_session)
):
    """Get comprehensive dashboard statistics"""
    try:
        # end_date = datetime.now()  # Unused for now
        # start_date = end_date - timedelta(days=days)  # Unused for now

        # Mock data for now
        total_failures = 15

        # Mock data for now
        today_failures = 5

        # Mock data for now
        avg_processing_time = 2.3

        # Mock categories data
        categories = {
            "resource_error": 5,
            "data_quality": 3,
            "dependency_failure": 4,
            "timeout_error": 2,
            "unknown": 1,
        }

        # Get resolution rate (mock for now - would need more data model)
        resolution_rate = 0.75  # 75% placeholder

        # Mock top failing DAGs
        top_dags = [
            {"dag_id": "etl_pipeline_1", "failures": 5},
            {"dag_id": "data_ingestion", "failures": 3},
            {"dag_id": "reporting_job", "failures": 2},
        ]

        return DashboardStats(
            total_failures=total_failures,
            today_failures=today_failures,
            average_processing_time=float(avg_processing_time),
            resolution_rate=resolution_rate,
            error_categories=categories,
            top_failing_dags=top_dags,
            period_days=days,
        )

    except Exception as e:
        logger.error(f"Failed to get dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends", response_model=List[ErrorTrendData])
async def get_error_trends(
    days: int = Query(7, ge=1, le=30), _db=Depends(get_db_session)
):
    """Get error trends over time"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Generate mock daily trends
        trends = []
        mock_failures = [3, 7, 5, 12, 8, 6, 4]  # Mock data for last 7 days

        for i in range(days):
            day_start = start_date + timedelta(days=i)
            daily_failures = mock_failures[i % len(mock_failures)]

            trends.append(
                ErrorTrendData(
                    date=day_start.strftime("%Y-%m-%d"),
                    failures=daily_failures,
                    resolved=int(daily_failures * 0.7),  # Mock resolution data
                )
            )

        return trends

    except Exception as e:
        logger.error(f"Failed to get error trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent-failures", response_model=List[RecentFailureItem])
async def get_recent_failures(
    limit: int = Query(10, ge=1, le=50), _db=Depends(get_db_session)
):
    """Get recent failure items"""
    try:
        # Mock recent failures data
        items = [
            RecentFailureItem(
                dag_id="etl_pipeline_1",
                task_id="extract_data",
                run_id="manual__2025-08-14T10:30:00",
                timestamp=datetime.now() - timedelta(minutes=30),
                error_message="Connection timeout to database server",
                category="resource_error",
                severity="high",
            ),
            RecentFailureItem(
                dag_id="data_ingestion",
                task_id="validate_schema",
                run_id="scheduled__2025-08-14T09:00:00",
                timestamp=datetime.now() - timedelta(hours=1),
                error_message="Schema validation failed: missing required column 'id'",
                category="data_quality",
                severity="medium",
            ),
            RecentFailureItem(
                dag_id="reporting_job",
                task_id="generate_report",
                run_id="manual__2025-08-14T08:15:00",
                timestamp=datetime.now() - timedelta(hours=2),
                error_message="Insufficient memory to process large dataset",
                category="resource_error",
                severity="critical",
            ),
        ][:limit]

        return items

    except Exception as e:
        logger.error(f"Failed to get recent failures: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline", response_model=List[FailureTimelineItem])
async def get_failure_timeline(
    _hours: int = Query(24, ge=1, le=168), _db=Depends(get_db_session)
):
    """Get failure timeline for the last N hours"""
    try:
        # end_time = datetime.now()  # Unused for now
        # start_time = end_time - timedelta(hours=hours)  # Unused for now

        # Mock timeline data
        timeline_items = [
            FailureTimelineItem(
                timestamp=datetime.now() - timedelta(minutes=30),
                dag_id="etl_pipeline_1",
                task_id="extract_data",
                severity="high",
                category="resource_error",
            ),
            FailureTimelineItem(
                timestamp=datetime.now() - timedelta(hours=1),
                dag_id="data_ingestion",
                task_id="validate_schema",
                severity="medium",
                category="data_quality",
            ),
        ]

        return timeline_items

    except Exception as e:
        logger.error(f"Failed to get failure timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-score")
async def get_health_score(_db=Depends(get_db_session)):
    """Calculate overall system health score"""
    try:
        # Mock health calculation data
        recent_failures = 5
        week_failures = 30

        # Calculate health score (0-100)
        daily_avg = week_failures / 7
        if daily_avg == 0:
            health_score = 100
        else:
            ratio = recent_failures / daily_avg
            health_score = max(0, 100 - (ratio * 20))  # Penalize higher than average

        return {
            "health_score": int(health_score),
            "status": (
                "excellent"
                if health_score >= 90
                else (
                    "good"
                    if health_score >= 70
                    else "warning" if health_score >= 50 else "critical"
                )
            ),
            "recent_failures": recent_failures,
            "trend": (
                "improving"
                if recent_failures < daily_avg
                else "stable" if recent_failures == daily_avg else "degrading"
            ),
        }

    except Exception as e:
        logger.error(f"Failed to calculate health score: {e}")
        raise HTTPException(status_code=500, detail=str(e))
