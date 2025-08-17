"""
Database utilities and session management for DAGnostics.

This module provides database connection and session management functionality.
"""

import logging
import os
from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# Database engine and session factory
_engine = None
_SessionLocal = None


def get_database_url():
    """Get database URL from environment or use SQLite default"""
    return os.getenv("DATABASE_URL", "sqlite:///./dagnostics.db")


def init_database():
    """Initialize database engine and create tables"""
    global _engine, _SessionLocal

    if _engine is None:
        database_url = get_database_url()
        _engine = create_engine(database_url, echo=False)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

        # Create all tables
        Base.metadata.create_all(bind=_engine)
        logger.info(f"Database initialized with URL: {database_url}")

    return _engine


def get_db_session() -> Generator[Session, None, None]:
    """
    Get database session for dependency injection.
    """
    if _SessionLocal is None:
        init_database()

    if _SessionLocal is not None:
        session = _SessionLocal()
        try:
            yield session
        finally:
            session.close()
    else:
        raise RuntimeError("Failed to initialize database session")


class MockDBSession:
    """Mock database session for development/testing"""

    def query(self, _model):
        """Mock query method"""
        return MockQuery()

    def commit(self):
        """Mock commit method"""
        pass

    def rollback(self):
        """Mock rollback method"""
        pass

    def close(self):
        """Mock close method"""
        pass


class MockQuery:
    """Mock query object"""

    def filter(self, *_args):
        """Mock filter method"""
        return self

    def order_by(self, *_args):
        """Mock order_by method"""
        return self

    def limit(self, _limit):
        """Mock limit method"""
        return self

    def count(self):
        """Mock count method"""
        return 0

    def all(self):
        """Mock all method"""
        return []

    def first(self):
        """Mock first method"""
        return None

    def scalar(self):
        """Mock scalar method"""
        return None


def get_mock_db_session() -> Generator:
    """Get mock database session for development"""
    session = MockDBSession()
    try:
        yield session
    finally:
        session.close()


class ErrorCandidateModel(Base):
    """Database model for training dataset error candidates"""

    __tablename__ = "error_candidates"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    dag_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False)

    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    raw_logs: Mapped[str] = mapped_column(Text, nullable=False)

    # LLM analysis
    llm_category: Mapped[Optional[str]] = mapped_column(String, index=True)
    llm_severity: Mapped[Optional[str]] = mapped_column(String, index=True)
    llm_confidence: Mapped[Optional[float]] = mapped_column(Float)
    llm_reasoning: Mapped[Optional[str]] = mapped_column(Text)

    # Human feedback
    human_category: Mapped[Optional[str]] = mapped_column(String)
    human_severity: Mapped[Optional[str]] = mapped_column(String)
    human_feedback: Mapped[Optional[str]] = mapped_column(Text)

    status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending", index=True
    )  # pending, approved, rejected, modified
    reviewed_by: Mapped[Optional[str]] = mapped_column(String)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, index=True
    )
    processing_time: Mapped[float] = mapped_column(Float, default=0.0)


class AnalysisRecord(Base):
    """Database model for analysis results"""

    __tablename__ = "analysis_records"

    id = Column(String, primary_key=True)
    dag_id = Column(String, nullable=False, index=True)
    task_id = Column(String, nullable=False, index=True)
    run_id = Column(String, nullable=False)

    error_message = Column(Text)
    category = Column(String, index=True)
    severity = Column(String, index=True)
    confidence = Column(Float)

    suggested_actions = Column(JSON)
    processing_time = Column(Float)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    success = Column(Boolean, default=True)

    # LLM analysis details
    llm_reasoning = Column(Text)
    raw_error_lines = Column(JSON)


# class BaselineRecord(Base):
#     """Database model for baseline clusters"""

#     __tablename__ = "baseline_records"

#     id = Column(Integer, primary_key=True)
#     dag_id = Column(String, nullable=False, index=True)
#     task_id = Column(String, nullable=False, index=True)
#     cluster_id = Column(String, nullable=False)
#     template = Column(Text, nullable=False)
#     log_count = Column(Integer, default=0)
#     created_at = Column(DateTime, default=datetime.now)
#     last_updated = Column(DateTime, default=datetime.now)
#     confidence_score = Column(Float, default=0.0)


# class DatabaseManager:
#     """Database operations manager"""

#     def __init__(self, database_url: str):
#         self.engine = create_engine(database_url)
#         self.SessionLocal = sessionmaker(bind=self.engine)
#         Base.metadata.create_all(self.engine)

#     def get_session(self) -> Session:
#         """Get database session"""
#         return self.SessionLocal()

#     def store_analysis_result(self, result: AnalysisResult):
#         """Store analysis result in database"""
#         with self.get_session() as session:
#             record = AnalysisRecord(
#                 id=result.id,
#                 dag_id=result.dag_id,
#                 task_id=result.task_id,
#                 run_id=result.run_id,
#                 error_message=(
#                     result.analysis.error_message if result.analysis else None
#                 ),
#                 category=result.analysis.category.value if result.analysis else None,
#                 severity=result.analysis.severity.value if result.analysis else None,
#                 confidence=result.analysis.confidence if result.analysis else None,
#                 suggested_actions=(
#                     result.analysis.suggested_actions if result.analysis else []
#                 ),
#                 processing_time=result.processing_time,
#                 timestamp=result.timestamp,
#                 success=result.success,
#                 llm_reasoning=(
#                     result.analysis.llm_reasoning if result.analysis else None
#                 ),
#                 raw_error_lines=(
#                     result.analysis.raw_error_lines if result.analysis else []
#                 ),
#             )
#             session.add(record)
#             session.commit()

#     def get_daily_summary(self, date: datetime) -> dict:
#         """Get daily summary statistics"""
#         with self.get_session() as session:
#             # Query analysis records for the day
#             start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
#             end_date = date.replace(hour=23, minute=59, second=59, microsecond=999999)

#             records = (
#                 session.query(AnalysisRecord)
#                 .filter(
#                     AnalysisRecord.timestamp >= start_date,
#                     AnalysisRecord.timestamp <= end_date,
#                     AnalysisRecord.success,
#                 )
#                 .all()
#             )

#             if not records:
#                 return {"total_failures": 0, "categories": {}, "top_failing_dags": []}

#             # Calculate statistics
#             categories: Dict[str, int] = {}
#             dag_failures: Dict[str, int] = {}

#             for record in records:
#                 # Explicitly cast to str to tell Pylance it's a string value
#                 # from the instance, not the Column object.
#                 record_category = cast(str, record.category)
#                 record_dag_id = cast(str, record.dag_id)

#                 if record_category is not None:
#                     categories[record_category] = categories.get(record_category, 0) + 1

#                 dag_failures[record_dag_id] = dag_failures.get(record_dag_id, 0) + 1

#             top_failing_dags = [
#                 {"dag_id": dag_id, "failures": count}
#                 for dag_id, count in sorted(
#                     dag_failures.items(), key=lambda x: x[1], reverse=True
#                 )[:5]
#             ]

#             return {
#                 "total_failures": len(records),
#                 "categories": categories,
#                 "top_failing_dags": top_failing_dags,
#                 "average_processing_time": sum(r.processing_time for r in records)
#                 / len(records),
#             }
