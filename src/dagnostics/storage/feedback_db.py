"""
Robust Feedback Storage System

SQLite-based storage for error analysis, feedback, and training data
with proper indexing, querying, and data management.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from dagnostics.core.config import load_config
from dagnostics.web.feedback import FeedbackSubmission


class FeedbackDatabase:
    """SQLite-based feedback storage with comprehensive querying"""

    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = load_config(config_path)
        self.feedback_config = self.config.feedback

        # Setup database path from config
        db_url = self.feedback_config.storage.database_url

        # Handle environment variable expansion (basic implementation)
        if "${" in db_url:
            # Simple environment variable expansion
            if "DAGNOSTICS_FEEDBACK_DB_URL:-" in db_url:
                default_value = db_url.split(":-")[1].rstrip("}")
                env_var = db_url.split("${")[1].split(":-")[0]
                db_url = os.getenv(env_var, default_value)

        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
        else:
            # For other databases like PostgreSQL/MySQL, would need different handling
            raise ValueError(f"Database type not supported yet: {db_url}")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup other configured paths
        self.jsonl_file = Path(self.feedback_config.storage.jsonl_file)
        self.training_data_dir = Path(self.feedback_config.storage.training_data_dir)
        self.raw_logs_dir = Path(self.feedback_config.storage.raw_logs_dir)

        # Create directories
        self.jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_logs_dir.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize database tables and indexes"""

        with sqlite3.connect(self.db_path) as conn:
            # Error logs table (raw error data from Airflow/ETL systems)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,  -- YYYY-MM-DD for easy querying
                    dag_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    run_id TEXT,
                    error_message TEXT NOT NULL,
                    log_context TEXT,  -- Full log context
                    error_category TEXT,
                    severity TEXT,
                    resolved BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # AI Analysis table (LLM analysis results)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_log_id INTEGER NOT NULL,
                    analysis_timestamp TEXT NOT NULL,
                    extracted_error TEXT,
                    predicted_category TEXT,
                    predicted_severity TEXT,
                    confidence_score REAL,
                    reasoning TEXT,
                    model_version TEXT,
                    analysis_metadata TEXT,  -- JSON metadata
                    FOREIGN KEY (error_log_id) REFERENCES error_logs (id)
                )
            """
            )

            # User Feedback table (human corrections and ratings)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_log_id INTEGER NOT NULL,
                    ai_analysis_id INTEGER,
                    user_id TEXT NOT NULL,
                    feedback_timestamp TEXT NOT NULL,
                    corrected_error TEXT,
                    corrected_category TEXT,
                    corrected_severity TEXT,
                    corrected_confidence REAL,
                    corrected_reasoning TEXT,
                    user_rating INTEGER,  -- 1-5 rating of AI analysis
                    comments TEXT,
                    feedback_metadata TEXT,  -- JSON metadata
                    FOREIGN KEY (error_log_id) REFERENCES error_logs (id),
                    FOREIGN KEY (ai_analysis_id) REFERENCES ai_analysis (id)
                )
            """
            )

            # Training Sessions table (track model training runs)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_timestamp TEXT NOT NULL,
                    model_name TEXT,
                    training_data_size INTEGER,
                    validation_data_size INTEGER,
                    hyperparameters TEXT,  -- JSON
                    training_metrics TEXT,  -- JSON
                    model_path TEXT,
                    session_metadata TEXT  -- JSON
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_error_logs_date ON error_logs(date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_error_logs_dag_task ON error_logs(dag_id, task_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_analysis_error_id ON ai_analysis(error_log_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_feedback_error_id ON user_feedback(error_log_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp ON user_feedback(feedback_timestamp)"
            )

    def store_error_log(
        self,
        dag_id: str,
        task_id: str,
        run_id: str,
        error_message: str,
        log_context: str,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Store raw error log entry"""

        if timestamp is None:
            timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO error_logs
                (timestamp, date, dag_id, task_id, run_id, error_message, log_context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp.isoformat(),
                    timestamp.strftime("%Y-%m-%d"),
                    dag_id,
                    task_id,
                    run_id,
                    error_message,
                    log_context,
                ),
            )
            return cursor.lastrowid or 0

    def store_ai_analysis(
        self,
        error_log_id: int,
        extracted_error: str,
        category: str,
        severity: str,
        confidence: float,
        reasoning: str,
        model_version: str = "v1.0",
    ) -> int:
        """Store AI analysis results"""

        analysis_timestamp = datetime.now().isoformat()
        metadata = json.dumps({"model_version": model_version})

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO ai_analysis
                (error_log_id, analysis_timestamp, extracted_error, predicted_category,
                 predicted_severity, confidence_score, reasoning, model_version, analysis_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    error_log_id,
                    analysis_timestamp,
                    extracted_error,
                    category,
                    severity,
                    confidence,
                    reasoning,
                    model_version,
                    metadata,
                ),
            )
            return cursor.lastrowid or 0

    def store_user_feedback(
        self,
        feedback: FeedbackSubmission,
        error_log_id: int,
        ai_analysis_id: Optional[int] = None,
    ) -> int:
        """Store user feedback"""

        feedback_timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO user_feedback
                (error_log_id, ai_analysis_id, user_id, feedback_timestamp,
                 corrected_error, corrected_category, corrected_severity,
                 corrected_confidence, corrected_reasoning, user_rating, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    error_log_id,
                    ai_analysis_id,
                    feedback.user_id,
                    feedback_timestamp,
                    feedback.corrected_analysis.error_message,
                    feedback.corrected_analysis.category,
                    feedback.corrected_analysis.severity,
                    feedback.corrected_analysis.confidence,
                    feedback.corrected_analysis.reasoning,
                    feedback.user_rating,
                    feedback.comments,
                ),
            )
            return cursor.lastrowid or 0

    def get_daily_errors(
        self,
        date: datetime,
        _include_analysis: bool = True,
        _include_feedback: bool = True,
    ) -> List[Dict]:
        """Get all errors for a specific date with analysis and feedback"""

        date_str = date.strftime("%Y-%m-%d")

        query = """
            SELECT e.*,
                   a.extracted_error, a.predicted_category, a.predicted_severity,
                   a.confidence_score, a.reasoning as ai_reasoning,
                   f.corrected_error, f.corrected_category, f.corrected_severity,
                   f.user_rating, f.comments
            FROM error_logs e
            LEFT JOIN ai_analysis a ON e.id = a.error_log_id
            LEFT JOIN user_feedback f ON e.id = f.error_log_id
            WHERE e.date = ?
            ORDER BY e.timestamp DESC
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (date_str,)).fetchall()

            errors = []
            for row in rows:
                error_dict = dict(row)
                error_dict["has_analysis"] = bool(row["extracted_error"])
                error_dict["has_feedback"] = bool(row["corrected_error"])
                errors.append(error_dict)

            return errors

    def get_feedback_candidates(
        self, days: int = 7, require_analysis: bool = True, max_results: int = 50
    ) -> List[Dict]:
        """Get errors that need feedback (have analysis but no feedback)"""

        since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        if require_analysis:
            query = """
                SELECT e.*, a.extracted_error, a.predicted_category,
                       a.predicted_severity, a.confidence_score, a.reasoning
                FROM error_logs e
                INNER JOIN ai_analysis a ON e.id = a.error_log_id
                LEFT JOIN user_feedback f ON e.id = f.error_log_id
                WHERE e.date >= ? AND f.id IS NULL
                ORDER BY e.timestamp DESC
                LIMIT ?
            """
        else:
            query = """
                SELECT e.*, a.extracted_error, a.predicted_category,
                       a.predicted_severity, a.confidence_score, a.reasoning
                FROM error_logs e
                LEFT JOIN ai_analysis a ON e.id = a.error_log_id
                LEFT JOIN user_feedback f ON e.id = f.error_log_id
                WHERE e.date >= ? AND f.id IS NULL
                ORDER BY e.timestamp DESC
                LIMIT ?
            """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (since_date, max_results)).fetchall()
            return [dict(row) for row in rows]

    def get_feedback_stats(self, days: int = 30) -> Dict:
        """Get comprehensive feedback statistics"""

        since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            # Total feedback count
            total_feedback = conn.execute(
                """
                SELECT COUNT(*) FROM user_feedback
                WHERE feedback_timestamp >= ?
            """,
                (since_date,),
            ).fetchone()[0]

            # Average rating
            avg_rating = (
                conn.execute(
                    """
                SELECT AVG(CAST(user_rating AS REAL)) FROM user_feedback
                WHERE feedback_timestamp >= ? AND user_rating IS NOT NULL
            """,
                    (since_date,),
                ).fetchone()[0]
                or 0.0
            )

            # Category distribution
            category_dist = conn.execute(
                """
                SELECT corrected_category, COUNT(*) as count
                FROM user_feedback
                WHERE feedback_timestamp >= ? AND corrected_category IS NOT NULL
                GROUP BY corrected_category
                ORDER BY count DESC
            """,
                (since_date,),
            ).fetchall()

            # Recent feedback (last 7 days)
            recent_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            recent_feedback = conn.execute(
                """
                SELECT COUNT(*) FROM user_feedback
                WHERE feedback_timestamp >= ?
            """,
                (recent_date,),
            ).fetchone()[0]

            return {
                "total_feedback_count": total_feedback,
                "avg_user_rating": round(avg_rating, 2),
                "category_distribution": {cat: count for cat, count in category_dist},
                "recent_feedback_count": recent_feedback,
            }

    def export_training_data(
        self, min_rating: Optional[int] = None, output_file: Optional[str] = None
    ) -> int:
        """Export quality feedback for training"""

        # Use configured defaults if not specified
        if min_rating is None:
            min_rating = self.feedback_config.collection.min_quality_rating

        if output_file is None:
            output_file = str(self.training_data_dir / "feedback_export.jsonl")

        query = """
            SELECT e.log_context, e.error_message, e.dag_id, e.task_id,
                   a.extracted_error as original_analysis,
                   f.corrected_error, f.corrected_category, f.corrected_severity,
                   f.corrected_reasoning, f.user_rating
            FROM user_feedback f
            INNER JOIN error_logs e ON f.error_log_id = e.id
            LEFT JOIN ai_analysis a ON f.ai_analysis_id = a.id
            WHERE f.user_rating >= ?
            ORDER BY f.feedback_timestamp DESC
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (min_rating,)).fetchall()

            export_count = 0
            with open(output_file, "w") as f:
                for row in rows:
                    training_example = {
                        "instruction": "Analyze the Airflow ETL error and extract the root cause in JSON format.",
                        "input": row["log_context"],
                        "output": json.dumps(
                            {
                                "error_message": row["corrected_error"],
                                "category": row["corrected_category"],
                                "severity": row["corrected_severity"],
                                "confidence": 0.9,  # High confidence for human-corrected data
                                "reasoning": row["corrected_reasoning"],
                            }
                        ),
                        "metadata": {
                            "dag_id": row["dag_id"],
                            "task_id": row["task_id"],
                            "user_rating": row["user_rating"],
                            "source": "user_feedback",
                        },
                    }

                    f.write(json.dumps(training_example) + "\n")
                    export_count += 1

            return export_count

    def get_training_progress(self) -> Dict:
        """Get training data collection progress"""

        with sqlite3.connect(self.db_path) as conn:
            # Total errors logged
            total_errors = conn.execute("SELECT COUNT(*) FROM error_logs").fetchone()[0]

            # Errors with analysis
            analyzed_errors = conn.execute(
                """
                SELECT COUNT(DISTINCT error_log_id) FROM ai_analysis
            """
            ).fetchone()[0]

            # Errors with feedback
            feedback_errors = conn.execute(
                """
                SELECT COUNT(DISTINCT error_log_id) FROM user_feedback
            """
            ).fetchone()[0]

            # Quality feedback (rating >= 3)
            quality_feedback = conn.execute(
                """
                SELECT COUNT(*) FROM user_feedback WHERE user_rating >= 3
            """
            ).fetchone()[0]

            return {
                "total_errors": total_errors,
                "analyzed_errors": analyzed_errors,
                "feedback_errors": feedback_errors,
                "quality_feedback": quality_feedback,
                "analysis_rate": (
                    round(analyzed_errors / total_errors * 100, 1)
                    if total_errors > 0
                    else 0
                ),
                "feedback_rate": (
                    round(feedback_errors / total_errors * 100, 1)
                    if total_errors > 0
                    else 0
                ),
            }

    def cleanup_old_data(self, retention_days: Optional[int] = None) -> int:
        """Clean up old data based on retention policy"""

        if retention_days is None:
            retention_days = self.feedback_config.storage.retention_days

        if not self.feedback_config.storage.cleanup_enabled:
            return 0

        cutoff_date = (datetime.now() - timedelta(days=retention_days)).strftime(
            "%Y-%m-%d"
        )

        with sqlite3.connect(self.db_path) as conn:
            # Delete old error logs (cascading will handle related records)
            cursor = conn.execute(
                "DELETE FROM error_logs WHERE date < ?", (cutoff_date,)
            )
            deleted_count = cursor.rowcount

            # Vacuum to reclaim space
            conn.execute("VACUUM")

            return deleted_count

    def backup_data(self, backup_dir: Optional[str] = None) -> str:
        """Create backup of feedback database"""

        if backup_dir is None:
            backup_dir = self.feedback_config.storage.backup.backup_dir

        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"feedback_backup_{timestamp}.db"

        # Copy database
        import shutil

        shutil.copy2(self.db_path, backup_file)

        # Cleanup old backups
        max_backups = self.feedback_config.storage.backup.max_backups
        backup_files = sorted(backup_path.glob("feedback_backup_*.db"))

        if len(backup_files) > max_backups:
            for old_backup in backup_files[:-max_backups]:
                old_backup.unlink()

        return str(backup_file)

    def get_configured_categories(self) -> List[str]:
        """Get error categories from configuration"""
        return self.feedback_config.collection.categories

    def get_configured_severities(self) -> List[str]:
        """Get severity levels from configuration"""
        return self.feedback_config.collection.severities

    def should_auto_export(self) -> bool:
        """Check if automatic export threshold has been reached"""
        recent_feedback = self.get_feedback_stats(days=7)["recent_feedback_count"]
        threshold = self.feedback_config.collection.auto_export_threshold
        return recent_feedback >= threshold

    def get_storage_info(self) -> Dict:
        """Get information about storage configuration and usage"""

        # Database file size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        # Backup info
        backup_dir = Path(self.feedback_config.storage.backup.backup_dir)
        backup_count = (
            len(list(backup_dir.glob("feedback_backup_*.db")))
            if backup_dir.exists()
            else 0
        )

        return {
            "database_path": str(self.db_path),
            "database_size_mb": round(db_size / 1024 / 1024, 2),
            "jsonl_backup_path": str(self.jsonl_file),
            "training_data_dir": str(self.training_data_dir),
            "backup_count": backup_count,
            "backup_enabled": self.feedback_config.storage.backup.enabled,
            "cleanup_enabled": self.feedback_config.storage.cleanup_enabled,
            "retention_days": self.feedback_config.storage.retention_days,
        }


# Convenience functions for backward compatibility
def get_feedback_db(config_path: Optional[str] = None) -> FeedbackDatabase:
    """Get feedback database instance with configuration"""
    return FeedbackDatabase(config_path)
