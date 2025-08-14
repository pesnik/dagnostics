"""
User Feedback Collection for Fine-tuning

Web interface for collecting user corrections and ratings on LLM error analysis.
This feedback is used to improve the fine-tuned model accuracy over time.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dagnostics.core.models import ErrorAnalysis, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

# Feedback storage path
FEEDBACK_FILE = Path("data/feedback_data.jsonl")


class FeedbackCorrection(BaseModel):
    """User correction for an error analysis"""

    error_message: str
    confidence: float
    category: ErrorCategory
    severity: ErrorSeverity
    reasoning: str
    error_lines: List[str]


class FeedbackSubmission(BaseModel):
    """Complete feedback submission"""

    log_context: str
    dag_id: str
    task_id: str
    original_analysis: ErrorAnalysis
    corrected_analysis: FeedbackCorrection
    user_rating: int  # 1-5 stars
    user_id: str
    comments: Optional[str] = None


class FeedbackStats(BaseModel):
    """Feedback collection statistics"""

    total_feedback_count: int
    avg_user_rating: float
    category_distribution: Dict[str, int]
    recent_feedback_count: int  # Last 7 days


router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackCollector:
    """Collect and store user feedback for model improvement"""

    def __init__(self, feedback_file: Path = FEEDBACK_FILE):
        self.feedback_file = feedback_file
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

    def save_feedback(self, feedback: FeedbackSubmission) -> bool:
        """Save user feedback to JSONL file"""

        try:
            # Convert to storage format
            feedback_record = {
                "log_context": feedback.log_context,
                "dag_id": feedback.dag_id,
                "task_id": feedback.task_id,
                "original_analysis": feedback.original_analysis.model_dump(),
                "corrected_analysis": feedback.corrected_analysis.model_dump(),
                "user_rating": feedback.user_rating,
                "user_id": feedback.user_id,
                "comments": feedback.comments,
                "timestamp": datetime.now().isoformat(),
                "feedback_id": f"{feedback.dag_id}_{feedback.task_id}_{int(datetime.now().timestamp())}",
            }

            # Append to JSONL file
            with open(self.feedback_file, "a") as f:
                f.write(json.dumps(feedback_record) + "\n")

            logger.info(
                f"Feedback saved for {feedback.dag_id}.{feedback.task_id} by {feedback.user_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False

    def load_all_feedback(self) -> List[Dict]:
        """Load all feedback records"""

        feedback_records = []
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, "r") as f:
                    for line in f:
                        feedback_records.append(json.loads(line.strip()))
            except Exception as e:
                logger.error(f"Failed to load feedback: {e}")

        return feedback_records

    def get_feedback_stats(self) -> FeedbackStats:
        """Get statistics about collected feedback"""

        feedback_records = self.load_all_feedback()

        if not feedback_records:
            return FeedbackStats(
                total_feedback_count=0,
                avg_user_rating=0.0,
                category_distribution={},
                recent_feedback_count=0,
            )

        # Calculate statistics
        total_count = len(feedback_records)
        avg_rating = (
            sum(record["user_rating"] for record in feedback_records) / total_count
        )

        # Category distribution
        category_dist: dict[str, int] = {}
        for record in feedback_records:
            category = record["corrected_analysis"]["category"]
            category_dist[category] = category_dist.get(category, 0) + 1

        # Recent feedback (last 7 days)
        from datetime import timedelta

        week_ago = datetime.now() - timedelta(days=7)
        recent_count = sum(
            1
            for record in feedback_records
            if datetime.fromisoformat(record["timestamp"]) > week_ago
        )

        return FeedbackStats(
            total_feedback_count=total_count,
            avg_user_rating=round(avg_rating, 2),
            category_distribution=category_dist,
            recent_feedback_count=recent_count,
        )

    def export_for_training(self, min_rating: int = 3) -> str:
        """Export high-quality feedback for training dataset"""

        feedback_records = self.load_all_feedback()

        # Filter high-quality feedback
        quality_feedback = [
            record for record in feedback_records if record["user_rating"] >= min_rating
        ]

        # Export path
        export_path = Path("data/training/feedback_export.jsonl")
        export_path.parent.mkdir(parents=True, exist_ok=True)

        with open(export_path, "w") as f:
            for record in quality_feedback:
                f.write(json.dumps(record) + "\n")

        logger.info(
            f"Exported {len(quality_feedback)} quality feedback records to {export_path}"
        )
        return str(export_path)


# Global feedback collector instance
feedback_collector = FeedbackCollector()


@router.post("/submit")
async def submit_feedback(feedback: FeedbackSubmission):
    """Submit user feedback for an error analysis"""

    success = feedback_collector.save_feedback(feedback)

    if success:
        return {"status": "success", "message": "Feedback submitted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save feedback")


@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats():
    """Get feedback collection statistics"""

    return feedback_collector.get_feedback_stats()


@router.post("/export")
async def export_feedback_for_training(min_rating: int = 3):
    """Export feedback for training dataset generation"""

    try:
        export_path = feedback_collector.export_for_training(min_rating)
        return {
            "status": "success",
            "export_path": export_path,
            "message": f"Feedback exported for training (min rating: {min_rating})",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/recent")
async def get_recent_feedback(limit: int = 10):
    """Get recent feedback submissions"""

    feedback_records = feedback_collector.load_all_feedback()

    # Sort by timestamp, most recent first
    sorted_feedback = sorted(
        feedback_records, key=lambda x: x["timestamp"], reverse=True
    )

    return {"feedback": sorted_feedback[:limit], "total_count": len(feedback_records)}


@router.get("/user/{user_id}")
async def get_user_feedback(user_id: str):
    """Get feedback submissions by specific user"""

    feedback_records = feedback_collector.load_all_feedback()

    user_feedback = [
        record for record in feedback_records if record["user_id"] == user_id
    ]

    return {
        "user_id": user_id,
        "feedback_count": len(user_feedback),
        "feedback": user_feedback,
    }


# HTML templates for feedback interface (simplified)
FEEDBACK_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DAGnostics - Error Analysis Feedback</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .log-context { background: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace; }
        .analysis-section { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .rating { margin: 10px 0; }
        .rating input[type="radio"] { margin: 0 5px; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        textarea, input, select { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>DAGnostics Error Analysis Feedback</h1>
        <p>Help improve our error analysis by reviewing and correcting the AI analysis below.</p>

        <div id="feedback-form">
            <!-- Log Context -->
            <h3>Log Context</h3>
            <div class="log-context" id="log-context">
                <!-- Log will be populated here -->
            </div>

            <!-- Original Analysis -->
            <div class="analysis-section">
                <h3>AI Analysis</h3>
                <div id="original-analysis">
                    <!-- Original analysis will be populated here -->
                </div>
            </div>

            <!-- Correction Form -->
            <div class="analysis-section">
                <h3>Your Corrections</h3>
                <form id="correction-form">
                    <label>Error Message:</label>
                    <textarea id="error-message" rows="3" placeholder="Corrected error message"></textarea>

                    <label>Category:</label>
                    <select id="category">
                        <option value="configuration_error">Configuration Error</option>
                        <option value="timeout_error">Timeout Error</option>
                        <option value="data_quality">Data Quality</option>
                        <option value="dependency_failure">Dependency Failure</option>
                        <option value="resource_error">Resource Error</option>
                        <option value="permission_error">Permission Error</option>
                        <option value="unknown">Unknown</option>
                    </select>

                    <label>Severity:</label>
                    <select id="severity">
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                        <option value="critical">Critical</option>
                    </select>

                    <label>Confidence (0.0 - 1.0):</label>
                    <input type="number" id="confidence" min="0" max="1" step="0.1" value="0.8">

                    <label>Reasoning:</label>
                    <textarea id="reasoning" rows="2" placeholder="Brief explanation of the error"></textarea>

                    <div class="rating">
                        <label>Rate the AI Analysis Quality:</label><br>
                        <input type="radio" name="rating" value="1"> 1 (Poor)
                        <input type="radio" name="rating" value="2"> 2 (Fair)
                        <input type="radio" name="rating" value="3"> 3 (Good)
                        <input type="radio" name="rating" value="4"> 4 (Very Good)
                        <input type="radio" name="rating" value="5"> 5 (Excellent)
                    </div>

                    <label>Additional Comments (optional):</label>
                    <textarea id="comments" rows="2" placeholder="Any additional feedback"></textarea>

                    <button type="submit">Submit Feedback</button>
                </form>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('correction-form').addEventListener('submit', function(e) {
            e.preventDefault();

            // Collect form data
            const feedback = {
                error_message: document.getElementById('error-message').value,
                category: document.getElementById('category').value,
                severity: document.getElementById('severity').value,
                confidence: parseFloat(document.getElementById('confidence').value),
                reasoning: document.getElementById('reasoning').value,
                user_rating: parseInt(document.querySelector('input[name="rating"]:checked')?.value || 3),
                comments: document.getElementById('comments').value
            };

            // Submit via API (implementation depends on your setup)
            console.log('Submitting feedback:', feedback);
            alert('Feedback submitted! Thank you for helping improve DAGnostics.');
        });
    </script>
</body>
</html>
"""


@router.get("/interface")
async def get_feedback_interface():
    """Serve feedback interface HTML"""
    from fastapi.responses import HTMLResponse

    return HTMLResponse(content=FEEDBACK_HTML_TEMPLATE)
