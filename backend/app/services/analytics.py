"""Analytics service with MySQL database persistence."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, Optional
from sqlalchemy import func
from sqlalchemy.orm import Session
import logging

from app.database import Prediction, SessionLocal, check_db_connection

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Analytics service with MySQL database storage."""

    def __init__(self) -> None:
        self.db_available = check_db_connection()
        if not self.db_available:
            logger.warning("Database connection not available. Analytics will not persist.")

    def record_prediction(self, predicted_class: str, confidence: float) -> None:
        """Record one successful prediction to MySQL database."""
        if not self.db_available:
            logger.warning("Database unavailable. Prediction not recorded.")
            return

        try:
            db: Session = SessionLocal()
            try:
                prediction = Prediction(
                    predicted_class=predicted_class,
                    confidence=float(confidence),
                    timestamp=datetime.utcnow()
                )
                db.add(prediction)
                db.commit()
                logger.info(f"Prediction recorded: {predicted_class} ({confidence:.2f})")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")

    def get_summary(self) -> Dict[str, Optional[float]]:
        """Aggregate total predictions, average confidence and latest timestamp from database."""
        if not self.db_available:
            return {
                "total_predictions": 0,
                "avg_confidence": 0.0,
                "last_prediction_time": None,
            }

        try:
            db: Session = SessionLocal()
            try:
                total_predictions = db.query(func.count(Prediction.id)).scalar() or 0
                
                if total_predictions == 0:
                    return {
                        "total_predictions": 0,
                        "avg_confidence": 0.0,
                        "last_prediction_time": None,
                    }

                avg_confidence = db.query(func.avg(Prediction.confidence)).scalar() or 0.0
                
                last_prediction = db.query(Prediction).order_by(Prediction.timestamp.desc()).first()
                last_prediction_time = last_prediction.timestamp.isoformat() if last_prediction else None

                return {
                    "total_predictions": total_predictions,
                    "avg_confidence": float(avg_confidence),
                    "last_prediction_time": last_prediction_time,
                }
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return {
                "total_predictions": 0,
                "avg_confidence": 0.0,
                "last_prediction_time": None,
            }

    def get_class_distribution(self) -> Dict[str, int]:
        """Return class frequency map from database."""
        if not self.db_available:
            return {}

        try:
            db: Session = SessionLocal()
            try:
                results = db.query(
                    Prediction.predicted_class,
                    func.count(Prediction.id).label('count')
                ).group_by(Prediction.predicted_class).all()

                return {row.predicted_class: row.count for row in results}
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to get class distribution: {e}")
            return {}


analytics_service = AnalyticsService()
