from pydantic import BaseModel
from typing import Dict, Optional


class PredictionResponse(BaseModel):
    """Response schema for waste classification prediction."""
    predicted_class: str
    probabilities: Dict[str, float]


class GradCamResponse(BaseModel):
    """Response schema for transformer attention visualization."""

    predicted_class: str
    confidence: float
    heatmap: str


class StatsSummaryResponse(BaseModel):
    """Response schema for summary telemetry stats."""

    total_predictions: int
    avg_confidence: float
    last_prediction_time: Optional[str]


class DisposalRuleItem(BaseModel):
    """Single disposal rule with title and description."""
    title: str
    description: str


class DisposalRulesResponse(BaseModel):
    """Response schema for disposal rules."""
    rules: Dict[str, DisposalRuleItem]


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
