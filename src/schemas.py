"""Typed schemas used by agents and pipeline outputs."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelPerformance(BaseModel):
    """Regression and directional metrics for a model."""

    model_config = ConfigDict(protected_namespaces=())

    rmse: float
    mae: float
    r2: float
    directional_accuracy: float


class ForecastOutput(BaseModel):
    """Latest return forecast and model context."""

    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    predicted_return: float
    model_name: str
    confidence_context: str
    metrics: Dict[str, ModelPerformance]


class RiskOutput(BaseModel):
    """Risk scoring output from volatility and drawdown."""

    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    risk_score: float = Field(ge=0.0, le=1.0)
    volatility_20d: float
    drawdown_20d: float
    risk_level: str


class AnomalyOutput(BaseModel):
    """Anomaly detector output for latest observation."""

    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    is_anomaly: bool
    anomaly_score: float
    recent_anomaly_rate: float


class SentimentOutput(BaseModel):
    """Aggregated sentiment signal from financial headlines."""

    model_config = ConfigDict(protected_namespaces=())

    available: bool
    source: str
    sentiment_label: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    headline_count: int


class ExplanationOutput(BaseModel):
    """Top explainability drivers for the latest prediction."""

    model_config = ConfigDict(protected_namespaces=())

    available: bool
    model_type: str
    top_features: List[Dict[str, float]]
    latest_expected_value: Optional[float] = None


class DecisionOutput(BaseModel):
    """Final decision-support summary."""

    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    latest_predicted_return: float
    direction: str
    risk_level: str
    anomaly_status: str
    sentiment_summary: str
    top_drivers: List[str]
    recommendation_summary: str
    caution_notes: List[str]
