"""Decision engine that combines model, risk, anomaly, sentiment, and explainability signals."""

from __future__ import annotations

from typing import List

from src.schemas import (
    AnomalyOutput,
    DecisionOutput,
    ExplanationOutput,
    ForecastOutput,
    RiskOutput,
    SentimentOutput,
)


def _direction_from_return(predicted_return: float) -> str:
    if predicted_return > 0.001:
        return "bullish"
    if predicted_return < -0.001:
        return "bearish"
    return "neutral"


def make_decision(
    forecast: ForecastOutput,
    risk: RiskOutput,
    anomaly: AnomalyOutput,
    sentiment: SentimentOutput,
    explanation: ExplanationOutput,
) -> DecisionOutput:
    """Produce final business-style recommendation summary."""
    direction = _direction_from_return(forecast.predicted_return)
    anomaly_status = "flagged" if anomaly.is_anomaly else "not flagged"

    top_drivers = [
        str(item.get("feature", "unknown"))
        for item in explanation.top_features[:5]
        if "feature" in item
    ]

    caution_notes: List[str] = []
    if risk.risk_level in {"medium", "high"}:
        caution_notes.append("Elevated volatility/drawdown profile may increase forecast uncertainty.")
    if anomaly.is_anomaly:
        caution_notes.append("Current market behavior is unusual relative to recent patterns.")
    if sentiment.available and sentiment.sentiment_label == "negative":
        caution_notes.append("Recent sentiment is negative and may pressure short-term performance.")
    if not caution_notes:
        caution_notes.append("No major cautionary flags beyond normal market uncertainty.")

    sentiment_summary = (
        f"{sentiment.sentiment_label} ({sentiment.sentiment_score:.2f})"
        if sentiment.available
        else "unavailable"
    )

    recommendation_summary = (
        f"The model indicates a {direction} next-day return signal "
        f"({forecast.predicted_return:.4f}). "
        f"Risk is {risk.risk_level} and anomaly status is {anomaly_status}. "
        f"Sentiment is {sentiment_summary}. "
        "Use this as risk-aware decision support, not a trading instruction."
    )

    return DecisionOutput(
        ticker=forecast.ticker,
        latest_predicted_return=forecast.predicted_return,
        direction=direction,
        risk_level=risk.risk_level,
        anomaly_status=anomaly_status,
        sentiment_summary=sentiment_summary,
        top_drivers=top_drivers,
        recommendation_summary=recommendation_summary,
        caution_notes=caution_notes,
    )
