"""Internal lightweight multi-agent layer for decision support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.anomaly_detection import latest_anomaly_summary
from src.decision_engine import make_decision
from src.explainability import build_explanations
from src.schemas import (
    AnomalyOutput,
    DecisionOutput,
    ExplanationOutput,
    ForecastOutput,
    ModelPerformance,
    RiskOutput,
    SentimentOutput,
)
from src.sentiment import SentimentAnalyzer
from src.utils import map_risk_level, normalize_to_unit_interval


@dataclass
class QuantAgent:
    """Wrap forecasting output into a structured agent response."""

    def run(self, ticker: str, forecast_payload: dict) -> ForecastOutput:
        metrics = {
            k: ModelPerformance(**v)
            for k, v in forecast_payload["metrics_by_model"].items()
            if "test" in k or "val" in k
        }

        return ForecastOutput(
            ticker=ticker.upper(),
            predicted_return=float(forecast_payload["latest_prediction"]),
            model_name=forecast_payload["best_model_name"],
            confidence_context=(
                "Confidence is proxied by validation/test RMSE and directional accuracy "
                "from time-aware evaluation."
            ),
            metrics=metrics,
        )


@dataclass
class RiskAgent:
    """Compute interpretable risk score from volatility and drawdown."""

    def run(self, ticker: str, features_df: pd.DataFrame) -> RiskOutput:
        latest = features_df.iloc[-1]
        vol_20 = float(latest.get("volatility_20", 0.0))
        drawdown_20 = abs(float(latest.get("drawdown_20", 0.0)))

        vol_score = normalize_to_unit_interval(vol_20, min_value=0.005, max_value=0.05)
        dd_score = normalize_to_unit_interval(drawdown_20, min_value=0.0, max_value=0.2)
        risk_score = float(0.65 * vol_score + 0.35 * dd_score)

        return RiskOutput(
            ticker=ticker.upper(),
            risk_score=risk_score,
            volatility_20d=vol_20,
            drawdown_20d=float(latest.get("drawdown_20", 0.0)),
            risk_level=map_risk_level(risk_score),
        )


@dataclass
class AnomalyAgent:
    """Convert anomaly detection dataframe into structured output."""

    def run(self, ticker: str, anomaly_df: pd.DataFrame) -> AnomalyOutput:
        summary = latest_anomaly_summary(anomaly_df)
        return AnomalyOutput(
            ticker=ticker.upper(),
            is_anomaly=summary["is_anomaly"],
            anomaly_score=summary["anomaly_score"],
            recent_anomaly_rate=summary["recent_anomaly_rate"],
        )


@dataclass
class SentimentAgent:
    """Produce optional sentiment signal."""

    analyzer: SentimentAnalyzer

    def run(self, headlines: List[str], source: str = "mock_headlines") -> SentimentOutput:
        return self.analyzer.aggregate(headlines=headlines, source=source)


@dataclass
class ExplanationAgent:
    """Create SHAP-based reasoning output."""

    def run(self, model, features_df: pd.DataFrame, feature_columns: List[str]) -> ExplanationOutput:
        return build_explanations(model=model, X_reference=features_df[feature_columns])


@dataclass
class DecisionAgent:
    """Merge all agent signals into final recommendation."""

    def run(
        self,
        forecast: ForecastOutput,
        risk: RiskOutput,
        anomaly: AnomalyOutput,
        sentiment: SentimentOutput,
        explanation: ExplanationOutput,
    ) -> DecisionOutput:
        return make_decision(
            forecast=forecast,
            risk=risk,
            anomaly=anomaly,
            sentiment=sentiment,
            explanation=explanation,
        )
