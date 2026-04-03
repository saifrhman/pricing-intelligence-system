from src.decision_engine import make_decision
from src.schemas import (
    AnomalyOutput,
    ExplanationOutput,
    ForecastOutput,
    ModelPerformance,
    RiskOutput,
    SentimentOutput,
)


def test_decision_output_has_required_fields() -> None:
    forecast = ForecastOutput(
        ticker="AAPL",
        predicted_return=0.004,
        model_name="xgboost",
        confidence_context="test",
        metrics={
            "xgboost_test": ModelPerformance(
                rmse=0.01, mae=0.008, r2=0.2, directional_accuracy=0.56
            )
        },
    )
    risk = RiskOutput(
        ticker="AAPL",
        risk_score=0.4,
        volatility_20d=0.02,
        drawdown_20d=-0.03,
        risk_level="medium",
    )
    anomaly = AnomalyOutput(
        ticker="AAPL",
        is_anomaly=False,
        anomaly_score=0.12,
        recent_anomaly_rate=0.1,
    )
    sentiment = SentimentOutput(
        available=True,
        source="mock",
        sentiment_label="positive",
        sentiment_score=0.3,
        headline_count=3,
    )
    explanation = ExplanationOutput(
        available=True,
        model_type="XGBRegressor",
        top_features=[{"feature": "lag_return_1", "importance": 0.2}],
        latest_expected_value=0.0,
    )

    out = make_decision(forecast, risk, anomaly, sentiment, explanation)

    assert out.ticker == "AAPL"
    assert out.direction in {"bullish", "bearish", "neutral"}
    assert isinstance(out.recommendation_summary, str)
    assert len(out.caution_notes) >= 1
