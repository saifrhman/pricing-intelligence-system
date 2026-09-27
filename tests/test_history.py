from pathlib import Path
from src.harness.state import HistoricalRunStore, JsonSessionStore
from src.schemas import (
    ForecastOutput,
    RiskOutput,
    AnomalyOutput,
    SentimentOutput,
    ExplanationOutput,
    DecisionOutput,
)

def _results():
    return {
        "forecast": ForecastOutput(
            ticker="AAPL",
            predicted_return=0.01,
            model_name="x",
            confidence_context="c",
            metrics={},
        ),
        "risk": RiskOutput(
            ticker="AAPL",
            risk_score=0.3,
            volatility_20d=0.02,
            drawdown_20d=-0.01,
            risk_level="low",
        ),
        "anomaly": AnomalyOutput(
            ticker="AAPL",
            is_anomaly=False,
            anomaly_score=0.2,
            recent_anomaly_rate=0.05,
        ),
        "sentiment": SentimentOutput(
            available=False,
            source="disabled",
            sentiment_label="unavailable",
            sentiment_score=0,
            headline_count=0,
        ),
        "explanation": ExplanationOutput(
            available=False,
            model_type="x",
            top_features=[],
        ),
        "decision": DecisionOutput(
            ticker="AAPL",
            latest_predicted_return=0.01,
            direction="bullish",
            risk_level="low",
            anomaly_status="not flagged",
            sentiment_summary="unavailable",
            top_drivers=[],
            recommendation_summary="x",
            caution_notes=[],
        ),
        "provenance": {
            "ingestion": {
                "source_type": "fresh"
            }
        },
    }

def test_history_round_trip(tmp_path: Path):
    store = HistoricalRunStore(
        tmp_path / "history"
    )
    summary = store.from_pipeline_results(
        _results(),
        "AAPL",
        run_id="run1",
    )
    path = store.save(summary)
    assert path.exists()

    latest = store.latest("AAPL")
    assert latest is not None
    assert latest.run_id == "run1"
    assert (
        latest.forecast["predicted_return"]
        == 0.01
    )

def test_session_round_trip(tmp_path: Path):
    store = JsonSessionStore(
        tmp_path / "sessions"
    )
    store.append(
        "s1",
        "user",
        "hello",
    )
    store.append(
        "s1",
        "assistant",
        "hi",
    )
    rows = store.load("s1")
    assert [
        row["role"]
        for row in rows
    ] == [
        "user",
        "assistant",
    ]
