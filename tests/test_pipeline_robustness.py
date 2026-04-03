from __future__ import annotations

import pandas as pd
import pytest

import src.pipeline as pipeline
from src.data_ingestion import IngestionMetadata, IngestionResult
from src.schemas import (
    AnomalyOutput,
    DecisionOutput,
    ExplanationOutput,
    ForecastOutput,
    ModelPerformance,
    RiskOutput,
    SentimentOutput,
)


class DummyQuantAgent:
    def run(self, ticker: str, forecast_payload: dict) -> ForecastOutput:
        return ForecastOutput(
            ticker=ticker,
            predicted_return=0.01,
            model_name="dummy",
            confidence_context="test",
            metrics={
                "dummy_test": ModelPerformance(
                    rmse=0.01,
                    mae=0.01,
                    r2=0.1,
                    directional_accuracy=0.6,
                )
            },
        )


class DummyRiskAgent:
    def run(self, ticker: str, features_df: pd.DataFrame) -> RiskOutput:
        return RiskOutput(
            ticker=ticker,
            risk_score=0.3,
            volatility_20d=0.02,
            drawdown_20d=-0.01,
            risk_level="low",
        )


class DummyAnomalyAgent:
    def run(self, ticker: str, anomaly_df: pd.DataFrame) -> AnomalyOutput:
        return AnomalyOutput(
            ticker=ticker,
            is_anomaly=False,
            anomaly_score=0.1,
            recent_anomaly_rate=0.05,
        )


class DummySentimentAgent:
    def __init__(self, analyzer) -> None:
        self.analyzer = analyzer

    def run(self, headlines, source="headlines", allow_rule_based_fallback=False) -> SentimentOutput:
        return SentimentOutput(
            available=True,
            source=f"{source}:stub",
            sentiment_label="neutral",
            sentiment_score=0.0,
            headline_count=len(headlines),
        )


class DummyExplanationAgent:
    def run(self, model, features_df: pd.DataFrame, feature_columns):
        return ExplanationOutput(
            available=True,
            model_type="Dummy",
            top_features=[{"feature": "lag_return_1", "importance": 0.1}],
            latest_expected_value=0.0,
        )


class DummyDecisionAgent:
    def run(self, forecast, risk, anomaly, sentiment, explanation):
        return DecisionOutput(
            ticker=forecast.ticker,
            latest_predicted_return=forecast.predicted_return,
            direction="bullish",
            risk_level=risk.risk_level,
            anomaly_status="not flagged",
            sentiment_summary=sentiment.source,
            top_drivers=["lag_return_1"],
            recommendation_summary="ok",
            caution_notes=["note"],
        )


def _ingestion_metadata(source: str = "fresh") -> IngestionMetadata:
    return IngestionMetadata(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2020-12-31",
        interval="1d",
        source_type=source,
        status="success" if source == "fresh" else "degraded",
        attempts=1,
        warnings=[],
        error_messages=[],
    )


def test_pipeline_rejects_empty_raw_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        pipeline,
        "fetch_market_data",
        lambda **kwargs: IngestionResult(data=pd.DataFrame(), metadata=_ingestion_metadata()),
    )

    with pytest.raises(ValueError, match="empty raw data"):
        pipeline.run_pipeline(
            ticker="AAPL",
            start_date="2020-01-01",
            end_date="2020-12-31",
            interval="1d",
            include_sentiment=False,
            use_transformer_sentiment=False,
            outputs_dir=str(tmp_path),
        )


def test_pipeline_sentiment_unavailable_without_demo_or_manual(monkeypatch, tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=120, freq="B"),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1_000_000,
        }
    )
    features = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=120, freq="B"),
            "lag_return_1": 0.001,
            "volatility_20": 0.02,
            "drawdown_20": -0.01,
            "target_next_return": 0.001,
        }
    )
    anomaly = features.copy()
    anomaly["Close"] = 100.0
    anomaly["is_anomaly"] = False
    anomaly["anomaly_score"] = 0.1

    monkeypatch.setattr(
        pipeline,
        "fetch_market_data",
        lambda **kwargs: IngestionResult(data=raw, metadata=_ingestion_metadata()),
    )
    monkeypatch.setattr(pipeline, "engineer_features", lambda _: features)
    monkeypatch.setattr(
        pipeline,
        "train_and_evaluate",
        lambda *args, **kwargs: {
            "feature_columns": ["lag_return_1", "volatility_20", "drawdown_20"],
            "metrics_by_model": {},
            "best_model_name": "dummy",
            "best_model": object(),
            "model_path": "m",
            "prediction_plot_path": "p",
            "predictions_df": pd.DataFrame(
                {
                    "Date": pd.date_range("2020-01-01", periods=10, freq="B"),
                    "actual_next_return": 0.001,
                    "predicted_next_return": 0.001,
                }
            ),
            "latest_prediction": 0.001,
        },
    )
    monkeypatch.setattr(pipeline, "detect_anomalies", lambda *args, **kwargs: anomaly)

    monkeypatch.setattr(pipeline, "QuantAgent", DummyQuantAgent)
    monkeypatch.setattr(pipeline, "RiskAgent", DummyRiskAgent)
    monkeypatch.setattr(pipeline, "AnomalyAgent", DummyAnomalyAgent)
    monkeypatch.setattr(pipeline, "SentimentAgent", DummySentimentAgent)
    monkeypatch.setattr(pipeline, "ExplanationAgent", DummyExplanationAgent)
    monkeypatch.setattr(pipeline, "DecisionAgent", DummyDecisionAgent)

    result = pipeline.run_pipeline(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2020-12-31",
        interval="1d",
        include_sentiment=True,
        use_transformer_sentiment=True,
        outputs_dir=str(tmp_path),
        demo_mode=False,
        manual_headlines=None,
    )

    assert result["sentiment"].available is False
    assert result["sentiment"].source == "unavailable_no_headlines"


def test_pipeline_stops_when_features_empty(monkeypatch, tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=120, freq="B"),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1_000_000,
        }
    )

    monkeypatch.setattr(
        pipeline,
        "fetch_market_data",
        lambda **kwargs: IngestionResult(data=raw, metadata=_ingestion_metadata()),
    )
    monkeypatch.setattr(pipeline, "engineer_features", lambda _: pd.DataFrame())

    with pytest.raises(ValueError, match="zero usable rows"):
        pipeline.run_pipeline(
            ticker="AAPL",
            start_date="2020-01-01",
            end_date="2020-12-31",
            interval="1d",
            include_sentiment=False,
            use_transformer_sentiment=False,
            outputs_dir=str(tmp_path),
        )
