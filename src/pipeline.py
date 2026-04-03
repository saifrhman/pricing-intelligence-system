"""Orchestration for pricing intelligence workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.agents import (
    AnomalyAgent,
    DecisionAgent,
    ExplanationAgent,
    QuantAgent,
    RiskAgent,
    SentimentAgent,
)
from src.anomaly_detection import detect_anomalies
from src.data_ingestion import fetch_market_data
from src.feature_engineering import engineer_features, save_processed_data
from src.forecasting import train_and_evaluate
from src.sentiment import SentimentAnalyzer, default_mock_headlines
from src.utils import save_json


def run_pipeline(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    include_sentiment: bool,
    use_transformer_sentiment: bool,
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    outputs_dir: str = "outputs",
    manual_headlines: Optional[List[str]] = None,
) -> Dict:
    """Run ingestion -> features -> forecasting -> anomaly -> sentiment -> explainability -> decision."""
    outputs_path = Path(outputs_dir)
    plots_dir = outputs_path / "plots"
    models_dir = outputs_path / "models"
    reports_dir = outputs_path / "reports"

    raw_df = fetch_market_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        raw_data_dir=raw_data_dir,
    )

    features_df = engineer_features(raw_df)
    save_processed_data(features_df, ticker=ticker, processed_data_dir=processed_data_dir)

    forecast_payload = train_and_evaluate(
        features_df,
        ticker=ticker,
        model_output_dir=models_dir,
        plot_output_dir=plots_dir,
    )

    anomaly_df = detect_anomalies(
        features_df,
        ticker=ticker,
        output_dir=reports_dir,
        plot_dir=plots_dir,
    )

    quant_agent = QuantAgent()
    risk_agent = RiskAgent()
    anomaly_agent = AnomalyAgent()
    sentiment_agent = SentimentAgent(SentimentAnalyzer(use_transformer=use_transformer_sentiment))
    explanation_agent = ExplanationAgent()
    decision_agent = DecisionAgent()

    quant_output = quant_agent.run(ticker=ticker, forecast_payload=forecast_payload)
    risk_output = risk_agent.run(ticker=ticker, features_df=features_df)
    anomaly_output = anomaly_agent.run(ticker=ticker, anomaly_df=anomaly_df)

    if include_sentiment:
        headlines = manual_headlines or default_mock_headlines(ticker)
        sentiment_output = sentiment_agent.run(headlines=headlines)
    else:
        sentiment_output = sentiment_agent.run(headlines=[])

    explanation_output = explanation_agent.run(
        model=forecast_payload["best_model"],
        features_df=features_df,
        feature_columns=forecast_payload["feature_columns"],
    )

    decision_output = decision_agent.run(
        forecast=quant_output,
        risk=risk_output,
        anomaly=anomaly_output,
        sentiment=sentiment_output,
        explanation=explanation_output,
    )

    decision_report = decision_output.model_dump()
    report_path = reports_dir / f"{ticker.upper()}_decision_report.json"
    save_json(report_path, decision_report)

    return {
        "raw_df": raw_df,
        "features_df": features_df,
        "predictions_df": forecast_payload["predictions_df"],
        "anomaly_df": anomaly_df,
        "forecast": quant_output,
        "risk": risk_output,
        "anomaly": anomaly_output,
        "sentiment": sentiment_output,
        "explanation": explanation_output,
        "decision": decision_output,
        "artifacts": {
            "model_path": forecast_payload["model_path"],
            "prediction_plot_path": forecast_payload["prediction_plot_path"],
            "decision_report_path": str(report_path),
        },
    }


def format_markdown_report(results: Dict) -> str:
    """Create a concise markdown report from pipeline results."""
    decision = results["decision"]
    risk = results["risk"]
    anomaly = results["anomaly"]
    sentiment = results["sentiment"]

    lines = [
        f"# Pricing Intelligence Report - {decision.ticker}",
        "",
        "## Executive Summary",
        decision.recommendation_summary,
        "",
        "## Signal Snapshot",
        f"- Latest predicted return: {decision.latest_predicted_return:.4f}",
        f"- Direction: {decision.direction}",
        f"- Risk level: {decision.risk_level} (score={risk.risk_score:.3f})",
        f"- Anomaly status: {decision.anomaly_status} (recent rate={anomaly.recent_anomaly_rate:.2%})",
        f"- Sentiment: {decision.sentiment_summary}",
        "",
        "## Top Drivers",
    ]

    if decision.top_drivers:
        lines.extend([f"- {d}" for d in decision.top_drivers])
    else:
        lines.append("- Explainability unavailable")

    lines.extend([
        "",
        "## Caution Notes",
    ])
    lines.extend([f"- {note}" for note in decision.caution_notes])

    lines.extend([
        "",
        "## Disclaimer",
        "This report is for educational and research purposes only. It is not investment advice.",
    ])

    return "\n".join(lines)


def save_markdown_report(markdown_text: str, ticker: str, output_dir: str = "outputs/reports") -> str:
    """Save markdown report."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / f"{ticker.upper()}_report.md"
    report_path.write_text(markdown_text, encoding="utf-8")
    return str(report_path)
