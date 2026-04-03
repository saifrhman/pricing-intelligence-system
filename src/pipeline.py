"""Orchestration for pricing intelligence workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.data_ingestion import IngestionMetadata, fetch_market_data
from src.feature_engineering import engineer_features, save_processed_data
from src.forecasting import train_and_evaluate
from src.schemas import SentimentOutput
from src.sentiment import SentimentAnalysisError, SentimentAnalyzer, default_mock_headlines
from src.utils import save_json


logger = logging.getLogger(__name__)

REQUIRED_RAW_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}


def _validate_raw_data(raw_df: pd.DataFrame) -> None:
    if raw_df is None or raw_df.empty:
        raise ValueError("Ingestion produced empty raw data.")

    missing = REQUIRED_RAW_COLUMNS - set(raw_df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")

    if len(raw_df) < 80:
        raise ValueError(
            f"Raw data has only {len(raw_df)} rows. At least 80 rows are required for stable training."
        )


def _build_provenance(
    ingestion_meta: IngestionMetadata,
    sentiment_output: SentimentOutput,
    warnings: List[str],
) -> Dict[str, Any]:
    return {
        "ingestion": {
            "ticker": ingestion_meta.ticker,
            "source_type": ingestion_meta.source_type,
            "status": ingestion_meta.status,
            "attempts": ingestion_meta.attempts,
            "warnings": ingestion_meta.warnings,
            "errors": ingestion_meta.error_messages,
        },
        "sentiment": {
            "status": "available" if sentiment_output.available else "unavailable",
            "source": sentiment_output.source,
            "headline_count": sentiment_output.headline_count,
        },
        "warnings": warnings,
    }


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
    allow_cache_fallback: bool = False,
    demo_mode: bool = False,
    ingestion_timeout_seconds: int = 30,
    ingestion_max_retries: int = 3,
) -> Dict:
    """Run ingestion -> features -> forecasting -> anomaly -> sentiment -> explainability -> decision."""
    logger.info("Pipeline started for ticker=%s", ticker)
    outputs_path = Path(outputs_dir)
    plots_dir = outputs_path / "plots"
    models_dir = outputs_path / "models"
    reports_dir = outputs_path / "reports"
    warnings: List[str] = []

    ingestion_result = fetch_market_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        raw_data_dir=raw_data_dir,
        timeout_seconds=ingestion_timeout_seconds,
        max_retries=ingestion_max_retries,
        allow_cache_fallback=allow_cache_fallback,
        allow_demo_fallback=demo_mode,
    )
    raw_df = ingestion_result.data
    _validate_raw_data(raw_df)

    if ingestion_result.metadata.source_type != "fresh":
        warnings.append(
            f"Using {ingestion_result.metadata.source_type} market data. Treat outputs as lower confidence."
        )
    logger.info(
        "Ingestion completed: source=%s rows=%s",
        ingestion_result.metadata.source_type,
        len(raw_df),
    )

    features_df = engineer_features(raw_df)
    if features_df.empty:
        raise ValueError("Feature engineering produced zero usable rows.")
    logger.info("Feature engineering completed: rows=%s", len(features_df))

    save_processed_data(features_df, ticker=ticker, processed_data_dir=processed_data_dir)

    forecast_payload = train_and_evaluate(
        features_df,
        ticker=ticker,
        model_output_dir=models_dir,
        plot_output_dir=plots_dir,
    )
    if forecast_payload.get("predictions_df") is None or forecast_payload["predictions_df"].empty:
        raise ValueError("Forecasting produced no predictions.")
    logger.info("Forecasting completed using model=%s", forecast_payload["best_model_name"])

    anomaly_df = detect_anomalies(
        features_df,
        ticker=ticker,
        output_dir=reports_dir,
        plot_dir=plots_dir,
    )
    if anomaly_df.empty:
        raise ValueError("Anomaly detection produced no results.")
    logger.info("Anomaly detection completed")

    quant_agent = QuantAgent()
    risk_agent = RiskAgent()
    anomaly_agent = AnomalyAgent()
    sentiment_agent = SentimentAgent(SentimentAnalyzer(use_transformer=use_transformer_sentiment))
    explanation_agent = ExplanationAgent()
    decision_agent = DecisionAgent()

    quant_output = quant_agent.run(ticker=ticker, forecast_payload=forecast_payload)
    risk_output = risk_agent.run(ticker=ticker, features_df=features_df)
    anomaly_output = anomaly_agent.run(ticker=ticker, anomaly_df=anomaly_df)

    if not include_sentiment:
        sentiment_output = SentimentOutput(
            available=False,
            source="disabled",
            sentiment_label="unavailable",
            sentiment_score=0.0,
            headline_count=0,
        )
        logger.info("Sentiment disabled")
    else:
        if manual_headlines:
            try:
                source = "manual"
                allow_rb_fallback = not use_transformer_sentiment
                sentiment_output = sentiment_agent.run(
                    headlines=manual_headlines,
                    source=source,
                    allow_rule_based_fallback=allow_rb_fallback,
                )
                logger.info("Sentiment completed from manual headlines")
            except SentimentAnalysisError as exc:
                sentiment_output = SentimentOutput(
                    available=False,
                    source="manual_failed",
                    sentiment_label="unavailable",
                    sentiment_score=0.0,
                    headline_count=len(manual_headlines),
                )
                warnings.append(f"Sentiment unavailable due to analysis failure: {exc}")
                logger.warning("Sentiment analysis failed: %s", exc)
        elif demo_mode:
            mock_headlines = default_mock_headlines(ticker)
            sentiment_output = sentiment_agent.run(
                headlines=mock_headlines,
                source="demo_mock",
                allow_rule_based_fallback=True,
            )
            warnings.append("Sentiment uses mock headlines because demo mode is enabled.")
            logger.warning("Sentiment running in demo mock mode")
        else:
            sentiment_output = SentimentOutput(
                available=False,
                source="unavailable_no_headlines",
                sentiment_label="unavailable",
                sentiment_score=0.0,
                headline_count=0,
            )
            warnings.append(
                "Sentiment requested but no manual headlines were provided; sentiment marked unavailable."
            )
            logger.warning("Sentiment unavailable: no headlines provided and demo mode disabled")

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

    provenance = _build_provenance(
        ingestion_meta=ingestion_result.metadata,
        sentiment_output=sentiment_output,
        warnings=warnings,
    )

    decision_report = {
        "decision": decision_output.model_dump(),
        "provenance": provenance,
        "artifacts": {
            "model_path": forecast_payload["model_path"],
            "prediction_plot_path": forecast_payload["prediction_plot_path"],
        },
    }
    report_path = reports_dir / f"{ticker.upper()}_decision_report.json"
    save_json(report_path, decision_report)
    logger.info("Decision report saved to %s", report_path)

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
        "provenance": provenance,
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
    provenance = results.get("provenance", {})
    ingestion_meta = provenance.get("ingestion", {})
    pipeline_warnings = provenance.get("warnings", [])

    lines = [
        f"# Pricing Intelligence Report - {decision.ticker}",
        "",
        "## Executive Summary",
        decision.recommendation_summary,
        "",
        "## Data Provenance",
        f"- Ingestion source: {ingestion_meta.get('source_type', 'unknown')}",
        f"- Ingestion status: {ingestion_meta.get('status', 'unknown')}",
        f"- Ingestion attempts: {ingestion_meta.get('attempts', 'unknown')}",
        f"- Sentiment source: {sentiment.source}",
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

    if pipeline_warnings:
        lines.extend([
            "",
            "## Pipeline Warnings",
        ])
        lines.extend([f"- {warning}" for warning in pipeline_warnings])

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
