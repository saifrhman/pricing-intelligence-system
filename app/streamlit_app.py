"""Streamlit dashboard for pricing intelligence system."""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.pipeline import run_pipeline


st.set_page_config(page_title="Pricing Intelligence System", layout="wide")
st.title("Pricing Intelligence and Risk Decision Support")
st.caption(
    "Portfolio-quality analytics workflow for return forecasting, risk scoring, anomalies, sentiment, and explainability."
)

with st.sidebar:
    st.header("Run Settings")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    start_date = st.date_input("Start Date", value=dt.date(2020, 1, 1))
    end_date = st.date_input("End Date", value=dt.date(2025, 1, 1))
    include_sentiment = st.checkbox("Enable sentiment analysis", value=True)
    use_transformer_sentiment = st.checkbox("Use FinBERT (transformers)", value=True)
    run_button = st.button("Run Pipeline", type="primary")

if run_button:
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()

    with st.spinner("Running pipeline..."):
        try:
            results = run_pipeline(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                interval="1d",
                include_sentiment=include_sentiment,
                use_transformer_sentiment=use_transformer_sentiment,
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    raw_df: pd.DataFrame = results["raw_df"]
    features_df: pd.DataFrame = results["features_df"]
    predictions_df: pd.DataFrame = results["predictions_df"]
    anomaly_df: pd.DataFrame = results["anomaly_df"]

    decision = results["decision"]
    forecast = results["forecast"]
    risk = results["risk"]
    sentiment = results["sentiment"]
    explanation = results["explanation"]

    st.subheader("Final Decision Panel")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Next-Day Return", f"{decision.latest_predicted_return:.4f}")
    c2.metric("Direction", decision.direction)
    c3.metric("Risk Level", decision.risk_level)
    c4.metric("Anomaly", decision.anomaly_status)
    st.info(decision.recommendation_summary)

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.tail(10), use_container_width=True)

    st.subheader("Engineered Features Preview")
    st.dataframe(features_df.tail(10), use_container_width=True)

    st.subheader("Model Performance")
    perf_df = pd.DataFrame.from_dict(
        {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in forecast.metrics.items()},
        orient="index",
    )
    st.dataframe(perf_df, use_container_width=True)

    st.subheader("Prediction Chart")
    fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
    ax_pred.plot(predictions_df["Date"], predictions_df["actual_next_return"], label="Actual")
    ax_pred.plot(predictions_df["Date"], predictions_df["predicted_next_return"], label="Predicted")
    ax_pred.set_title(f"{ticker} Next-Day Return Forecast")
    ax_pred.legend()
    ax_pred.grid(alpha=0.25)
    st.pyplot(fig_pred)

    st.subheader("Anomaly Detection")
    recent_anomalies = anomaly_df[["Date", "Close", "is_anomaly", "anomaly_score"]].tail(20)
    st.dataframe(recent_anomalies, use_container_width=True)

    fig_anom, ax_anom = plt.subplots(figsize=(10, 4))
    ax_anom.plot(anomaly_df["Date"], anomaly_df["Close"], label="Close")
    flagged = anomaly_df[anomaly_df["is_anomaly"]]
    ax_anom.scatter(flagged["Date"], flagged["Close"], color="red", s=20, label="Anomaly")
    ax_anom.set_title("Price Series with Anomaly Flags")
    ax_anom.legend()
    ax_anom.grid(alpha=0.25)
    st.pyplot(fig_anom)

    st.subheader("Sentiment Results")
    if sentiment.available:
        st.write(
            f"Label: {sentiment.sentiment_label} | Score: {sentiment.sentiment_score:.2f} | Headlines: {sentiment.headline_count}"
        )
    else:
        st.write("Sentiment module unavailable or disabled.")

    st.subheader("SHAP Explainability")
    if explanation.available and explanation.top_features:
        shap_df = pd.DataFrame(explanation.top_features)
        st.dataframe(shap_df, use_container_width=True)
        if "importance" in shap_df.columns:
            fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
            sorted_df = shap_df.sort_values("importance", ascending=True).tail(8)
            ax_shap.barh(sorted_df["feature"], sorted_df["importance"], color="#2f6f8f")
            ax_shap.set_title("Top Global SHAP Drivers")
            ax_shap.grid(axis="x", alpha=0.2)
            st.pyplot(fig_shap)
    else:
        st.write("SHAP output unavailable for this run.")

    st.subheader("Caution Notes")
    for note in decision.caution_notes:
        st.write(f"- {note}")

    st.success(
        f"Artifacts saved: {results['artifacts']['model_path']}, {results['artifacts']['prediction_plot_path']}, {results['artifacts']['decision_report_path']}"
    )

else:
    st.write("Select inputs and click 'Run Pipeline' to generate a full pricing intelligence report.")
