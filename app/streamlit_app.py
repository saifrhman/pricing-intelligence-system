"""Streamlit dashboard and grounded pricing-intelligence copilot."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic.engine import AgenticPricingEngine
from src.harness.config import AgentRuntimeConfig
from src.harness.runtime import AgentHarness, HarnessContext
from src.harness.state import HistoricalRunStore, JsonSessionStore
from src.harness.tracing import TraceRecorder
from src.pipeline import run_pipeline
from src.rag.retriever import EvidenceRetriever
from src.rag.vector_store import build_vector_store_from_env
from src.tools.intelligence_tools import build_tool_registry


st.set_page_config(
    page_title="Agentic Pricing Intelligence",
    layout="wide",
)
st.title("Agentic Pricing Intelligence Platform")
st.caption(
    "Deterministic forecasting and risk analytics with RAG, LLM specialist "
    "agents, verification, provenance and MCP interoperability."
)

with st.sidebar:
    st.header("Run Settings")
    ticker = st.text_input(
        "Ticker",
        value="AAPL",
    ).upper().strip()
    start_date = st.date_input(
        "Start Date",
        value=dt.date(2020, 1, 1),
    )
    end_date = st.date_input(
        "End Date",
        value=dt.date(2025, 1, 1),
    )
    include_sentiment = st.checkbox(
        "Enable sentiment analysis",
        value=True,
    )
    use_transformer_sentiment = st.checkbox(
        "Use FinBERT (transformers)",
        value=True,
    )
    allow_cache_fallback = st.checkbox(
        "Allow cached market data fallback",
        value=False,
    )
    demo_mode = st.checkbox(
        "Enable demo mode (synthetic data/mock sentiment)",
        value=False,
    )
    manual_headlines_text = st.text_area(
        "Manual sentiment headlines (one per line)",
        value="",
        help=(
            "If empty, sentiment is unavailable unless demo mode "
            "is enabled."
        ),
    )
    run_button = st.button(
        "Run Deterministic Pipeline",
        type="primary",
    )

if "agent_session_id" not in st.session_state:
    st.session_state.agent_session_id = uuid4().hex

if "copilot_messages" not in st.session_state:
    st.session_state.copilot_messages = []

if run_button:
    if start_date >= end_date:
        st.error(
            "Start date must be earlier than end date."
        )
        st.stop()

    with st.spinner(
        "Running deterministic analytics..."
    ):
        try:
            manual_headlines = [
                line.strip()
                for line in manual_headlines_text.splitlines()
                if line.strip()
            ] or None

            st.session_state.last_results = run_pipeline(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                interval="1d",
                include_sentiment=include_sentiment,
                use_transformer_sentiment=use_transformer_sentiment,
                allow_cache_fallback=allow_cache_fallback,
                demo_mode=demo_mode,
                manual_headlines=manual_headlines,
            )
            st.session_state.last_ticker = ticker
            st.session_state.copilot_messages = []
        except Exception as exc:
            st.error(
                f"Pipeline failed: {exc}"
            )
            st.stop()

results = st.session_state.get(
    "last_results"
)
if results is None:
    st.info(
        "Run the deterministic pipeline first. The agent layer never "
        "invents analytical values without an authoritative run."
    )
    st.stop()

active_ticker = st.session_state.get(
    "last_ticker",
    ticker,
)

raw_df: pd.DataFrame = results["raw_df"]
features_df: pd.DataFrame = results["features_df"]
predictions_df: pd.DataFrame = results["predictions_df"]
anomaly_df: pd.DataFrame = results["anomaly_df"]

decision = results["decision"]
forecast = results["forecast"]
risk = results["risk"]
sentiment = results["sentiment"]
explanation = results["explanation"]
provenance = results.get(
    "provenance",
    {},
)
ingestion = provenance.get(
    "ingestion",
    {},
)

left, right = st.columns(
    [1.35, 1]
)

with left:
    st.subheader(
        "Deterministic Decision Support"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Predicted Next-Day Return",
        f"{decision.latest_predicted_return:.4f}",
    )
    c2.metric(
        "Direction",
        decision.direction,
    )
    c3.metric(
        "Risk Level",
        decision.risk_level,
    )
    c4.metric(
        "Anomaly",
        decision.anomaly_status,
    )
    st.info(
        decision.recommendation_summary
    )

    st.subheader(
        "Provenance"
    )
    st.write(
        f"Market data source: {ingestion.get('source_type', 'unknown')} "
        f"(status={ingestion.get('status', 'unknown')}, "
        f"attempts={ingestion.get('attempts', 'unknown')})"
    )
    for warning in provenance.get(
        "warnings",
        [],
    ):
        st.warning(warning)

    with st.expander(
        "Model performance and data previews"
    ):
        perf_df = pd.DataFrame.from_dict(
            {
                key: (
                    value.model_dump()
                    if hasattr(
                        value,
                        "model_dump",
                    )
                    else value
                )
                for key, value in forecast.metrics.items()
            },
            orient="index",
        )
        st.dataframe(
            perf_df,
            use_container_width=True,
        )
        st.write(
            "Raw data"
        )
        st.dataframe(
            raw_df.tail(10),
            use_container_width=True,
        )
        st.write(
            "Engineered features"
        )
        st.dataframe(
            features_df.tail(10),
            use_container_width=True,
        )

    st.subheader(
        "Prediction Chart"
    )
    fig_pred, ax_pred = plt.subplots(
        figsize=(10, 4)
    )
    ax_pred.plot(
        predictions_df["Date"],
        predictions_df[
            "actual_next_return"
        ],
        label="Actual",
    )
    ax_pred.plot(
        predictions_df["Date"],
        predictions_df[
            "predicted_next_return"
        ],
        label="Predicted",
    )
    ax_pred.set_title(
        f"{active_ticker} Next-Day Return Forecast"
    )
    ax_pred.legend()
    ax_pred.grid(
        alpha=0.25
    )
    st.pyplot(
        fig_pred
    )

    st.subheader(
        "Anomaly Detection"
    )
    recent_anomalies = anomaly_df[
        [
            "Date",
            "Close",
            "is_anomaly",
            "anomaly_score",
        ]
    ].tail(20)
    st.dataframe(
        recent_anomalies,
        use_container_width=True,
    )

    st.subheader(
        "Sentiment"
    )
    if sentiment.available:
        st.write(
            f"{sentiment.sentiment_label} | "
            f"score={sentiment.sentiment_score:.2f} | "
            f"source={sentiment.source}"
        )
    else:
        st.write(
            "Sentiment unavailable or disabled."
        )

    st.subheader(
        "SHAP Explainability"
    )
    if (
        explanation.available
        and explanation.top_features
    ):
        shap_df = pd.DataFrame(
            explanation.top_features
        )
        st.dataframe(
            shap_df,
            use_container_width=True,
        )
        st.caption(
            "SHAP values describe model contribution, not market causality."
        )
    else:
        st.write(
            "SHAP output unavailable for this run."
        )

with right:
    st.subheader(
        "AI Pricing Intelligence Copilot"
    )
    if AgenticPricingEngine.llm_available():
        st.caption(
            "LLM manager enabled. Specialist agents dynamically select "
            "deterministic tools and RAG evidence."
        )
    else:
        st.caption(
            "LLM runtime not configured. Questions use transparent deterministic "
            "fallback routing; set OPENAI_API_KEY to enable LLM orchestration."
        )

    for message in st.session_state.copilot_messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        f"Ask about {active_ticker}: risk, forecast, SHAP, evidence, or changes over time"
    )
    if question:
        st.session_state.copilot_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )
        with st.chat_message(
            "user"
        ):
            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):
            with st.spinner(
                "Running grounded analysis..."
            ):
                try:
                    config = AgentRuntimeConfig.from_env()
                    tracer = TraceRecorder(
                        enabled=config.tracing_enabled
                    )
                    harness = AgentHarness(
                        build_tool_registry(),
                        config=config,
                        tracer=tracer,
                    )
                    context = HarnessContext(
                        ticker=active_ticker,
                        pipeline_results=results,
                        history_store=HistoricalRunStore(),
                        retriever=EvidenceRetriever(
                            build_vector_store_from_env()
                        ),
                    )
                    engine = AgenticPricingEngine(
                        harness,
                        config=config,
                        session_store=JsonSessionStore(),
                    )
                    report = engine.run(
                        question,
                        context,
                        session_id=(
                            st.session_state.agent_session_id
                        ),
                    )

                    st.markdown(
                        report.answer
                    )
                    st.session_state.copilot_messages.append(
                        {
                            "role": "assistant",
                            "content": report.answer,
                        }
                    )

                    if report.selected_agents:
                        st.caption(
                            "Specialists used: "
                            + ", ".join(
                                report.selected_agents
                            )
                        )

                    if report.citations:
                        with st.expander(
                            f"Retrieved evidence ({len(report.citations)})"
                        ):
                            for item in report.citations:
                                st.markdown(
                                    f"**{item.source}** · "
                                    f"\`{item.chunk_id}\` · "
                                    f"{item.date or 'date unknown'}"
                                )
                                st.write(
                                    item.text
                                )

                    with st.expander(
                        "Verification and execution trace"
                    ):
                        st.json(
                            report.verification.model_dump()
                        )
                        trace_rows = [
                            row
                            for row in tracer.read(
                                report.run_id
                            )
                            if row.get(
                                "status"
                            )
                            in {
                                "completed",
                                "failed",
                            }
                        ]
                        st.dataframe(
                            pd.DataFrame(
                                trace_rows
                            ),
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(
                        f"Copilot failed: {exc}"
                    )

st.caption(
    "Educational/research decision support only. No trade execution is "
    "implemented and outputs are not investment advice."
)
