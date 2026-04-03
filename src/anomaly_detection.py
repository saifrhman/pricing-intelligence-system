"""Anomaly detection module for unusual market behavior."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def default_anomaly_features(df: pd.DataFrame) -> List[str]:
    """Select a stable set of anomaly features from processed data."""
    candidate_cols = [
        "daily_return",
        "log_return",
        "volume_change",
        "volatility_10",
        "volatility_20",
        "momentum_5",
        "momentum_10",
        "drawdown_20",
    ]
    return [c for c in candidate_cols if c in df.columns]


def detect_anomalies(
    df: pd.DataFrame,
    ticker: str,
    feature_cols: List[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
    output_dir: str | Path = "outputs/reports",
    plot_dir: str | Path = "outputs/plots",
) -> pd.DataFrame:
    """Fit IsolationForest and append anomaly labels/scores."""
    if df is None or df.empty:
        raise ValueError("Anomaly detection received empty input dataframe.")
    if len(df) < 30:
        raise ValueError("Anomaly detection requires at least 30 rows.")

    data = df.copy()
    selected_features = feature_cols or default_anomaly_features(data)
    if not selected_features:
        raise ValueError("No valid features found for anomaly detection.")
    if data[selected_features].isna().any().any():
        raise ValueError("Anomaly features contain missing values.")

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    model.fit(data[selected_features])

    data["anomaly_label"] = model.predict(data[selected_features])
    data["anomaly_score"] = model.decision_function(data[selected_features])
    data["is_anomaly"] = data["anomaly_label"] == -1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker.upper()}_anomalies.csv"
    data[["Date", "is_anomaly", "anomaly_score"]].to_csv(output_path, index=False)

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{ticker.upper()}_anomaly_overlay.png"
    _save_anomaly_plot(data, plot_path, ticker)

    return data


def _save_anomaly_plot(df: pd.DataFrame, output_path: Path, ticker: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(df["Date"]), df["Close"], label="Close", linewidth=1.5)

    anomaly_points = df[df["is_anomaly"]]
    ax.scatter(
        pd.to_datetime(anomaly_points["Date"]),
        anomaly_points["Close"],
        color="red",
        label="Anomaly",
        s=20,
        alpha=0.8,
    )

    ax.set_title(f"{ticker.upper()} Price with Isolation Forest Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def latest_anomaly_summary(df_with_anomalies: pd.DataFrame) -> dict:
    """Create simple anomaly summary for decision layer."""
    latest = df_with_anomalies.iloc[-1]
    recent_window = df_with_anomalies.tail(20)
    recent_rate = float(recent_window["is_anomaly"].mean())
    return {
        "is_anomaly": bool(latest["is_anomaly"]),
        "anomaly_score": float(latest["anomaly_score"]),
        "recent_anomaly_rate": recent_rate,
    }
