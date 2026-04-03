"""SHAP-based explainability for tree-based forecasting models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.schemas import ExplanationOutput

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


def build_explanations(
    model,
    X_reference: pd.DataFrame,
    output_dir: str | Path = "outputs/shap",
    max_features: int = 10,
) -> ExplanationOutput:
    """Generate global and local SHAP artifacts for latest observation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if shap is None:
        return ExplanationOutput(
            available=False,
            model_type=type(model).__name__,
            top_features=[],
            latest_expected_value=None,
        )

    if X_reference.empty:
        return ExplanationOutput(
            available=False,
            model_type=type(model).__name__,
            top_features=[],
            latest_expected_value=None,
        )

    # Use a manageable subset for speed and laptop friendliness.
    sample = X_reference.tail(min(250, len(X_reference))).copy()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
    except Exception:
        return ExplanationOutput(
            available=False,
            model_type=type(model).__name__,
            top_features=[],
            latest_expected_value=None,
        )

    shap_matrix = np.array(shap_values)
    global_importance = np.abs(shap_matrix).mean(axis=0)
    features = sample.columns.tolist()

    top_idx = np.argsort(global_importance)[::-1][:max_features]
    top_features: List[Dict[str, float]] = [
        {"feature": features[i], "importance": float(global_importance[i])} for i in top_idx
    ]

    _save_global_importance_plot(features, global_importance, output_path / "global_feature_importance.png")
    _save_local_contribution_plot(
        sample.iloc[-1],
        shap_matrix[-1],
        output_path / "latest_local_explanation.png",
        top_n=min(8, max_features),
    )

    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = float(expected_value[0])

    return ExplanationOutput(
        available=True,
        model_type=type(model).__name__,
        top_features=top_features,
        latest_expected_value=float(expected_value),
    )


def _save_global_importance_plot(
    feature_names: List[str],
    importance: np.ndarray,
    output_path: Path,
) -> None:
    order = np.argsort(importance)[::-1][:12]
    ordered_features = [feature_names[i] for i in order]
    ordered_values = importance[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered_features[::-1], ordered_values[::-1], color="#2f6f8f")
    ax.set_title("SHAP Global Feature Importance")
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_local_contribution_plot(
    latest_row: pd.Series,
    latest_shap_values: np.ndarray,
    output_path: Path,
    top_n: int = 8,
) -> None:
    abs_values = np.abs(latest_shap_values)
    order = np.argsort(abs_values)[::-1][:top_n]

    labels = [latest_row.index[i] for i in order]
    values = latest_shap_values[order]
    colors = ["#3b8b4a" if v >= 0 else "#b74b4b" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_title("Latest Prediction: Top SHAP Contributions")
    ax.set_xlabel("SHAP contribution")
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
