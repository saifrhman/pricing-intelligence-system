"""Utility helpers for evaluation, config loading, and file handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config into a dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-aware split that preserves chronological order."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1.")

    n = len(df)
    if n < 30:
        raise ValueError("At least 30 rows are required for time-aware splitting.")

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Split produced an empty validation or test set.")

    return train_df, val_df, test_df


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute percentage of times predicted and true returns have same sign."""
    if len(y_true) == 0:
        return 0.0
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Standard regression metrics plus directional accuracy."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    dacc = directional_accuracy(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": dacc,
    }


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Persist a dictionary as a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_to_unit_interval(value: float, min_value: float, max_value: float) -> float:
    """Min-max normalize to [0, 1] with clipping."""
    if max_value <= min_value:
        return 0.0
    scaled = (value - min_value) / (max_value - min_value)
    return float(np.clip(scaled, 0.0, 1.0))


def map_risk_level(score: float) -> str:
    """Map continuous risk score to categorical levels."""
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"
