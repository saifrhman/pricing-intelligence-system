"""Forecasting module for next-day return prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import regression_metrics, time_split

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - handled at runtime for portability
    XGBRegressor = None


TARGET_COL = "target_next_return"
EXCLUDED_FEATURES = {"Date", TARGET_COL}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Infer feature columns from processed dataset."""
    return [col for col in df.columns if col not in EXCLUDED_FEATURES]


def _fit_and_predict(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def _plot_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true.index, y_true.values, label="Actual", linewidth=2)
    ax.plot(y_true.index, y_pred.values, label="Predicted", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Next-Day Return")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train_and_evaluate(
    df: pd.DataFrame,
    ticker: str,
    model_output_dir: str | Path = "outputs/models",
    plot_output_dir: str | Path = "outputs/plots",
) -> Dict:
    """Train baseline and ML models using time-aware splits."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataframe.")

    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns available for forecasting.")

    train_df, val_df, test_df = time_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    metrics_by_model: Dict[str, Dict[str, float]] = {}

    # Baseline: previous day's return as naive forecast.
    naive_val_pred = val_df["lag_return_1"] if "lag_return_1" in val_df else pd.Series(0, index=val_df.index)
    naive_test_pred = test_df["lag_return_1"] if "lag_return_1" in test_df else pd.Series(0, index=test_df.index)
    metrics_by_model["naive_previous_return_val"] = regression_metrics(y_val, naive_val_pred)
    metrics_by_model["naive_previous_return_test"] = regression_metrics(y_test, naive_test_pred)

    linear_model = Pipeline(
        steps=[("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )
    linear_val_pred = _fit_and_predict(linear_model, X_train, y_train, X_val)
    metrics_by_model["linear_regression_val"] = regression_metrics(y_val, pd.Series(linear_val_pred, index=y_val.index))

    tree_model_name = "xgboost"
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is required for the main tree-based model. Install dependencies from requirements.txt."
        )

    tree_model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    tree_val_pred = _fit_and_predict(tree_model, X_train, y_train, X_val)
    metrics_by_model[f"{tree_model_name}_val"] = regression_metrics(
        y_val, pd.Series(tree_val_pred, index=y_val.index)
    )

    candidate_scores: List[Tuple[str, float]] = [
        ("linear_regression", metrics_by_model["linear_regression_val"]["rmse"]),
        (tree_model_name, metrics_by_model[f"{tree_model_name}_val"]["rmse"]),
    ]
    best_model_name = min(candidate_scores, key=lambda x: x[1])[0]

    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    if best_model_name == "linear_regression":
        best_model = Pipeline(
            steps=[("scaler", StandardScaler()), ("regressor", LinearRegression())]
        )
    else:
        best_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )

    best_model.fit(X_train_full, y_train_full)
    test_pred = best_model.predict(X_test)
    test_pred_series = pd.Series(test_pred, index=y_test.index)

    metrics_by_model[f"{best_model_name}_test"] = regression_metrics(y_test, test_pred_series)

    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_output_dir / f"{ticker.upper()}_{best_model_name}.joblib"
    joblib.dump(best_model, model_path)

    plot_output_path = Path(plot_output_dir) / f"{ticker.upper()}_prediction_vs_actual.png"
    _plot_predictions(
        y_true=y_test,
        y_pred=test_pred_series,
        output_path=plot_output_path,
        title=f"{ticker.upper()} Next-Day Return: Actual vs Predicted",
    )

    predictions_df = test_df[["Date", TARGET_COL]].copy()
    predictions_df = predictions_df.rename(columns={TARGET_COL: "actual_next_return"})
    predictions_df["predicted_next_return"] = test_pred_series.values

    return {
        "feature_columns": feature_cols,
        "metrics_by_model": metrics_by_model,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "model_path": str(model_path),
        "prediction_plot_path": str(plot_output_path),
        "predictions_df": predictions_df,
        "latest_prediction": float(test_pred_series.iloc[-1]),
    }
