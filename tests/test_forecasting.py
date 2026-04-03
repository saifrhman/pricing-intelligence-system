import numpy as np
import pandas as pd
import pytest

from src.forecasting import train_and_evaluate

pytest.importorskip("xgboost")


def test_train_and_evaluate_returns_expected_keys(tmp_path) -> None:
    rng = np.random.default_rng(42)
    n = 180
    dates = pd.date_range("2021-01-01", periods=n, freq="D")

    daily_return = rng.normal(0.0005, 0.01, size=n)
    lag_1 = np.roll(daily_return, 1)
    lag_2 = np.roll(daily_return, 2)

    df = pd.DataFrame(
        {
            "Date": dates,
            "daily_return": daily_return,
            "lag_return_1": lag_1,
            "lag_return_2": lag_2,
            "volatility_10": pd.Series(daily_return).rolling(10).std().fillna(0.01),
            "volume_change": rng.normal(0.0, 0.1, size=n),
            "target_next_return": np.roll(daily_return, -1),
        }
    ).iloc[3:-1]

    result = train_and_evaluate(
        df=df,
        ticker="TEST",
        model_output_dir=tmp_path / "models",
        plot_output_dir=tmp_path / "plots",
    )

    assert "best_model_name" in result
    assert "metrics_by_model" in result
    assert "predictions_df" in result
    assert len(result["predictions_df"]) > 0
