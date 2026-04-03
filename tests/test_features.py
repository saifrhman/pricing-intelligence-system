import pandas as pd

from src.feature_engineering import engineer_features


def test_engineer_features_creates_target_and_lags() -> None:
    n = 80
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(range(100, 100 + n), dtype=float)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": [1_000_000 + i * 1000 for i in range(n)],
        }
    )

    out = engineer_features(df)

    assert "target_next_return" in out.columns
    assert "lag_return_1" in out.columns
    assert "lag_return_5" in out.columns
    assert "volatility_20" in out.columns
    assert len(out) > 0
