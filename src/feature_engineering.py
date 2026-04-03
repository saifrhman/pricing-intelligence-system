"""Feature engineering module tailored for return forecasting and risk context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # If there are no losses in the window, RSI is conventionally 100.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # If both gains and losses are zero, market is flat; use neutral RSI.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)

    return rsi


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create finance-oriented features and next-day return target."""
    if raw_df.empty:
        raise ValueError("Input dataframe is empty.")

    expected_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = expected_columns - set(raw_df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {sorted(missing)}")

    df = raw_df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    for lag in range(1, 6):
        df[f"lag_return_{lag}"] = df["daily_return"].shift(lag)

    for window in (5, 10, 20):
        df[f"ma_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"volatility_{window}"] = df["daily_return"].rolling(window=window).std()
        df[f"volume_ma_{window}"] = df["Volume"].rolling(window=window).mean()

    df["volume_change"] = df["Volume"].pct_change()
    df["momentum_5"] = df["Close"].pct_change(periods=5)
    df["momentum_10"] = df["Close"].pct_change(periods=10)

    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["rsi_14"] = _compute_rsi(df["Close"], window=14)

    df["rolling_max_20"] = df["Close"].rolling(window=20).max()
    df["drawdown_20"] = (df["Close"] / df["rolling_max_20"]) - 1.0

    df["target_next_return"] = df["daily_return"].shift(-1)

    helper_cols = ["ema_12", "ema_26", "rolling_max_20"]
    df = df.drop(columns=helper_cols)
    df = df.dropna().reset_index(drop=True)

    return df


def save_processed_data(
    processed_df: pd.DataFrame,
    ticker: str,
    processed_data_dir: str | Path = "data/processed",
) -> Path:
    """Save processed features to CSV."""
    output_dir = Path(processed_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{ticker.upper()}_processed.csv"
    processed_df.to_csv(output_path, index=False)
    return output_path
