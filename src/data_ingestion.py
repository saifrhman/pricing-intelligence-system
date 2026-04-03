"""Data ingestion module for downloading and storing market data."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


class MarketDataIngestor:
    """Download and validate market data from yfinance."""

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, raw_data_dir: str | Path = "data/raw") -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def download_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        max_retries: int = 3,
        retry_sleep_seconds: float = 2.0,
        allow_cache_fallback: bool = True,
        allow_synthetic_fallback: bool = True,
    ) -> pd.DataFrame:
        """Download OHLCV data with resilient fallback behavior."""
        if not ticker:
            raise ValueError("Ticker cannot be empty.")

        df = pd.DataFrame()
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    timeout=30,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                df = pd.DataFrame()

            if not df.empty:
                break

            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Downloaded data missing required columns: {missing_cols}")

            df = df[self.REQUIRED_COLUMNS].copy()
            df = df.reset_index()
            if "Date" not in df.columns:
                raise ValueError("Downloaded data does not include Date column.")

            df["data_source"] = "yfinance"
            return df

        if allow_cache_fallback:
            cached = self._load_cached_data(ticker=ticker, start_date=start_date, end_date=end_date)
            if not cached.empty:
                cached["data_source"] = "cache"
                return cached

        if allow_synthetic_fallback:
            synthetic = self._generate_synthetic_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            synthetic["data_source"] = "synthetic"
            return synthetic

        error_context = f"No data returned for ticker={ticker}, range={start_date} to {end_date}."
        if last_error is not None:
            error_context = f"{error_context} Last error: {last_error}"
        raise ValueError(error_context)

    def _load_cached_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load existing raw CSV data if available and filter date range."""
        path = self.raw_data_dir / f"{ticker.upper()}_raw.csv"
        if not path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

        required_with_date = ["Date", *self.REQUIRED_COLUMNS]
        if any(col not in df.columns for col in required_with_date):
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()
        if df.empty:
            return pd.DataFrame()

        return df[required_with_date].reset_index(drop=True)

    def _generate_synthetic_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate deterministic synthetic OHLCV data for offline demos."""
        dates = pd.bdate_range(start=start_date, end=end_date)
        if len(dates) < 80:
            raise ValueError("Date range too short for fallback synthetic data generation.")

        seed = int.from_bytes(ticker.upper().encode("utf-8"), "little") % (2**32 - 1)
        rng = np.random.default_rng(seed)

        # Geometric random walk with mild drift and realistic daily volatility.
        drift = 0.0003
        vol = 0.015
        daily_returns = rng.normal(drift, vol, size=len(dates))

        close = np.zeros(len(dates), dtype=float)
        close[0] = 100.0
        for i in range(1, len(dates)):
            close[i] = close[i - 1] * (1.0 + daily_returns[i])

        open_price = close * (1.0 + rng.normal(0.0, 0.002, size=len(dates)))
        high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.0005, 0.006, size=len(dates)))
        low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.0005, 0.006, size=len(dates)))
        volume = rng.integers(2_000_000, 12_000_000, size=len(dates))

        return pd.DataFrame(
            {
                "Date": dates,
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )

        return df

    def save_raw_data(self, df: pd.DataFrame, ticker: str) -> Path:
        """Save raw dataframe to CSV."""
        filename = f"{ticker.upper()}_raw.csv"
        output_path = self.raw_data_dir / filename
        df.to_csv(output_path, index=False)
        return output_path


def fetch_market_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    raw_data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Convenience function for one-shot ingestion."""
    ingestor = MarketDataIngestor(raw_data_dir=raw_data_dir)
    df = ingestor.download_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        max_retries=3,
        retry_sleep_seconds=2.0,
        allow_cache_fallback=True,
        allow_synthetic_fallback=True,
    )
    ingestor.save_raw_data(df, ticker=ticker)
    return df
