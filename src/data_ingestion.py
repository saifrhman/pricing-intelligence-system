"""Data ingestion module for downloading and storing market data."""

from __future__ import annotations

from pathlib import Path

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
    ) -> pd.DataFrame:
        """Download OHLCV data and perform light validation."""
        if not ticker:
            raise ValueError("Ticker cannot be empty.")

        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df.empty:
            raise ValueError(
                f"No data returned for ticker={ticker}, range={start_date} to {end_date}."
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Downloaded data missing required columns: {missing_cols}")

        df = df[self.REQUIRED_COLUMNS].copy()
        df = df.reset_index()
        if "Date" not in df.columns:
            raise ValueError("Downloaded data does not include Date column.")

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
    )
    ingestor.save_raw_data(df, ticker=ticker)
    return df
