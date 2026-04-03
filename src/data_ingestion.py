"""Data ingestion module for downloading and storing market data."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


class DataIngestionError(RuntimeError):
    """Raised when market data ingestion fails after allowed recovery options."""


@dataclass
class IngestionMetadata:
    """Data provenance and ingestion execution details."""

    ticker: str
    start_date: str
    end_date: str
    interval: str
    source_type: str
    status: str
    attempts: int
    warnings: List[str]
    error_messages: List[str]


@dataclass
class IngestionResult:
    """Market data plus metadata describing provenance and status."""

    data: pd.DataFrame
    metadata: IngestionMetadata


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
        timeout_seconds: int = 30,
        max_retries: int = 3,
        initial_retry_delay_seconds: float = 1.0,
        backoff_factor: float = 2.0,
        allow_cache_fallback: bool = False,
        allow_demo_fallback: bool = False,
    ) -> IngestionResult:
        """Download OHLCV data with strict validation and explicit fallback policy."""
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        logger.info(
            "Starting market data ingestion for ticker=%s start=%s end=%s interval=%s",
            ticker,
            start_date,
            end_date,
            interval,
        )

        df = pd.DataFrame()
        last_error: Exception | None = None
        error_messages: List[str] = []
        warnings: List[str] = []
        attempts = 0

        for attempt in range(1, max_retries + 1):
            attempts = attempt
            try:
                logger.info("Downloading data from yfinance (attempt %s/%s)", attempt, max_retries)
                df = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    timeout=timeout_seconds,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                error_messages.append(f"Attempt {attempt}: {exc}")
                df = pd.DataFrame()
                logger.warning("Market download failed on attempt %s: %s", attempt, exc)

            if not df.empty:
                normalized = self._normalize_download_output(df)
                validated = self._validate_ohlcv(normalized)
                metadata = IngestionMetadata(
                    ticker=ticker.upper(),
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    source_type="fresh",
                    status="success",
                    attempts=attempts,
                    warnings=warnings,
                    error_messages=error_messages,
                )
                return IngestionResult(data=validated, metadata=metadata)

            error_messages.append(f"Attempt {attempt}: empty dataframe returned by yfinance")
            logger.warning("Received empty dataframe on attempt %s/%s", attempt, max_retries)

            if attempt < max_retries:
                sleep_seconds = initial_retry_delay_seconds * (backoff_factor ** (attempt - 1))
                logger.info("Retrying download after %.1f seconds", sleep_seconds)
                time.sleep(sleep_seconds)

        if allow_cache_fallback:
            logger.warning("Attempting explicit cache fallback for ticker=%s", ticker)
            cached = self._load_cached_data(ticker=ticker, start_date=start_date, end_date=end_date)
            if not cached.empty:
                warnings.append("Using cached market data because fresh download failed.")
                metadata = IngestionMetadata(
                    ticker=ticker.upper(),
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    source_type="cached",
                    status="degraded",
                    attempts=attempts,
                    warnings=warnings,
                    error_messages=error_messages,
                )
                return IngestionResult(data=cached, metadata=metadata)
            error_messages.append("Cache fallback requested but no valid cached file found.")

        if allow_demo_fallback:
            logger.warning("Using demo fallback data for ticker=%s", ticker)
            synthetic = self._generate_synthetic_data(ticker=ticker, start_date=start_date, end_date=end_date)
            warnings.append("Using synthetic demo market data; outputs are not based on live market history.")
            metadata = IngestionMetadata(
                ticker=ticker.upper(),
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                source_type="demo",
                status="degraded",
                attempts=attempts,
                warnings=warnings,
                error_messages=error_messages,
            )
            return IngestionResult(data=synthetic, metadata=metadata)

        error_context = (
            f"Market data ingestion failed for ticker={ticker}, range={start_date} to {end_date}. "
            "No valid fresh data available and fallback policy did not provide usable data."
        )
        if last_error is not None:
            error_context = f"{error_context} Last error: {last_error}"
        if error_messages:
            error_context = f"{error_context} Errors: {' | '.join(error_messages)}"
        raise DataIngestionError(error_context)

    def _normalize_download_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance output shape before validation."""
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)

        if "Date" not in out.columns:
            out = out.reset_index()

        return out

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns and ensure non-empty OHLCV dataset."""
        if df.empty:
            raise DataIngestionError("Downloaded dataframe is empty.")

        required_with_date = ["Date", *self.REQUIRED_COLUMNS]
        missing_cols = [c for c in required_with_date if c not in df.columns]
        if missing_cols:
            raise DataIngestionError(f"Downloaded data missing required columns: {missing_cols}")

        out = df[required_with_date].copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])

        out = out.dropna(subset=self.REQUIRED_COLUMNS)
        if out.empty:
            raise DataIngestionError("No usable OHLCV rows available after validation.")

        return out.reset_index(drop=True)

    def _load_cached_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load existing raw CSV data if available and filter date range."""
        path = self.raw_data_dir / f"{ticker.upper()}_raw.csv"
        if not path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

        try:
            validated = self._validate_ohlcv(df)
        except DataIngestionError:
            return pd.DataFrame()

        validated["Date"] = pd.to_datetime(validated["Date"], errors="coerce")
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        validated = validated[(validated["Date"] >= start_ts) & (validated["Date"] <= end_ts)].copy()
        if validated.empty:
            return pd.DataFrame()

        return validated.reset_index(drop=True)

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
    timeout_seconds: int = 30,
    max_retries: int = 3,
    initial_retry_delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    allow_cache_fallback: bool = False,
    allow_demo_fallback: bool = False,
) -> IngestionResult:
    """Convenience function for one-shot ingestion with explicit fallback policy."""
    ingestor = MarketDataIngestor(raw_data_dir=raw_data_dir)
    result = ingestor.download_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        initial_retry_delay_seconds=initial_retry_delay_seconds,
        backoff_factor=backoff_factor,
        allow_cache_fallback=allow_cache_fallback,
        allow_demo_fallback=allow_demo_fallback,
    )
    ingestor.save_raw_data(result.data, ticker=ticker)
    return result
