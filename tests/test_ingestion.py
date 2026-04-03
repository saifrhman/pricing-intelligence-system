from __future__ import annotations

import pandas as pd
import pytest

from src.data_ingestion import DataIngestionError, MarketDataIngestor


@pytest.fixture
def ingestor(tmp_path):
    return MarketDataIngestor(raw_data_dir=tmp_path)


def test_ingestion_failure_without_fallback_raises(monkeypatch, ingestor) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("timeout")

    monkeypatch.setattr("src.data_ingestion.yf.download", _raise)

    with pytest.raises(DataIngestionError):
        ingestor.download_data(
            ticker="AAPL",
            start_date="2020-01-01",
            end_date="2020-12-31",
            max_retries=2,
            initial_retry_delay_seconds=0,
            allow_cache_fallback=False,
            allow_demo_fallback=False,
        )


def test_ingestion_uses_cache_when_allowed(monkeypatch, ingestor, tmp_path) -> None:
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    cached_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1_000_000,
        }
    )
    cached_df.to_csv(tmp_path / "AAPL_raw.csv", index=False)

    monkeypatch.setattr("src.data_ingestion.yf.download", lambda *args, **kwargs: pd.DataFrame())

    result = ingestor.download_data(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2020-12-31",
        max_retries=1,
        initial_retry_delay_seconds=0,
        allow_cache_fallback=True,
        allow_demo_fallback=False,
    )

    assert not result.data.empty
    assert result.metadata.source_type == "cached"
    assert result.metadata.status == "degraded"


def test_empty_download_without_cache_raises(monkeypatch, ingestor) -> None:
    monkeypatch.setattr("src.data_ingestion.yf.download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(DataIngestionError):
        ingestor.download_data(
            ticker="AAPL",
            start_date="2020-01-01",
            end_date="2020-12-31",
            max_retries=1,
            initial_retry_delay_seconds=0,
            allow_cache_fallback=False,
            allow_demo_fallback=False,
        )
