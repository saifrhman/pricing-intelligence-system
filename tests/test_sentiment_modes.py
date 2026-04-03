from __future__ import annotations

import pytest

from src.sentiment import SentimentAnalysisError, SentimentAnalyzer


def test_sentiment_transformer_failure_raises_when_fallback_disabled(monkeypatch) -> None:
    analyzer = SentimentAnalyzer(use_transformer=True)

    def _raise():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(analyzer, "_load_model", _raise)

    with pytest.raises(SentimentAnalysisError):
        analyzer.aggregate(
            headlines=["Company beats earnings estimates"],
            source="manual",
            allow_rule_based_fallback=False,
        )


def test_sentiment_rule_based_fallback_is_explicit(monkeypatch) -> None:
    analyzer = SentimentAnalyzer(use_transformer=True)

    def _raise():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(analyzer, "_load_model", _raise)

    out = analyzer.aggregate(
        headlines=["Company reports strong growth"],
        source="demo_mock",
        allow_rule_based_fallback=True,
    )

    assert out.available is True
    assert "rule_based" in out.source
    assert out.source.startswith("demo_mock")
