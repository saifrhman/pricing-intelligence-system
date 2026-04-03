"""Optional financial sentiment module based on transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.schemas import SentimentOutput

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - handled for environments without transformers
    pipeline = None


@dataclass
class HeadlineSentiment:
    headline: str
    label: str
    score: float


class SentimentAnalyzer:
    """Analyze sentiment from headlines with FinBERT or a fallback rule-based model."""

    def __init__(
        self,
        use_transformer: bool = True,
        model_name: str = "ProsusAI/finbert",
    ) -> None:
        self.use_transformer = use_transformer and pipeline is not None
        self.model_name = model_name
        self._clf = None

    def _load_model(self):
        if self._clf is None and self.use_transformer:
            self._clf = pipeline("text-classification", model=self.model_name)
        return self._clf

    def analyze_headlines(self, headlines: List[str]) -> List[HeadlineSentiment]:
        """Score each headline and return normalized outputs."""
        if not headlines:
            return []

        if self.use_transformer:
            try:
                clf = self._load_model()
                preds = clf(headlines)
                return [
                    HeadlineSentiment(
                        headline=h,
                        label=str(p["label"]).lower(),
                        score=float(p["score"]),
                    )
                    for h, p in zip(headlines, preds)
                ]
            except Exception:
                # Fallback to lexicon if model download/inference fails.
                pass

        return [self._rule_based_sentiment(h) for h in headlines]

    def _rule_based_sentiment(self, text: str) -> HeadlineSentiment:
        positive_terms = {"beat", "growth", "upgrade", "surge", "strong", "profit"}
        negative_terms = {"miss", "downgrade", "decline", "weak", "loss", "lawsuit"}

        text_lower = text.lower()
        pos_hits = sum(term in text_lower for term in positive_terms)
        neg_hits = sum(term in text_lower for term in negative_terms)

        if pos_hits > neg_hits:
            return HeadlineSentiment(text, "positive", min(0.6 + 0.1 * pos_hits, 0.95))
        if neg_hits > pos_hits:
            return HeadlineSentiment(text, "negative", min(0.6 + 0.1 * neg_hits, 0.95))
        return HeadlineSentiment(text, "neutral", 0.5)

    def aggregate(self, headlines: List[str], source: str = "headlines") -> SentimentOutput:
        """Aggregate multiple headline-level scores into one signal."""
        scored = self.analyze_headlines(headlines)
        if not scored:
            return SentimentOutput(
                available=False,
                source=source,
                sentiment_label="unavailable",
                sentiment_score=0.0,
                headline_count=0,
            )

        signed_scores = []
        for item in scored:
            if "positive" in item.label:
                signed_scores.append(item.score)
            elif "negative" in item.label:
                signed_scores.append(-item.score)
            else:
                signed_scores.append(0.0)

        mean_score = float(sum(signed_scores) / len(signed_scores))
        if mean_score > 0.15:
            label = "positive"
        elif mean_score < -0.15:
            label = "negative"
        else:
            label = "neutral"

        return SentimentOutput(
            available=True,
            source=source,
            sentiment_label=label,
            sentiment_score=max(-1.0, min(1.0, mean_score)),
            headline_count=len(scored),
        )


def default_mock_headlines(ticker: str) -> List[str]:
    """Mock headlines to keep pipeline runnable without external news API."""
    t = ticker.upper()
    return [
        f"{t} reports stronger-than-expected quarterly revenue growth.",
        f"Analysts maintain neutral outlook on {t} amid mixed macro signals.",
        f"{t} faces margin pressure concerns from rising input costs.",
    ]
