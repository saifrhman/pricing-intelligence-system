"""Embedding providers with a deterministic offline default."""
from __future__ import annotations
import hashlib
import math
import os
import re
from typing import Iterable, List, Protocol

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.]{1,}")

class Embedder(Protocol):
    def embed(self, texts: Iterable[str]) -> List[List[float]]: ...

class HashingEmbedder:
    """Dependency-light feature hashing embedder for local/offline retrieval."""
    def __init__(self, dimension: int = 384) -> None:
        if dimension < 64:
            raise ValueError("dimension must be >= 64")
        self.dimension = dimension

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm:
            vector = [v / norm for v in vector]
        return vector

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

class OpenAIEmbedder:
    """Optional OpenAI embeddings provider, imported lazily."""
    def __init__(self, model: str | None = None) -> None:
        self.model = (
            model
            or os.getenv("PRICING_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        values = list(texts)
        if not values:
            return []
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenAI embeddings requested but the openai package is unavailable"
            ) from exc
        client = OpenAI()
        response = client.embeddings.create(model=self.model, input=values)
        return [list(item.embedding) for item in response.data]
