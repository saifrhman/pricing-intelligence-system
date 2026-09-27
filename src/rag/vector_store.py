"""Persistent local and optional Qdrant vector stores."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Iterable, List
from src.agentic_schemas import EvidenceChunk, RetrievalQuery
from src.rag.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder

class JsonVectorStore:
    def __init__(
        self,
        path: str | Path = "data/vector_store/evidence.json",
        embedder: Embedder | None = None,
    ) -> None:
        self.path = Path(path)
        self.embedder = embedder or HashingEmbedder()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, rows: list[dict]) -> None:
        self.path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert(self, chunks: Iterable[EvidenceChunk]) -> int:
        incoming = list(chunks)
        if not incoming:
            return 0
        existing = {
            row["chunk"]["chunk_id"]: row
            for row in self._load()
        }
        vectors = self.embedder.embed(chunk.text for chunk in incoming)
        for chunk, vector in zip(incoming, vectors):
            existing[chunk.chunk_id] = {
                "chunk": chunk.model_dump(),
                "vector": vector,
            }
        self._save(list(existing.values()))
        return len(incoming)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        denom = (
            math.sqrt(sum(x * x for x in a))
            * math.sqrt(sum(y * y for y in b))
        )
        return (
            sum(x * y for x, y in zip(a, b)) / denom
            if denom
            else 0.0
        )

    @staticmethod
    def _date_ok(
        value: str | None,
        start: str | None,
        end: str | None,
    ) -> bool:
        if value is None:
            return start is None and end is None
        if start and value < start:
            return False
        if end and value > end:
            return False
        return True

    def search(self, request: RetrievalQuery) -> List[EvidenceChunk]:
        query_vec = self.embedder.embed([request.query])[0]
        scored: list[tuple[float, EvidenceChunk]] = []
        allowed_types = {
            item.lower()
            for item in request.document_types
        }

        for row in self._load():
            chunk = EvidenceChunk(**row["chunk"])
            if (
                request.ticker
                and (chunk.ticker or "").upper()
                != request.ticker.upper()
            ):
                continue
            if (
                allowed_types
                and chunk.document_type.lower()
                not in allowed_types
            ):
                continue
            if not self._date_ok(
                chunk.date,
                request.start_date,
                request.end_date,
            ):
                continue

            score = self._cosine(
                query_vec,
                row["vector"],
            )
            scored.append(
                (
                    score,
                    chunk.model_copy(
                        update={"relevance_score": float(score)}
                    ),
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            chunk
            for _, chunk in scored[: request.top_k]
        ]

class QdrantVectorStore:
    """Optional Qdrant-backed vector store for production-like deployments."""
    def __init__(
        self,
        collection: str = "pricing_evidence",
        *,
        url: str | None = None,
        api_key: str | None = None,
        local_path: str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "QdrantVectorStore requires qdrant-client"
            ) from exc

        self.collection = collection
        self.embedder = embedder or HashingEmbedder()
        if url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
            )
        else:
            self.client = QdrantClient(
                path=local_path or "data/qdrant"
            )

    def _ensure_collection(self, dimension: int) -> None:
        from qdrant_client import models
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(self, chunks: Iterable[EvidenceChunk]) -> int:
        from uuid import NAMESPACE_URL, uuid5
        from qdrant_client import models

        incoming = list(chunks)
        if not incoming:
            return 0

        vectors = self.embedder.embed(
            chunk.text
            for chunk in incoming
        )
        self._ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(incoming, vectors)
        ]
        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )
        return len(points)

    def search(self, request: RetrievalQuery) -> List[EvidenceChunk]:
        query_vec = self.embedder.embed([request.query])[0]
        if not self.client.collection_exists(self.collection):
            return []

        points = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=min(
                100,
                max(
                    request.top_k * 6,
                    request.top_k,
                ),
            ),
        ).points
        allowed_types = {
            item.lower()
            for item in request.document_types
        }
        rows: List[EvidenceChunk] = []

        for point in points:
            payload = dict(point.payload or {})
            try:
                chunk = EvidenceChunk(**payload)
            except Exception:
                continue
            if (
                request.ticker
                and (chunk.ticker or "").upper()
                != request.ticker.upper()
            ):
                continue
            if (
                allowed_types
                and chunk.document_type.lower()
                not in allowed_types
            ):
                continue
            if not JsonVectorStore._date_ok(
                chunk.date,
                request.start_date,
                request.end_date,
            ):
                continue
            rows.append(
                chunk.model_copy(
                    update={
                        "relevance_score": float(point.score)
                    }
                )
            )
            if len(rows) >= request.top_k:
                break
        return rows

def build_vector_store_from_env(
    embedder: Embedder | None = None,
):
    """Select Qdrant when configured, otherwise the offline JSON store."""
    import os

    if embedder is None and os.getenv("PRICING_EMBEDDING_MODEL", "").strip():
        embedder = OpenAIEmbedder()

    url = os.getenv("QDRANT_URL", "").strip()
    if url:
        return QdrantVectorStore(
            collection=os.getenv(
                "QDRANT_COLLECTION",
                "pricing_evidence",
            ),
            url=url,
            api_key=os.getenv("QDRANT_API_KEY") or None,
            embedder=embedder,
        )

    return JsonVectorStore(
        os.getenv(
            "PRICING_VECTOR_STORE",
            "data/vector_store/evidence.json",
        ),
        embedder=embedder,
    )
