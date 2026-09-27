"""Retrieval facade used by agents, CLI, Streamlit and MCP."""
from __future__ import annotations
from typing import List
from src.agentic_schemas import EvidenceChunk, RetrievalQuery

class EvidenceRetriever:
    def __init__(self, store) -> None:
        self.store = store

    def retrieve(
        self,
        query: str,
        *,
        ticker: str | None = None,
        top_k: int = 6,
        document_types: List[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> List[EvidenceChunk]:
        return self.store.search(
            RetrievalQuery(
                query=query,
                ticker=ticker,
                top_k=top_k,
                document_types=document_types or [],
                start_date=start_date,
                end_date=end_date,
            )
        )
