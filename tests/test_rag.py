from pathlib import Path
from src.rag.ingestion import DocumentRecord, chunk_document
from src.rag.vector_store import JsonVectorStore
from src.agentic_schemas import RetrievalQuery

def test_chunk_and_retrieve_with_ticker_filter(tmp_path: Path):
    record = DocumentRecord(
        document_id="doc1",
        ticker="AAPL",
        document_type="10-Q",
        source="test",
        date="2026-08-01",
        text=(
            "Apple margin guidance and services growth remain important. "
            * 30
        ),
    )
    chunks = chunk_document(
        record,
        chunk_size=300,
        overlap=40,
    )
    assert len(chunks) > 1

    store = JsonVectorStore(
        tmp_path / "evidence.json"
    )
    assert store.upsert(chunks) == len(chunks)

    found = store.search(
        RetrievalQuery(
            query="margin guidance services growth",
            ticker="AAPL",
            top_k=3,
        )
    )
    assert found
    assert all(
        item.ticker == "AAPL"
        for item in found
    )
    assert all(
        item.relevance_score is not None
        for item in found
    )

    none = store.search(
        RetrievalQuery(
            query="margin",
            ticker="MSFT",
            top_k=3,
        )
    )
    assert none == []
