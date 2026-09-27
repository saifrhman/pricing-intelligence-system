"""CLI for indexing local unstructured evidence."""
from __future__ import annotations
import argparse
import os
from src.rag.ingestion import ingest_directory
from src.rag.vector_store import build_vector_store_from_env

def main() -> None:
    parser = argparse.ArgumentParser(description="Index local company evidence for RAG.")
    parser.add_argument("directory")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--document-type", default="local")
    parser.add_argument("--source", default="local")
    parser.add_argument("--store", default="data/vector_store/evidence.json")
    args = parser.parse_args()
    chunks = ingest_directory(
        args.directory,
        ticker=args.ticker,
        document_type=args.document_type,
        source=args.source,
    )
    os.environ["PRICING_VECTOR_STORE"] = args.store
    count = build_vector_store_from_env().upsert(chunks)
    print(f"Indexed {count} chunks for {args.ticker.upper()} into {args.store}")

if __name__ == "__main__":
    main()
