"""Local-first retrieval subsystem for unstructured financial evidence."""
from .embeddings import HashingEmbedder
from .ingestion import DocumentRecord, chunk_document, ingest_directory
from .retriever import EvidenceRetriever
from .vector_store import JsonVectorStore, QdrantVectorStore, build_vector_store_from_env

__all__ = [
    "HashingEmbedder",
    "DocumentRecord",
    "chunk_document",
    "ingest_directory",
    "EvidenceRetriever",
    "JsonVectorStore",
    "QdrantVectorStore",
    "build_vector_store_from_env",
]
