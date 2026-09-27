"""Document loading and chunking for financial evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.agentic_schemas import EvidenceChunk

class DocumentRecord(BaseModel):
    document_id: str
    text: str
    source: str
    ticker: Optional[str] = None
    document_type: str = "unknown"
    source_url: Optional[str] = None
    date: Optional[str] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

def _stable_id(*parts: str) -> str:
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]

def chunk_document(
    record: DocumentRecord,
    chunk_size: int = 1200,
    overlap: int = 180,
) -> List[EvidenceChunk]:
    if chunk_size < 200:
        raise ValueError("chunk_size must be >= 200")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = " ".join(record.text.split())
    if not text:
        return []

    chunks: List[EvidenceChunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        body = text[start:end].strip()
        if body:
            chunks.append(
                EvidenceChunk(
                    chunk_id=_stable_id(
                        record.document_id,
                        str(index),
                        body[:80],
                    ),
                    document_id=record.document_id,
                    ticker=record.ticker.upper() if record.ticker else None,
                    document_type=record.document_type,
                    source=record.source,
                    source_url=record.source_url,
                    date=record.date,
                    section=record.section,
                    text=body,
                    metadata={
                        **record.metadata,
                        "chunk_index": index,
                    },
                )
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        index += 1
    return chunks

def _load_text_file(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, str):
            return raw
        return json.dumps(raw, ensure_ascii=False, indent=2)
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF ingestion requires pypdf; install project dependencies"
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported document type: {path.suffix}")

def ingest_directory(
    directory: str | Path,
    *,
    ticker: str | None = None,
    document_type: str = "local",
    source: str = "local",
) -> List[EvidenceChunk]:
    root = Path(directory)
    if not root.exists():
        return []

    chunks: List[EvidenceChunk] = []
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".txt", ".md", ".json", ".pdf"}
    )
    for path in paths:
        text = _load_text_file(path)
        record = DocumentRecord(
            document_id=_stable_id(
                str(path.resolve()),
                str(path.stat().st_mtime_ns),
            ),
            text=text,
            source=source,
            ticker=ticker,
            document_type=document_type,
            metadata={"path": str(path)},
        )
        chunks.extend(chunk_document(record))
    return chunks
