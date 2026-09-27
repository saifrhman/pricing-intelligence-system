"""Session memory and authoritative structured analytical history."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from src.agentic_schemas import HistoricalRunSummary

class JsonSessionStore:
    def __init__(self, directory: str | Path = "outputs/sessions") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError("Invalid session_id")
        return self.directory / f"{safe}.json"

    def load(self, session_id: str) -> List[Dict[str, str]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def append(self, session_id: str, role: str, content: str) -> None:
        history = self.load(session_id)
        history.append(
            {
                "role": role,
                "content": content,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._path(session_id).write_text(
            json.dumps(history[-30:], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

class HistoricalRunStore:
    def __init__(self, directory: str | Path = "outputs/history") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_pipeline_results(
        results: Dict[str, Any],
        ticker: str,
        run_id: Optional[str] = None,
    ) -> HistoricalRunSummary:
        def dump(name: str) -> Dict[str, Any]:
            value = results[name]
            return value.model_dump() if hasattr(value, "model_dump") else dict(value)

        return HistoricalRunSummary(
            run_id=run_id or uuid4().hex,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ticker=ticker.upper(),
            forecast=dump("forecast"),
            risk=dump("risk"),
            anomaly=dump("anomaly"),
            sentiment=dump("sentiment"),
            explanation=dump("explanation"),
            decision=dump("decision"),
            provenance=dict(results.get("provenance", {})),
        )

    def save(self, summary: HistoricalRunSummary) -> Path:
        ticker_dir = self.directory / summary.ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        path = ticker_dir / (
            f"{summary.timestamp_utc.replace(':', '-')}_{summary.run_id}.json"
        )
        path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        return path

    def list(self, ticker: str, limit: int = 20) -> List[HistoricalRunSummary]:
        ticker_dir = self.directory / ticker.upper()
        if not ticker_dir.exists():
            return []
        paths = sorted(ticker_dir.glob("*.json"), reverse=True)[: max(1, limit)]
        return [
            HistoricalRunSummary.model_validate_json(p.read_text(encoding="utf-8"))
            for p in paths
        ]

    def latest(self, ticker: str) -> Optional[HistoricalRunSummary]:
        rows = self.list(ticker, limit=1)
        return rows[0] if rows else None
