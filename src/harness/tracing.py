"""Safe operational tracing without private chain-of-thought."""
from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Any, Dict, List
from src.agentic_schemas import ToolTraceEvent

_SENSITIVE_KEYS = {"api_key", "authorization", "password", "secret", "token"}

def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: ("***" if k.lower() in _SENSITIVE_KEYS else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value

class TraceRecorder:
    def __init__(
        self,
        path: str | Path = "outputs/traces/agent_runs.jsonl",
        enabled: bool = True,
    ) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: ToolTraceEvent) -> None:
        if not self.enabled:
            return
        payload = _redact(event.model_dump())
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read(self, run_id: str | None = None) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [r for r in rows if run_id is None or r.get("run_id") == run_id]
