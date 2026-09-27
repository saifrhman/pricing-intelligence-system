"""Environment-driven configuration for the agentic runtime."""
from __future__ import annotations
import os
from dataclasses import dataclass

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class AgentRuntimeConfig:
    model: str = "gpt-5.6-terra"
    max_steps: int = 12
    max_revision_cycles: int = 1
    tracing_enabled: bool = True
    tool_timeout_seconds: float = 20.0
    rag_top_k: int = 6

    @classmethod
    def from_env(cls) -> "AgentRuntimeConfig":
        return cls(
            model=os.getenv("PRICING_LLM_MODEL", "gpt-5.6-terra"),
            max_steps=max(1, int(os.getenv("PRICING_AGENT_MAX_STEPS", "12"))),
            max_revision_cycles=max(0, int(os.getenv("PRICING_AGENT_MAX_REVISIONS", "1"))),
            tracing_enabled=_env_bool("PRICING_AGENT_TRACING", True),
            tool_timeout_seconds=max(1.0, float(os.getenv("PRICING_TOOL_TIMEOUT_SECONDS", "20"))),
            rag_top_k=max(1, min(25, int(os.getenv("PRICING_RAG_TOP_K", "6")))),
        )
