"""Execution harness: permissions, limits, timeouts, tracing and shared context."""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4
from src.harness.config import AgentRuntimeConfig
from src.harness.registry import ToolRegistry
from src.harness.tracing import TraceRecorder
from src.agentic_schemas import ToolTraceEvent

@dataclass
class HarnessContext:
    ticker: str
    pipeline_results: Optional[Dict[str, Any]] = None
    retriever: Any = None
    history_store: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)

class AgentHarness:
    def __init__(
        self,
        registry: ToolRegistry,
        config: AgentRuntimeConfig | None = None,
        tracer: TraceRecorder | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or AgentRuntimeConfig.from_env()
        self.tracer = tracer or TraceRecorder(enabled=self.config.tracing_enabled)

    def new_run_id(self) -> str:
        return uuid4().hex

    def call_tool(
        self,
        *,
        run_id: str,
        agent_name: str,
        tool_name: str,
        context: HarnessContext,
        **kwargs: Any,
    ) -> Any:
        spec = self.registry.get(tool_name, agent_name)
        self.tracer.record(
            ToolTraceEvent(
                run_id=run_id,
                event_type="tool",
                name=tool_name,
                status="started",
                metadata={"agent": agent_name},
            )
        )
        started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(spec.handler, context, **kwargs)
                result = future.result(timeout=self.config.tool_timeout_seconds)
        except FutureTimeout as exc:
            duration = (time.perf_counter() - started) * 1000
            self.tracer.record(
                ToolTraceEvent(
                    run_id=run_id,
                    event_type="tool",
                    name=tool_name,
                    status="failed",
                    duration_ms=duration,
                    metadata={"agent": agent_name, "error": "timeout"},
                )
            )
            raise TimeoutError(
                f"Tool '{tool_name}' exceeded {self.config.tool_timeout_seconds}s"
            ) from exc
        except Exception as exc:
            duration = (time.perf_counter() - started) * 1000
            self.tracer.record(
                ToolTraceEvent(
                    run_id=run_id,
                    event_type="tool",
                    name=tool_name,
                    status="failed",
                    duration_ms=duration,
                    metadata={
                        "agent": agent_name,
                        "error_type": type(exc).__name__,
                    },
                )
            )
            raise

        duration = (time.perf_counter() - started) * 1000
        self.tracer.record(
            ToolTraceEvent(
                run_id=run_id,
                event_type="tool",
                name=tool_name,
                status="completed",
                duration_ms=duration,
                metadata={"agent": agent_name},
            )
        )
        return result
