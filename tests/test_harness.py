from pathlib import Path
import pytest
from src.harness.config import AgentRuntimeConfig
from src.harness.registry import ToolRegistry, ToolSpec
from src.harness.runtime import AgentHarness, HarnessContext
from src.harness.tracing import TraceRecorder

def test_registry_enforces_permissions(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "secretish",
            "read",
            lambda context: 7,
            frozenset({"quant"}),
        )
    )
    harness = AgentHarness(
        registry,
        config=AgentRuntimeConfig(
            tool_timeout_seconds=2,
            tracing_enabled=True,
        ),
        tracer=TraceRecorder(
            tmp_path / "trace.jsonl"
        ),
    )

    assert (
        harness.call_tool(
            run_id="r1",
            agent_name="quant",
            tool_name="secretish",
            context=HarnessContext("AAPL"),
        )
        == 7
    )

    with pytest.raises(PermissionError):
        harness.call_tool(
            run_id="r2",
            agent_name="research",
            tool_name="secretish",
            context=HarnessContext("AAPL"),
        )

def test_trace_does_not_require_llm(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "x",
            "read",
            lambda context: {"ok": True},
            frozenset({"*"}),
        )
    )
    tracer = TraceRecorder(
        tmp_path / "trace.jsonl"
    )
    harness = AgentHarness(
        registry,
        tracer=tracer,
    )
    harness.call_tool(
        run_id="abc",
        agent_name="orchestrator",
        tool_name="x",
        context=HarnessContext("AAPL"),
    )
    rows = tracer.read("abc")
    assert [
        row["status"]
        for row in rows
    ] == [
        "started",
        "completed",
    ]
