from pathlib import Path
from src.agentic.engine import AgenticPricingEngine
from src.harness.config import AgentRuntimeConfig
from src.harness.runtime import AgentHarness, HarnessContext
from src.harness.state import HistoricalRunStore
from src.harness.tracing import TraceRecorder
from src.tools.intelligence_tools import build_tool_registry
from tests.test_history import _results

def test_offline_agent_uses_deterministic_tools(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )
    history = HistoricalRunStore(
        tmp_path / "history"
    )
    history.save(
        history.from_pipeline_results(
            _results(),
            "AAPL",
            run_id="run1",
        )
    )

    config = AgentRuntimeConfig(
        tracing_enabled=True
    )
    harness = AgentHarness(
        build_tool_registry(),
        config,
        TraceRecorder(
            tmp_path / "trace.jsonl"
        ),
    )
    engine = AgenticPricingEngine(
        harness,
        config=config,
    )
    report = engine.run(
        "What is the predicted return?",
        HarnessContext(
            "AAPL",
            history_store=history,
        ),
    )

    assert "0.0100" in report.answer
    assert report.selected_agents == ["quant"]
    assert report.verification.passed
