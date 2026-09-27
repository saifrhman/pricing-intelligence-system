"""CLI for asking the agentic pricing-intelligence layer questions."""
from __future__ import annotations
import argparse
from src.agentic.engine import AgenticPricingEngine
from src.harness.config import AgentRuntimeConfig
from src.harness.runtime import AgentHarness, HarnessContext
from src.harness.state import HistoricalRunStore, JsonSessionStore
from src.harness.tracing import TraceRecorder
from src.rag.retriever import EvidenceRetriever
from src.rag.vector_store import build_vector_store_from_env
from src.tools.intelligence_tools import build_tool_registry

def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Pricing Intelligence agentic layer a grounded question.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--session-id", default="cli")
    parser.add_argument("--require-llm", action="store_true", help="Fail instead of deterministic fallback when the LLM runtime is unavailable.")
    args = parser.parse_args()
    config = AgentRuntimeConfig.from_env()
    history = HistoricalRunStore()
    context = HarnessContext(
        ticker=args.ticker.upper(),
        history_store=history,
        retriever=EvidenceRetriever(build_vector_store_from_env()),
    )
    harness = AgentHarness(
        build_tool_registry(),
        config=config,
        tracer=TraceRecorder(enabled=config.tracing_enabled),
    )
    engine = AgenticPricingEngine(harness, config=config, session_store=JsonSessionStore())
    report = engine.run(
        args.question,
        context,
        session_id=args.session_id,
        require_llm=args.require_llm,
    )
    print(report.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
