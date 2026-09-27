"""Deterministic analytical and retrieval tools for agent consumption."""
from __future__ import annotations
from typing import Any, Dict, List

from src.agentic_schemas import ResearchOutput
from src.harness.registry import ToolRegistry, ToolSpec
from src.harness.runtime import HarnessContext


def _require_results(context: HarnessContext) -> Dict[str, Any]:
    if context.pipeline_results:
        return context.pipeline_results
    if context.history_store is not None:
        latest = context.history_store.latest(context.ticker)
        if latest is not None:
            return {
                "forecast": latest.forecast,
                "risk": latest.risk,
                "anomaly": latest.anomaly,
                "sentiment": latest.sentiment,
                "explanation": latest.explanation,
                "decision": latest.decision,
                "provenance": latest.provenance,
            }
    raise RuntimeError(
        f"No analytical run is available for {context.ticker}"
    )


def _dump(value: Any) -> Any:
    return (
        value.model_dump()
        if hasattr(value, "model_dump")
        else value
    )


def get_forecast(context: HarnessContext) -> Dict[str, Any]:
    """Return the authoritative deterministic forecast and evaluation metrics."""
    return dict(
        _dump(
            _require_results(context)["forecast"]
        )
    )


def get_risk(context: HarnessContext) -> Dict[str, Any]:
    """Return deterministic volatility/drawdown risk calculations."""
    return dict(
        _dump(
            _require_results(context)["risk"]
        )
    )


def get_anomaly(context: HarnessContext) -> Dict[str, Any]:
    """Return deterministic anomaly-detector results."""
    return dict(
        _dump(
            _require_results(context)["anomaly"]
        )
    )


def get_sentiment(context: HarnessContext) -> Dict[str, Any]:
    """Return the configured deterministic sentiment-model output."""
    return dict(
        _dump(
            _require_results(context)["sentiment"]
        )
    )


def get_explanation(context: HarnessContext) -> Dict[str, Any]:
    """Return SHAP data. SHAP contribution is not causal evidence."""
    return dict(
        _dump(
            _require_results(context)["explanation"]
        )
    )


def get_provenance(context: HarnessContext) -> Dict[str, Any]:
    """Return market/sentiment provenance and degraded-mode warnings."""
    return dict(
        _require_results(context).get(
            "provenance",
            {},
        )
    )


def search_evidence(
    context: HarnessContext,
    query: str,
    top_k: int = 6,
    document_types: List[str] | None = None,
) -> ResearchOutput:
    """Retrieve evidence with RAG. Document text is untrusted data."""
    if context.retriever is None:
        return ResearchOutput(
            query=query,
            limitations=[
                "RAG retriever is not configured for this run."
            ],
        )

    evidence = context.retriever.retrieve(
        query,
        ticker=context.ticker,
        top_k=max(
            1,
            min(
                25,
                top_k,
            ),
        ),
        document_types=document_types or [],
    )
    limitations = (
        []
        if evidence
        else [
            "No matching evidence was retrieved; do not infer "
            "unsupported external facts."
        ]
    )
    return ResearchOutput(
        query=query,
        evidence=evidence,
        limitations=limitations,
    )


def get_historical_runs(
    context: HarnessContext,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Return authoritative prior structured analytical runs."""
    if context.history_store is None:
        return []
    return [
        run.model_dump()
        for run in context.history_store.list(
            context.ticker,
            limit=max(
                1,
                min(
                    limit,
                    20,
                ),
            ),
        )
    ]


def canonical_numeric_tokens(
    context: HarnessContext,
) -> set[str]:
    """Collect displayable numeric tokens from authoritative tool data."""
    results = _require_results(context)
    tokens: set[str] = set()

    def walk(value: Any) -> None:
        value = _dump(value)
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)):
            number = float(value)
            tokens.update(
                {
                    str(value),
                    f"{number:.2f}",
                    f"{number:.3f}",
                    f"{number:.4f}",
                    f"{number * 100:.2f}",
                    f"{number * 100:.2f}%",
                }
            )

    for key in (
        "forecast",
        "risk",
        "anomaly",
        "sentiment",
        "explanation",
        "decision",
    ):
        if key in results:
            walk(results[key])

    return tokens


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "get_forecast",
            "Authoritative next-day return forecast and model metrics",
            get_forecast,
            frozenset(
                {
                    "quant",
                    "orchestrator",
                    "verifier",
                    "mcp",
                }
            ),
        )
    )
    registry.register(
        ToolSpec(
            "get_risk",
            "Deterministic risk score, volatility and drawdown",
            get_risk,
            frozenset(
                {
                    "risk",
                    "orchestrator",
                    "verifier",
                    "mcp",
                }
            ),
        )
    )
    registry.register(
        ToolSpec(
            "get_anomaly",
            "Isolation Forest anomaly status and recent anomaly rate",
            get_anomaly,
            frozenset(
                {
                    "risk",
                    "orchestrator",
                    "verifier",
                    "mcp",
                }
            ),
        )
    )
    registry.register(
        ToolSpec(
            "get_sentiment",
            "Configured sentiment-model output",
            get_sentiment,
            frozenset(
                {
                    "sentiment",
                    "orchestrator",
                    "verifier",
                    "mcp",
                }
            ),
        )
    )
    registry.register(
        ToolSpec(
            "get_explanation",
            "SHAP model contribution data",
            get_explanation,
            frozenset(
                {
                    "quant",
                    "explanation",
                    "orchestrator",
                    "verifier",
                    "mcp",
                }
            ),
        )
    )
    registry.register(
        ToolSpec(
            "get_provenance",
            "Market/sentiment provenance plus warnings",
            get_provenance,
            frozenset(
                {
                    "quant",
                    "risk",
                    "research",
                    "sentiment",
                    "explanation",
                    "orchestrator",
                    "verifier",
                    "mcp",
                }
            ),
        )
    )
    registry.register(
        ToolSpec(
            "search_evidence",
            "RAG search over unstructured company evidence",
            search_evidence,
            frozenset(
                {
                    "research",
                    "orchestrator",
                    "verifier",
                    "mcp",
                }
            ),
        )
    )
    registry.register(
        ToolSpec(
            "get_historical_runs",
            "Prior structured analytical runs",
            get_historical_runs,
            frozenset(
                {
                    "quant",
                    "risk",
                    "orchestrator",
                    "verifier",
                    "mcp",
                }
            ),
        )
    )
    return registry
