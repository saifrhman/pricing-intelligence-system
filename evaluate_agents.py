"""Offline routing evaluation; LLM evaluations can build on the same dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
from src.agentic.engine import AgenticPricingEngine
from src.evaluation.evaluator import EvaluationSummary, evaluate_routing, load_routing_cases

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--dataset", default="src/evaluation/datasets/routing_cases.json")
    parser.add_argument("--output", default="outputs/evaluation/routing_eval.json")
    args = parser.parse_args()
    cases = load_routing_cases(args.dataset)
    actual = [AgenticPricingEngine.heuristic_plan(args.ticker, c["question"]).specialists for c in cases]
    expected = [c["expected_agents"] for c in cases]
    exact, precision, recall = evaluate_routing(actual, expected)
    summary = EvaluationSummary(
        routing_exact_match=exact,
        routing_micro_precision=precision,
        routing_micro_recall=recall,
        cases=len(cases),
        notes=["Offline router benchmark. LLM routing should be evaluated separately when API access is configured."],
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(summary.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
