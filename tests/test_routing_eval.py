from src.agentic.engine import AgenticPricingEngine
from src.evaluation.evaluator import evaluate_routing

def test_heuristic_router_selects_relevant_specialists():
    assert (
        AgenticPricingEngine.heuristic_plan(
            "AAPL",
            "What is the predicted return?",
        ).specialists
        == ["quant"]
    )
    assert (
        AgenticPricingEngine.heuristic_plan(
            "AAPL",
            "What features are driving the forecast?",
        ).specialists
        == [
            "quant",
            "explanation",
        ]
    )
    full = AgenticPricingEngine.heuristic_plan(
        "AAPL",
        "Give me a complete intelligence assessment.",
    ).specialists
    assert full == [
        "quant",
        "risk",
        "research",
        "sentiment",
        "explanation",
    ]

def test_routing_metrics():
    exact, precision, recall = evaluate_routing(
        [
            ["quant"],
            [
                "risk",
                "research",
            ],
        ],
        [
            ["quant"],
            ["risk"],
        ],
    )
    assert exact == 0.5
    assert 0 < precision < 1
    assert recall == 1
