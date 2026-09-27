"""Machine-readable evaluation metrics for the agentic stack."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence
from pydantic import BaseModel, Field

_NUM = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")

class EvaluationSummary(BaseModel):
    routing_exact_match: float = 0.0
    routing_micro_precision: float = 0.0
    routing_micro_recall: float = 0.0
    citation_precision: float = 0.0
    unsupported_numeric_rate: float = 0.0
    cases: int = 0
    notes: List[str] = Field(default_factory=list)

def evaluate_routing(
    actual: Sequence[Sequence[str]],
    expected: Sequence[Sequence[str]],
) -> tuple[float, float, float]:
    if len(actual) != len(expected):
        raise ValueError("actual and expected must have equal length")
    if not actual:
        return 0.0, 0.0, 0.0

    exact = 0
    tp = fp = fn = 0
    for actual_row, expected_row in zip(actual, expected):
        actual_set = set(actual_row)
        expected_set = set(expected_row)
        exact += int(actual_set == expected_set)
        tp += len(actual_set & expected_set)
        fp += len(actual_set - expected_set)
        fn += len(expected_set - actual_set)

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return exact / len(actual), precision, recall

def citation_precision(
    cited_ids: Iterable[str],
    available_ids: Iterable[str],
) -> float:
    cited = list(cited_ids)
    available = set(available_ids)
    if not cited:
        return 1.0
    return (
        sum(1 for item in cited if item in available)
        / len(cited)
    )

def unsupported_numeric_rate(
    text: str,
    canonical_tokens: Iterable[str],
) -> float:
    numbers = _NUM.findall(text)
    if not numbers:
        return 0.0
    canonical = set(canonical_tokens)
    unsupported = sum(
        1
        for number in numbers
        if (
            number not in canonical
            and number.rstrip("%") not in canonical
        )
    )
    return unsupported / len(numbers)

def load_routing_cases(path: str | Path) -> list[dict]:
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )
