"""Grounding and trust-boundary checks."""
from __future__ import annotations
import re
from typing import Iterable, Mapping
from src.agentic_schemas import IntelligenceReportDraft, VerificationResult

_INSTRUCTION_LIKE = re.compile(
    r"(?i)(ignore (?:(?:all|any|the) )?previous|system message|developer message|you are chatgpt|execute this command)"
)
# Ground financial decimals/percentages without treating contextual integers
# such as "20-day volatility", dates, counts or model names as claims.
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\d+%)(?![A-Za-z])")

def mark_untrusted_document(text: str) -> str:
    """Wrap retrieved text so prompts treat it as evidence, never instructions."""
    flagged = bool(_INSTRUCTION_LIKE.search(text))
    marker = " [contains instruction-like text]" if flagged else ""
    return f"<untrusted_evidence{marker}>\n{text}\n</untrusted_evidence>"

def verify_draft(
    draft: IntelligenceReportDraft,
    *,
    canonical_numbers: Iterable[str],
    available_evidence_ids: Iterable[str],
    provenance: Mapping[str, object],
) -> VerificationResult:
    canonical = {str(x) for x in canonical_numbers}
    evidence_ids = set(available_evidence_ids)
    numerical_issues = []
    for number in _NUMBER.findall(draft.answer):
        normalized = number.rstrip("%")
        if number not in canonical and normalized not in canonical:
            numerical_issues.append(f"Unverified numeric token in answer: {number}")

    citation_issues = [
        f"Unknown evidence id: {eid}"
        for eid in draft.evidence_ids
        if eid not in evidence_ids
    ]
    provenance_issues = []
    ingestion = provenance.get("ingestion", {}) if isinstance(provenance, Mapping) else {}
    if (
        isinstance(ingestion, Mapping)
        and ingestion.get("source_type") in {"cached", "demo"}
    ):
        source_type = ingestion.get("source_type")
        if str(source_type) not in draft.answer.lower():
            provenance_issues.append(
                f"Answer does not disclose {source_type} market data provenance"
            )

    passed = not (numerical_issues or citation_issues or provenance_issues)
    corrections = numerical_issues + citation_issues + provenance_issues
    return VerificationResult(
        passed=passed,
        numerical_issues=numerical_issues,
        citation_issues=citation_issues,
        provenance_issues=provenance_issues,
        required_corrections=corrections,
    )
