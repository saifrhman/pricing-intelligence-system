"""Schemas for RAG, agent orchestration, verification, history and tracing."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

class EvidenceChunk(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    chunk_id: str
    document_id: str
    ticker: Optional[str] = None
    document_type: str = "unknown"
    source: str
    source_url: Optional[str] = None
    date: Optional[str] = None
    section: Optional[str] = None
    text: str
    relevance_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalQuery(BaseModel):
    query: str = Field(min_length=2)
    ticker: Optional[str] = None
    top_k: int = Field(default=6, ge=1, le=25)
    document_types: List[str] = Field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ResearchOutput(BaseModel):
    query: str
    evidence: List[EvidenceChunk] = Field(default_factory=list)
    supporting_findings: List[str] = Field(default_factory=list)
    conflicting_findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

class ToolTraceEvent(BaseModel):
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str
    event_type: Literal["agent", "tool", "retrieval", "verification", "error", "runtime"]
    name: str
    status: Literal["started", "completed", "failed", "skipped"]
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OrchestratorPlan(BaseModel):
    ticker: str
    question: str
    specialists: List[Literal["quant", "risk", "research", "sentiment", "explanation"]]
    requires_verification: bool = True
    rationale: str = ""

class VerificationResult(BaseModel):
    passed: bool
    unsupported_claims: List[str] = Field(default_factory=list)
    numerical_issues: List[str] = Field(default_factory=list)
    citation_issues: List[str] = Field(default_factory=list)
    provenance_issues: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    required_corrections: List[str] = Field(default_factory=list)

class IntelligenceReportDraft(BaseModel):
    ticker: str
    answer: str
    quantitative_summary: Optional[str] = None
    risk_summary: Optional[str] = None
    research_summary: Optional[str] = None
    sentiment_summary: Optional[str] = None
    supporting_factors: List[str] = Field(default_factory=list)
    conflicting_factors: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)

class IntelligenceReport(IntelligenceReportDraft):
    citations: List[EvidenceChunk] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    verification: VerificationResult
    selected_agents: List[str] = Field(default_factory=list)
    run_id: str

class HistoricalRunSummary(BaseModel):
    run_id: str
    timestamp_utc: str
    ticker: str
    forecast: Dict[str, Any]
    risk: Dict[str, Any]
    anomaly: Dict[str, Any]
    sentiment: Dict[str, Any]
    explanation: Dict[str, Any]
    decision: Dict[str, Any]
    provenance: Dict[str, Any]
