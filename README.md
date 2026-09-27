# Agentic Pricing Intelligence Platform

A production-style Python project for **risk-aware pricing intelligence and grounded financial decision support**.

The system combines deterministic time-series ML, risk modelling, anomaly detection, sentiment and SHAP with a separate LLM/RAG layer. The architectural rule is simple:

> **LLMs reason. Python calculates. RAG supplies evidence. Verification checks grounding.**

The LLM is never the source of truth for forecasts, volatility, drawdown, anomaly scores, sentiment-model outputs or SHAP values.

This project is educational/research decision support only. It does **not** execute trades.

## Core capabilities

### Deterministic analytics

- OHLCV ingestion with \`yfinance\`
- explicit fresh / cached / demo provenance
- return-focused feature engineering
- naive previous-return baseline
- linear-regression baseline
- XGBoost next-day return model
- time-aware validation/test splits
- RMSE, MAE, R² and directional accuracy
- volatility/drawdown risk scoring
- Isolation Forest anomaly detection
- FinBERT sentiment with explicit fallback policy
- SHAP model explainability

### Agentic intelligence

- OpenAI Agents SDK manager/orchestrator
- LLM-powered Quant, Risk, Research, Sentiment and Explanation specialists
- specialists exposed to the manager as tools
- dynamic routing rather than running every agent for every question
- deterministic analytical tools with typed outputs and tool permissions
- independent grounding verifier
- capped revision cycles
- operational tracing without private chain-of-thought
- session memory separated from authoritative analytical history
- graceful deterministic fallback when no LLM key is available

### Retrieval-Augmented Generation

- local evidence ingestion for TXT, Markdown, JSON and PDF
- chunk-level metadata
- ticker/document/date filtering
- deterministic local hashing embeddings by default
- optional OpenAI embeddings
- persistent local JSON vector store
- optional Qdrant vector store
- evidence IDs and source metadata propagated to agent responses
- retrieved text treated as untrusted evidence, not instructions

### MCP

- application-owned **read-only MCP server**
- latest persisted analysis
- historical analysis lookup
- RAG evidence search
- explicit capability/trust-boundary metadata
- no shell, arbitrary filesystem, credential or trading tools
- optional bridge for explicitly trusted external Streamable HTTP MCP servers

## Architecture

\`\`\`mermaid
flowchart TD
    U[User / Streamlit / CLI] --> O[Pricing Intelligence Orchestrator]

    O -->|as needed| Q[Quant Agent]
    O -->|as needed| R[Risk Agent]
    O -->|as needed| RE[Research Agent]
    O -->|as needed| S[Sentiment Agent]
    O -->|as needed| E[Explanation Agent]

    Q --> FT[Forecast + model tools]
    Q --> ST[SHAP tool]
    R --> RT[Risk + anomaly tools]
    S --> SET[Sentiment tool]
    E --> ST
    RE --> RAG[RAG retriever]

    FT --> D[Deterministic analytics]
    RT --> D
    SET --> D
    ST --> D

    RAG --> VS[Local JSON / Qdrant]
    VS --> DOCS[Filings / reports / transcripts / local evidence]

    O --> V[Grounding verifier]
    V --> OUT[Evidence-grounded response]

    D --> H[Structured analytical history]
    H --> O

    MCP[MCP server] --> H
    MCP --> RAG
\`\`\`

## Why the LLM sits above the analytics

A weak architecture would ask an LLM to calculate volatility, invent a confidence score or replace XGBoost.

This project instead uses:

\`\`\`text
Python/ML -> numerical facts
RAG       -> external/unstructured evidence
LLM       -> planning, tool selection and synthesis
Verifier  -> grounding and consistency checks
\`\`\`

That keeps quantitative results reproducible while still allowing flexible natural-language reasoning.

## Agent responsibilities

### Pricing Intelligence Orchestrator

The manager owns the final answer and chooses specialists dynamically.

Examples:

| User question | Typical routing |
|---|---|
| What is the predicted return? | Quant |
| Why is risk high? | Risk, optionally Research |
| What do recent filings say about margins? | Research/RAG |
| What features are driving the forecast? | Quant + Explanation |
| Give me a complete intelligence assessment. | Quant + Risk + Research + optional Sentiment + Explanation + Verifier |

### Quant Agent

Uses deterministic forecast, model-performance, SHAP and historical-run tools. It cannot invent numerical market/model values.

### Risk Agent

Uses deterministic risk, volatility, drawdown and anomaly tools.

### Research Agent

Uses RAG before making external factual claims. It receives source/chunk metadata, identifies conflicting evidence and is instructed to ignore instruction-like text inside retrieved documents.

### Sentiment Agent

Reasons over the existing deterministic sentiment module while keeping sentiment separate from factual evidence.

### Explanation Agent

Explains SHAP contribution data without presenting model attribution as market causality.

### Grounding Verifier

Complex LLM outputs are audited for:

- unsupported numerical claims
- evidence IDs that were never retrieved
- external claims without evidence
- stale/mismatched evidence
- missing cached/demo disclosure
- SHAP presented as causal proof
- sentiment presented as objective fact
- predictions presented as certainty
- unacknowledged conflicting signals

A deterministic pre-check runs in addition to the LLM verifier.

## Agent harness

\`src/harness/\` implements the runtime around agents:

- central model/runtime configuration
- typed tool registry
- least-privilege agent/tool permissions
- per-tool timeouts
- bounded agent turns
- bounded revision cycles
- structured Pydantic outputs
- safe JSONL tracing
- session storage
- structured historical-run storage
- provenance propagation

The trace records operational events such as tool name, calling specialist, duration and success/failure. It does not log hidden reasoning.

## RAG

RAG is used for **unstructured evidence**, not raw OHLCV rows.

Supported local formats:

- \`.txt\`
- \`.md\`
- \`.json\`
- \`.pdf\`

Each chunk can retain:

\`\`\`text
ticker
document_id
document_type
source
source_url
date
section
chunk_id
metadata
\`\`\`

### Local-first defaults

The default development path requires no embedding API:

- deterministic feature-hashing embedder
- JSON vector store at \`data/vector_store/evidence.json\`

Set \`QDRANT_URL\` to use Qdrant. The vector-store interface remains the same.

### Ingest evidence

\`\`\`bash
python rag_ingest.py knowledge/aapl \
  --ticker AAPL \
  --document-type filing \
  --source sec
\`\`\`

The repository implements local/file-based ingestion. Live SEC/news/transcript APIs can be added later as source adapters without changing the retrieval interface.

## MCP server

The repository exposes safe read-only intelligence through the official MCP Python SDK.

Run:

\`\`\`bash
python mcp_server.py
\`\`\`

Tools exposed:

- \`get_latest_analysis\`
- \`get_analysis_history\`
- \`search_company_evidence\`
- \`get_capabilities\`

The server intentionally does not expose trade execution, shell commands, arbitrary file access or credentials.

\`src/agentic/mcp_bridge.py\` also supports explicitly configured external Streamable HTTP MCP servers. External servers are opt-in and created with approval required for tool calls.

## Structured memory

The system separates two different concepts.

**Session memory** stores recent conversational context so follow-up questions are coherent. It is never authoritative for financial numbers.

**Analytical history** stores structured outputs from successful deterministic runs: forecast, risk, anomaly, sentiment, SHAP, decision and provenance. Questions such as "what changed since last time?" use this store.

Successful pipeline runs persist snapshots under \`outputs/history/\`.

## Repository structure

\`\`\`text
pricing-intelligence-system/
├── app/
│   └── streamlit_app.py
├── config/
│   └── config.yaml
├── src/
│   ├── agentic/
│   │   ├── engine.py
│   │   └── mcp_bridge.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── datasets/routing_cases.json
│   ├── harness/
│   │   ├── config.py
│   │   ├── guardrails.py
│   │   ├── registry.py
│   │   ├── runtime.py
│   │   ├── state.py
│   │   └── tracing.py
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── ingestion.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── tools/
│   │   └── intelligence_tools.py
│   ├── agentic_schemas.py
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   ├── sentiment.py
│   ├── explainability.py
│   ├── agents.py
│   ├── pipeline.py
│   └── schemas.py
├── tests/
├── agent_cli.py
├── rag_ingest.py
├── evaluate_agents.py
├── mcp_server.py
├── main.py
├── .env.example
└── requirements.txt
\`\`\`

\`src/agents.py\` remains the deterministic compatibility layer used by the original pipeline. Genuine LLM-powered agents live under \`src/agentic/\`.

## Installation

The agent/MCP stack requires Python 3.10+.

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
\`\`\`

Do not commit \`.env\`.

## Environment configuration

Important variables:

\`\`\`text
OPENAI_API_KEY=
PRICING_LLM_MODEL=gpt-5.6-terra

PRICING_AGENT_TRACING=true
PRICING_AGENT_MAX_STEPS=12
PRICING_AGENT_MAX_REVISIONS=1
PRICING_TOOL_TIMEOUT_SECONDS=20
PRICING_RAG_TOP_K=6

PRICING_EMBEDDING_MODEL=
PRICING_VECTOR_STORE=data/vector_store/evidence.json

QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=pricing_evidence

PRICING_EXTERNAL_MCP_URLS=
\`\`\`

The deterministic pipeline, local RAG, history store, MCP server and offline tests do not require an OpenAI API key.

## Usage

### 1. Run deterministic analytics

\`\`\`bash
python main.py \
  --ticker AAPL \
  --start-date 2020-01-01 \
  --end-date 2025-01-01
\`\`\`

Useful reliability flags:

\`\`\`bash
python main.py --allow-cache
python main.py --demo-mode
python main.py --disable-sentiment
python main.py --no-transformer
python main.py --sentiment-headlines-file headlines.txt
\`\`\`

Every successful run stores a compact historical snapshot for later agent/MCP comparison.

### 2. Ask the agentic layer

Run the deterministic pipeline first, then:

\`\`\`bash
python agent_cli.py \
  --ticker AAPL \
  --question "Why is risk elevated despite the forecast?"
\`\`\`

If \`OPENAI_API_KEY\` is configured, the manager/specialist/verifier LLM path is used.

Without an LLM key, the CLI degrades to transparent deterministic routing and explicitly states that qualitative LLM synthesis is unavailable.

Force the LLM path:

\`\`\`bash
python agent_cli.py \
  --ticker AAPL \
  --question "Give me a complete intelligence assessment" \
  --require-llm
\`\`\`

### 3. Streamlit dashboard + copilot

\`\`\`bash
python -m streamlit run app/streamlit_app.py
\`\`\`

The dashboard keeps the latest deterministic run in Streamlit session state so copilot questions stay grounded in the exact analytics shown on screen.

### 4. Index RAG evidence

\`\`\`bash
python rag_ingest.py knowledge/aapl \
  --ticker AAPL \
  --document-type filing \
  --source sec
\`\`\`

### 5. Run MCP

\`\`\`bash
python mcp_server.py
\`\`\`

### 6. Run offline evaluation

\`\`\`bash
python evaluate_agents.py
\`\`\`

Machine-readable output is written under \`outputs/evaluation/\`.

## Evaluation

The repository includes reusable metrics for:

- routing exact match
- routing micro-precision
- routing micro-recall
- citation precision
- unsupported-numeric rate

The included dataset evaluates expected specialist routing. The same framework can be extended to compare:

\`\`\`text
LLM only
vs
LLM + RAG
vs
single agent + tools
vs
multi-agent + tools + RAG
vs
multi-agent + tools + RAG + verifier
\`\`\`

Useful next metrics include retrieval precision/recall@k, evidence faithfulness, verifier catch rate, latency, token usage and estimated cost.

## Testing

\`\`\`bash
pytest -q
\`\`\`

The new RAG/harness/history/routing/grounding tests are offline and do not require paid API calls.

## Reliability and security

- Fresh market data is preferred by default.
- Cached/demo fallback is explicit and propagated through provenance.
- Demo data cannot silently masquerade as live analysis.
- Retrieved documents are untrusted data and cannot override agent instructions.
- Tool permissions are explicit.
- Agent turns and revision cycles are bounded.
- Secrets are excluded from repository/tracing paths.
- MCP is read-only and least-privilege.
- External MCP servers require explicit configuration and approval.
- Deterministic analytics remain usable if the LLM or vector service is unavailable.
- No autonomous trade execution exists.

## Limitations

- Daily OHLCV only; no intraday microstructure modelling.
- The next-day return model is an experimental signal, not a trading oracle.
- Local RAG ingestion does not automatically download live filings/news/transcripts.
- The local hashing embedder prioritizes offline reproducibility over semantic quality.
- OpenAI/Qdrant features require their respective credentials/services.
- LLM outputs remain probabilistic even after verification.
- Single-asset workflow; portfolio optimization and execution are out of scope.

## Disclaimer

This project is for educational and research purposes only. It is **not investment advice, financial advice, a recommendation to trade securities, or an autonomous trading system**.
