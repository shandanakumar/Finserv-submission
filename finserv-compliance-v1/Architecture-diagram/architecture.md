# FinServ Global — Regulatory Compliance Assistant
## Architecture Document

**Version:** 1.0  
**Date:** June 2026  
**Author:** AI Architect  
**Assignment:** Intellect Design Arena — AI Architect Technical Assessment

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [RAG Pipeline Design](#3-rag-pipeline-design)
4. [LLM Orchestration Layer](#4-llm-orchestration-layer)
5. [Agentic Workflow Design](#5-agentic-workflow-design)
6. [Infrastructure and Non-Functional Requirements](#6-infrastructure-and-non-functional-requirements)
7. [Security and Data Residency](#7-security-and-data-residency)
8. [Observability and Evaluation](#8-observability-and-evaluation)
9. [Trade-off Analysis](#9-trade-off-analysis)
10. [Architecture Decision Records](#10-architecture-decision-records)

---

## 1. Executive Summary

FinServ Global operates across India, EU, and US markets. Its compliance team of ~40 officers currently spends 60%+ of their time manually searching regulatory PDFs. This system replaces that manual process with an AI-powered assistant that:

- Answers natural-language regulatory queries with cited, versioned answers
- Screens financial transactions against applicable regulations in real-time
- Tracks regulatory changes across document versions (Basel III 2010 vs 2017, RBI KYC 2016 vs 2025)
- Generates structured compliance reports for audit committees
- Flags potential violations with risk ratings and required actions

**Prototype scope:** 9 regulatory documents, 2,932 chunks, 5 regulation families, working FastAPI + Streamlit UI, evaluated on 20 Q&A pairs.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  User Interface Layer                                           │
│  Streamlit :8501 · Swagger :8000/docs · REST clients           │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTPS
┌─────────────────────▼───────────────────────────────────────────┐
│  API Layer — FastAPI :8000                                      │
│  POST /query · POST /screen-transaction · GET /health           │
│  API key auth · CORS middleware                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │ agent.run(query)
┌─────────────────────▼───────────────────────────────────────────┐
│  LangGraph Compliance Agent                                     │
│  ① Router → ② RAG Retrieval → ③ Reranker →                    │
│  ④ Synthesis → ⑤ Reflection → ⑥ Guardrails                   │
└──────┬──────────────────────────────────┬───────────────────────┘
       │ query_points()                   │ invoke_model()
┌──────▼──────────┐              ┌────────▼────────────────────────┐
│  Qdrant         │              │  AWS Bedrock (us-east-1)        │
│  Vector Store   │              │  Mistral 7B · Mixtral 8x7B      │
│  2,932 vectors  │              │  Titan Embed v2 (1024-dim)      │
│  disk persist.  │              │                                 │
└─────────────────┘              └─────────────────────────────────┘

Offline ingestion (run once):
PDFs → pdfminer + camelot → SemanticChunker → Titan → Qdrant
```

---

## 3. RAG Pipeline Design

### 3.1 Document Ingestion Strategy

The system ingests regulatory documents from multiple formats and jurisdictions. Every document is registered in the version registry before ingestion, which attaches lifecycle metadata to every chunk stored in the vector database.

**Supported formats:** PDF (primary), TXT (clean reference documents for table-heavy PDFs), DOCX, HTML, Markdown.

**Version registry design:** Each document maps to a `RegulationVersion` entry containing:

```python
RegulationVersion(
    regulation_id   = "RBI_KYC",          # family identifier
    version         = "v2_2025",           # specific version
    effective_from  = "2025-11-28",        # when it came into effect
    effective_to    = None,                # None = currently active
    status          = "active",            # active | superseded | withdrawn
    filename        = "RBI_KYC_CommercialBanks_2025_v2.pdf",
)
```

This enables three retrieval modes: current-only (filter `is_superseded=False`), point-in-time (filter by `version`), and comparison (retrieve two versions of the same regulation for diff analysis).

**Documents ingested (9 total):**

| Document | Regulation | Version | Status | Chunks |
|---|---|---|---|---|
| BIS_Basel3_2010_v1.pdf | BASEL_III | v1_2010 | superseded | 336 |
| BIS_Basel3_d424_v2.pdf | BASEL_III | v2_2017 | active | 788 |
| FATF Recommendations 2012.pdf | FATF_AML | v1_2012 | active | 423 |
| MiFID CELEX_32014L0065_EN_TXT.pdf | MIFID_II | v1_2014 | active | 615 |
| RBI_KYC_2016_v1.pdf | RBI_KYC | v1_2016 | superseded | 278 |
| RBI_KYC_2024_003_v5.txt | RBI_KYC | v1_5_2024 | active | 15 |
| RBI_KYC_CommercialBanks_2025_v2.pdf | RBI_KYC | v2_2025 | active | 230 |
| RBI_PSL_2020_v1.pdf | RBI_PSL | v1_2020 | superseded | 122 |
| RBI_PSL_2025_v2.pdf | RBI_PSL | v2_2025 | active | 125 |
| **Total** | | | | **2,932** |

### 3.2 Chunking Strategy

**Strategy: Hierarchical Semantic Chunking (3 stages)**

Stage 1 — structural split by headings using 6 heading regex patterns:

```python
HEADING_PATTERNS = [
    re.compile(r'^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{3,80})', re.MULTILINE),   # Basel III: 4.2 Capital
    re.compile(r'^(Article\s+\d+[A-Z]?)[\s:\—\-]+([^\n]{0,80})', ...),    # MiFID II: Article 25
    re.compile(r'^(Section\s+\d+(?:\.\d+)*)[\s:\—\-]*([^\n]{0,80})', ...), # Section 4.2
    re.compile(r'^(Para(?:graph)?\s+\d+|^\d+\.)[\s:\—\-]+([^\n]{0,80})', ...), # Para 5
    re.compile(r'^(Part\s+[IVXLC]+|Chapter\s+[IVXLC]+)[\s:\—\-]*', ...),  # Part III
    re.compile(r'^([A-Z]\.|[IVX]+\.)[\s]+([A-Z][^\n]{3,80})', ...),        # RBI: A. Applicability
]
```

Each heading pattern fires on a different regulatory document style. The regex finds boundary positions in the text string, then slices the document between adjacent boundaries. Each slice becomes one logical section — one complete regulatory clause.

Stage 2 — size enforcement: sections larger than 400 tokens are split at sentence boundaries to prevent oversized chunks that degrade retrieval precision.

Stage 3 — overlap injection: 50 tokens (10%) of the previous chunk's tail are prepended to each chunk to prevent boundary loss when a regulatory clause spans a chunk edge.

**Fallback:** When no heading patterns match (e.g. FATF PDF which uses numbered paragraphs without section titles), the chunker splits at double-newlines (`\n\n`). This respects paragraph boundaries rather than cutting mid-sentence.

**Table extraction:** RBI PSL and Basel III contain structured data tables where pdfminer loses column associations during extraction. The pipeline uses camelot-py (lattice mode) as primary table extractor with pdfplumber fallback. Table rows are converted to natural language sentences before chunking:

```
Input (pdfminer raw):  "18 per cent of ANBC or CEOBSE whichever is higher"
Output (camelot):      "The Agriculture Priority Sector Lending target is 18 per cent of ANBC or CEOBSE, whichever is higher."
```

**Chunk parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Max tokens | 400 | ~1600 chars = 1-2 regulatory clauses. Balances context vs precision. |
| Min tokens | 50 | Prevents stub chunks with no retrievable content. |
| Overlap | 50 tokens (10%) | Prevents boundary loss. Same-section only — no cross-section overlap. |

### 3.3 Embedding Model Selection

**Selected: Amazon Titan Text Embeddings v2** (`amazon.titan-embed-text-v2:0`)

| Model | Dimensions | Cost | Data residency | Decision |
|---|---|---|---|---|
| Titan Embed v2 | 1024 | $0.00002/1K tokens | AWS us-east-1 | ✅ Selected |
| OpenAI ada-002 | 1536 | $0.0001/1K tokens | US servers | ❌ Data residency |
| nomic-embed-text | 768 | Free (self-hosted) | 100% local | ✅ V2 alternative |
| all-MiniLM-L6-v2 | 384 | Free | Local | ❌ Too small for regulatory text |

Titan was selected for V1 (Bedrock) because it runs natively within the AWS infrastructure boundary, requires no external API calls, and provides 1024-dimensional vectors with strong semantic similarity for regulatory domain text.

### 3.4 Vector Database

**Selected: Qdrant (local disk mode)**

Full trade-off analysis in ADR-001. Key reasons:
- Open source, zero data egress
- Rich payload filtering for version-aware retrieval
- Disk persistence — no re-ingestion on server restart
- `is_superseded` and `regulation_id` filters directly in vector search

### 3.5 Retrieval Strategy

**Hybrid Search: Dense + BM25 + RRF**

```
Query
  ├── Dense retrieval: Titan embeds query → cosine search in Qdrant
  │   Filter: is_superseded=False AND regulation_id=<selected>
  │
  ├── BM25 retrieval: keyword scoring over active chunk pool
  │   Same regulation filter applied to BM25 pool
  │
  └── RRF fusion: score = Σ 1/(k + rank_i) where k=60
        → top-10 fused candidates
        → keyword reranker → top-5 to LLM
```

**Why hybrid:** Dense search finds semantically similar chunks even when the query uses different vocabulary than the document (e.g. "re-verification" matches "updation"). BM25 keyword search finds exact regulatory terms and numbers (e.g. "18 per cent", "4.5%", "Recommendation 20"). RRF fusion with k=60 rewards chunks that rank highly in both searches.

**Version-aware filtering:** The `is_superseded=False` filter is applied to both the Qdrant dense search (`query_filter` parameter) and the BM25 index construction (`scroll_filter` in `_build_bm25`). This ensures superseded regulation chunks never reach the LLM regardless of their vector similarity score.

---

## 4. LLM Orchestration Layer

### 4.1 Model Selection

**Primary: Mistral 7B Instruct v0.2** (`mistral.mistral-7b-instruct-v0:2`)
- Intent classification, regulatory Q&A, reflection
- Latency: ~1.5s, cost: $0.00015/1K tokens
- Apache 2.0 licence — deployable on-premise

**Complex: Mixtral 8×7B Instruct v0.1** (`mistral.mixtral-8x7b-instruct-v0:1`)
- Report generation, multi-regulation impact analysis
- Mixture of Experts: 45B parameter quality at 7B inference cost
- Activated when context > 3000 chars or intent = report/impact_analysis

**Why Mistral over LLaMA:** Mistral 7B outperforms LLaMA 2 13B on reasoning benchmarks while requiring only 4GB VRAM (vs 26GB). Grouped Query Attention (GQA) reduces memory bandwidth requirements enabling faster inference on smaller GPUs.

**Why not GPT-4/Claude:** Proprietary models send regulatory document content to US-hosted external APIs. RBI data localisation guidelines and MiFID II data protection requirements prohibit this without explicit data processing agreements.

### 4.2 Multi-Model Routing

```python
def get_llm_client(use_complex: bool = False):
    if use_complex:
        return BedrockLLMClient(model_id="mistral.mixtral-8x7b-instruct-v0:1")
    return BedrockLLMClient(model_id="mistral.mistral-7b-instruct-v0:2")
```

Routing logic in `synthesis_node`:
- `intent in ("report", "impact_analysis")` → Mixtral 8×7B
- `len(context) > 3000` → Mixtral 8×7B (large context benefits from stronger model)
- All other queries → Mistral 7B

### 4.3 Prompt Engineering Framework

**System prompt** instructs the LLM on output format and grounding rules:

```
You are a regulatory compliance expert. You MUST answer questions 
using ONLY the provided context.

CRITICAL RULES:
1. The context below contains the answer — READ IT CAREFULLY
2. If the context mentions ANY timeframe, percentage, or requirement — state it explicitly
3. NEVER say "context does not provide" if numbers appear in the context
4. Always cite the paragraph number you found the answer in

OUTPUT FORMAT:
ANSWER: <specific answer with exact numbers>
SOURCES: <paragraph and document>
CONFIDENCE: HIGH | MEDIUM | LOW
CROSS_JURISDICTION_FLAG: YES | NO
```

**User template** injects the retrieved context as a focused 3000-char window (not the full 7500-char context) to improve Mistral 7B instruction following on specific factual queries.

### 4.4 Guardrails

Four guardrail layers run in sequence after synthesis:

1. **PII redaction** — regex patterns for account numbers, PAN, IBAN, SWIFT BIC, credit card numbers. Detected PII is redacted before the response reaches the user.
2. **Citation validation** — every citation in the answer (e.g. `[Basel III, Section 4.2]`) is verified against the retrieved chunks. Hallucinated citations are flagged.
3. **Confidence routing** — LOW confidence answers are flagged for human review. The `requires_human_review` flag is returned to the UI.
4. **Reflection loop** — before guardrails, the reflection node asks the LLM to self-evaluate answer quality. If quality is poor, retrieval retries with expanded search terms (max 2 cycles).

---

## 5. Agentic Workflow Design

### 5.1 Framework Selection

**Selected: LangGraph** — full rationale in ADR-002.

Key reasons over CrewAI: deterministic execution path (same input = same graph traversal), typed state via TypedDict, cyclic graph support for the reflection retry loop, append-only audit log via `Annotated[list, operator.add]`.

### 5.2 Agent State

```python
class ComplianceAgentState(TypedDict):
    input:                  str
    intent:                 str           # qa | screen_transaction | report
    transaction:            Optional[dict]
    search_query:           str
    retrieved_chunks:       list
    jurisdictions_triggered:list
    regulations_triggered:  list
    context:                str
    draft_response:         str
    citations:              list
    confidence:             str           # HIGH | MEDIUM | LOW
    reflection_count:       int
    requires_human_review:  bool
    broader_search_needed:  bool
    broader_search_terms:   list
    error:                  Optional[str]
    final_response:         dict
    audit_log:              Annotated[list, operator.add]  # append-only
```

The `audit_log` field is the key compliance feature. Every node appends its execution record with timestamp, action, and details. This produces a complete, tamper-evident audit trail for every query — satisfying the assignment's "traceability is non-negotiable" requirement.

### 5.3 Graph Topology

```
router_node
    │
    ▼
rag_retrieval_node ◄─── (retry from reflection)
    │
    ▼
synthesis_node
    │
    ▼
reflection_node ──── quality=poor? → rag_retrieval_node (max 2×)
    │                quality=good? ↓
    ▼
guardrail_validator_node
    │
    ▼
formatter_node → final_response
```

### 5.4 Tool Definitions

Five tools are defined in `src/agent/tools.py`:

```python
TOOL_SCHEMAS = [
    {
        "name": "regulatory_search",
        "description": "Search the regulatory knowledge base for relevant rules and requirements",
        "parameters": {
            "query": str,
            "regulation_families": Optional[list[str]],
            "jurisdictions": Optional[list[str]],
            "active_only": bool  # default True
        }
    },
    {
        "name": "transaction_lookup",
        "description": "Fetch transaction details by ID for compliance screening",
        "parameters": {"transaction_id": str}
    },
    {
        "name": "regulation_diff",
        "description": "Compare two versions of the same regulation to identify changes",
        "parameters": {
            "regulation_id": str,
            "version_a": str,
            "version_b": str,
            "topic": Optional[str]
        }
    },
    {
        "name": "generate_compliance_report",
        "description": "Generate a structured compliance assessment report",
        "parameters": {
            "transactions": list,
            "period": str,
            "regulations": list[str]
        }
    },
    {
        "name": "flag_for_human_review",
        "description": "Escalate a compliance decision to a human compliance officer",
        "parameters": {
            "reason": str,
            "confidence": str,
            "regulation": str
        }
    }
]
```

### 5.5 Error Handling Strategy

Every node is wrapped in try/except with three tiers of handling:

1. **Recoverable errors** (e.g. retrieval returns empty results): agent continues with degraded context, marks `confidence=LOW`.
2. **LLM timeout errors**: synthesis node catches `ReadTimeoutError`, returns the partial response with `confidence=LOW` and `requires_human_review=True`.
3. **Fatal errors**: node sets `state["error"]` and the formatter includes it in the final response. The API returns HTTP 200 with the error in the body (not HTTP 500) so the audit trail is preserved.

**FallbackComplianceAgent:** If LangGraph fails to initialise (import error, version mismatch), a plain Python sequential implementation runs the same 6 nodes without the graph framework.

---

## 6. Infrastructure and Non-Functional Requirements

### 6.1 Cloud Architecture (AWS)

```
Internet
    │
    ▼
Route 53 → CloudFront
    │
    ▼
ALB (Application Load Balancer)
    │
    ├── ECS Fargate — FastAPI containers (auto-scaling, 2-10 tasks)
    ├── ECS Fargate — Streamlit containers (2-4 tasks)
    │
    ├── Qdrant Server (EC2 c5.2xlarge, multi-AZ)
    │
    └── AWS Bedrock (managed, same region)
```

**Container specs:**
- FastAPI: 1 vCPU, 2GB RAM per task
- Streamlit: 0.5 vCPU, 1GB RAM per task
- Auto-scaling: target CPU utilisation 60%, scale-out when queue depth > 10 requests

### 6.2 Cost Estimation (500 concurrent users, 10K queries/day)

| Component | Unit cost | Daily usage | Daily cost |
|---|---|---|---|
| Bedrock Mistral 7B | $0.00015/1K tokens | 10K queries × 2K tokens avg | $3.00 |
| Bedrock Mixtral 8×7B | $0.00045/1K tokens | 1K complex queries × 5K tokens | $2.25 |
| Titan Embeddings | $0.00002/1K tokens | 10K queries × 0.1K tokens | $0.02 |
| ECS Fargate (FastAPI) | $0.04048/vCPU-hr | 4 tasks × 24hr | $3.88 |
| ECS Fargate (Streamlit) | $0.04048/vCPU-hr | 2 tasks × 24hr | $0.97 |
| ALB | $0.008/LCU-hr | ~10 LCUs | $1.92 |
| EC2 Qdrant (c5.2xlarge) | $0.34/hr | 24hr | $8.16 |
| **Total** | | | **~$20/day (~$600/month)** |

**Ingestion cost (one-time):** 2,932 chunks × 0.5K tokens × $0.00002 = $0.03. Negligible.

### 6.3 Auto-scaling Strategy

```yaml
# ECS Service Auto Scaling
ScalingPolicy:
  TargetTrackingConfiguration:
    TargetValue: 60.0          # 60% CPU utilisation
    ScaleOutCooldown: 60       # seconds
    ScaleInCooldown: 300       # seconds
MinCapacity: 2
MaxCapacity: 10
```

For model inference at scale, AWS Bedrock handles elasticity automatically — no GPU provisioning required. For V2 (self-hosted vLLM), a separate auto-scaling group for GPU instances is required with minimum 2 × g4dn.2xlarge for high availability.

---

## 7. Security and Data Residency

### 7.1 Data Residency

| Regulation | Jurisdiction | Requirement | Implementation |
|---|---|---|---|
| RBI guidelines | India | Data must stay in India | Deploy to AWS ap-south-1 (Mumbai) for production RBI queries |
| MiFID II | EU | EU data protection | Deploy to AWS eu-west-1 for MiFID II queries |
| Basel III | Global | No specific residency | AWS us-east-1 (prototype) |

For the prototype, all data stays within AWS us-east-1. No regulatory document content is sent to external APIs (OpenAI, Anthropic, Cohere) — Bedrock keeps all inference within the AWS network boundary.

### 7.2 Encryption

- **At rest:** Qdrant storage encrypted with EBS encryption (AES-256). S3 document store with SSE-S3.
- **In transit:** All API calls over TLS 1.3. Bedrock API calls use AWS Signature V4 signed requests over HTTPS.

### 7.3 Access Controls

- API key authentication on all endpoints
- IAM roles for ECS tasks with minimum required Bedrock permissions
- No IAM keys in code — credentials via EC2 instance profile / ECS task role
- All Bedrock model invocations logged to CloudTrail

### 7.4 Audit Logging

Every query produces an `audit_log` array in the response — timestamp, node, action, and details for each of the 6 agent steps. This satisfies the assignment's "documented evidence of how compliance decisions were reached" requirement. Logs are persisted to CloudWatch Logs and retained for 7 years (regulatory requirement).

---

## 8. Observability and Evaluation

### 8.1 Monitoring Stack

```
Application metrics → CloudWatch
LLM metrics:
  - Latency per model (p50, p95, p99)
  - Token usage (input/output) per query
  - Cost per query
  - Confidence distribution (HIGH/MEDIUM/LOW ratio)
  - Human review flag rate

Retrieval metrics:
  - Average retrieval latency
  - Chunks retrieved per query
  - Regulation filter usage distribution
  - BM25 vs dense score distribution
```

### 8.2 Evaluation Pipeline

**RAGAS-style automated evaluation** runs on a 20-question ground truth dataset:

| Metric | Score (prototype) | Target | Definition |
|---|---|---|---|
| Faithfulness | 0.632 | >= 0.80 | Answer claims supported by retrieved context |
| Answer Relevance | 0.516 | >= 0.75 | Key terms from ground truth present in answer |
| Context Precision | 0.715 | >= 0.75 | Retrieved chunks relevant to query |
| Context Recall | 0.715 | >= 0.70 | Ground truth content present in retrieved context |
| Avg Latency | 3,712ms | <= 10,000ms | End-to-end query response time |
| HIGH confidence | 85% (17/20) | — | System self-assessed answer quality |

Context Recall exceeds the 0.70 target. Faithfulness and Answer Relevance are below target due to the custom keyword-overlap metric penalising correct answers that use different phrasing than the ground truth (e.g. "every two years" vs "2 years"). A production implementation would use LLM-as-judge evaluation via the RAGAS library for more precise faithfulness scoring.

### 8.3 Drift Detection

Monthly automated re-evaluation run detects retrieval relevance drift:
- Re-run the 20-question eval dataset
- Alert if Context Recall drops > 0.05 from baseline
- Common causes: new document ingestion changes BM25 index balance, model update changes embedding space
- Response: re-tune BM25 weights or re-embed all documents with new model

---

## 9. Trade-off Analysis

### 9.1 Single vector space vs separate collections per regulation

**Current design:** One Qdrant collection `regulatory_docs` with all 2,932 vectors. Regulation filtering via `regulation_id` payload filter.

**Trade-off:** Single collection is simpler to maintain and enables cross-regulation queries. However it causes vocabulary contamination — Basel III chunks share words like "risk", "years", "requirements" with RBI KYC chunks, causing cross-regulation false positives in BM25 search.

**Mitigation implemented:** Explicit `regulation_id` filter in both dense search and BM25 index when the user selects a regulation in the UI. Auto-detection via query keywords as fallback.

**Production recommendation:** Separate collections per regulation for clean isolation, with a meta-collection for cross-regulation queries.

### 9.2 Hierarchical semantic chunking vs recursive character splitting

**Current design:** Heading regex detection → size enforcement → overlap injection.

**Trade-off:** Heading-based chunking produces semantically coherent chunks that map to actual regulatory sections, enabling accurate citations. However it requires 6 document-specific regex patterns and fails when PDFs use non-standard heading formats.

**Alternative:** LangChain `RecursiveCharacterTextSplitter` splits by `\n\n` → `\n` → ` ` in cascade. Simpler but produces chunks that cut mid-clause, degrading citation accuracy.

**Decision:** Hierarchical chunking with paragraph fallback for unrecognised formats. Production improvement: add pattern auto-detection from the first 500 characters of each document.

### 9.3 Table extraction approach

**Current design:** camelot-py lattice mode (primary) + pdfplumber (fallback) + clean TXT reference documents for the most table-heavy PDFs.

**Trade-off:** camelot-py correctly preserves column associations (Agriculture = 18% of ANBC) but requires Ghostscript as a system dependency. The clean TXT workaround is not scalable — it requires manual curation for each new document.

**Production recommendation:** Multimodal approach — screenshot each table page, send to Pixtral Large on Bedrock (vision-language model), extract structured JSON. Zero dependencies, handles any table format, scales automatically.

---

## 10. Architecture Decision Records

See individual ADR files in `docs/adr/`:

- `ADR-001-vector-database.md` — Qdrant vs Pinecone vs pgvector
- `ADR-002-orchestration-framework.md` — LangGraph vs CrewAI vs LangChain
- `ADR-003-llm-hosting-strategy.md` — AWS Bedrock vs self-hosted Ollama/vLLM
