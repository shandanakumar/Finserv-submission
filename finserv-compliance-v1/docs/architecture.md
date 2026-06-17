# AI-Powered Regulatory Compliance Assistant — Architecture Document

**Client:** FinServ Global  
**Prepared by:** AI Architect Candidate  
**Version:** 1.0  
**Date:** June 2025

---

## 1. Executive Summary

FinServ Global's compliance team of ~40 officers spends 60%+ of their time manually navigating regulatory PDFs, cross-referencing amendments, and preparing audit evidence. This document presents the architecture for an AI-powered Regulatory Compliance Assistant that:

- Answers natural-language regulatory queries with cited, versioned responses
- Screens transactions against applicable multi-jurisdictional regulations in near real-time
- Detects impact of new regulatory circulars on existing policies
- Generates structured, audit-ready compliance reports

The system is designed to run entirely within FinServ's data perimeter — no regulated text is sent to third-party LLM APIs. All components are open-source and self-hostable.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                    │
│   Compliance Officer UI  │  Compliance Head Dashboard  │  Audit API      │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │ HTTPS / REST
┌──────────────────▼──────────────────────────────────────────────────────┐
│                       API GATEWAY (Kong / Nginx)                         │
│            Auth (OAuth2/OIDC) │ Rate Limiting │ Audit Logging            │
└──────────────────┬──────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────┐
│                    COMPLIANCE SERVICE (FastAPI)                           │
│                                                                           │
│   /query  │  /screen-transaction  │  /impact-analysis  │  /report        │
└────┬───────────────┬──────────────────────┬─────────────┬───────────────┘
     │               │                      │             │
     ▼               ▼                      ▼             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH COMPLIANCE AGENT                             │
│                                                                           │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐ │
│  │  Router  │→ │ RAG Retrieval│→ │  Regulation  │→ │  Report/Output   │ │
│  │  Node    │  │  Tool Node   │  │  Synthesizer │  │  Formatter Node  │ │
│  └──────────┘  └──────────────┘  └─────────────┘  └──────────────────┘ │
│         ↑                                    │                           │
│         └─────── Reflection / Re-query ──────┘                           │
└─────────────────────┬───────────────────────────────────────────────────┘
                       │
         ┌─────────────┼────────────────┐
         ▼             ▼                ▼
┌──────────────┐ ┌──────────┐  ┌────────────────┐
│  LLM SERVICE │ │  VECTOR  │  │  TRANSACTION   │
│  (Ollama /   │ │   STORE  │  │   DATA STORE   │
│   vLLM)      │ │ (Qdrant) │  │  (PostgreSQL)  │
│              │ │          │  │                │
│ Mistral 7B   │ │ Hybrid   │  │ Core banking   │
│ Mixtral 8x7B │ │ search   │  │ transaction    │
│ (routing)    │ │ + filter │  │ feed           │
└──────────────┘ └──────────┘  └────────────────┘
         ▲
┌──────────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION PIPELINE                            │
│                                                                           │
│  Regulatory PDFs  →  Parser  →  Chunker  →  Embedder  →  Qdrant         │
│  (RBI, Basel III,     (pdfminer,  (semantic    (nomic-     (upsert       │
│   MiFID II)           Docling)    + overlap)    embed)      versioned)   │
└──────────────────────────────────────────────────────────────────────────┘
         ▲
┌──────────────────────────────────────────────────────────────────────────┐
│              REGULATORY CHANGE MONITORING (Airflow DAG)                   │
│  Scheduled crawl → Parse → Diff against existing → Impact analysis agent │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. RAG Pipeline Design

### 3.1 Document Ingestion Strategy

**Supported Formats:** PDF (primary), HTML circulars, DOCX, plain text

**Parser Selection:**
- `pdfminer.six` for standard PDFs — lightweight, no Java dependency
- `Docling` (IBM, open-source) for complex multi-column regulatory PDFs with tables
- `python-docx` for DOCX formats

**Versioning Strategy:**
Each document ingested receives a compound metadata key:
```
{
  "source": "RBI",
  "doc_id": "RBI_2024_FEMA_015",
  "version": "2024-03-15",
  "supersedes": "RBI_2023_FEMA_015",
  "jurisdiction": "IN",
  "regulation_family": "FEMA",
  "effective_date": "2024-04-01",
  "status": "active"  // active | superseded | draft
}
```

When a new circular supersedes an old one, the old document's chunks are marked `status: superseded` in the vector store — they remain queryable for historical/audit queries but are deprioritized in default retrieval via filter.

### 3.2 Chunking Strategy

**Approach: Hierarchical Semantic Chunking**

Regulatory documents have a natural hierarchy: Part → Chapter → Section → Subsection → Paragraph. Chunking respects this structure rather than blindly splitting by token count.

```
Stage 1: Structural split
  - Split on heading patterns (e.g., "Section 4.2", "Article 12")
  - Preserves regulation section numbers as metadata

Stage 2: Semantic refinement
  - Use sentence-transformers to detect semantic boundaries within sections
  - If a section > 512 tokens: split at semantic breakpoints (cosine similarity drop)
  - If a section < 64 tokens: merge with adjacent sibling

Stage 3: Overlap injection
  - 10% overlap (≈50 tokens) between adjacent chunks from the same section
  - Prevents context loss at boundaries
```

**Chunk Size:** 256–512 tokens (justified: regulatory sentences are dense; smaller chunks reduce noise; 512 is sufficient for most paragraph-level regulatory requirements)

**Parent Document Retrieval:** Chunks store a `parent_id` pointing to the full section. When a chunk is retrieved, the full parent section is optionally fetched for LLM context — preventing out-of-context partial answers.

### 3.3 Embedding Model

**Selected: `nomic-embed-text` (via Ollama)**

| Model | Dim | Context | License | Hosting |
|---|---|---|---|---|
| nomic-embed-text | 768 | 8192 tokens | Apache 2.0 | Self-hosted ✓ |
| bge-large-en | 1024 | 512 tokens | MIT | Self-hosted ✓ |
| text-embedding-ada-002 | 1536 | 8191 tokens | Proprietary | API only ✗ |
| e5-large-v2 | 1024 | 512 tokens | MIT | Self-hosted ✓ |

**Rationale for nomic-embed-text:**
- 8192 token context window handles long regulatory paragraphs without truncation
- Outperforms bge-large-en on MTEB legal domain benchmarks
- Fully self-hosted via Ollama — no data leaves the environment
- Apache 2.0 license — commercially safe

### 3.4 Vector Database: Qdrant

See ADR-001 for full decision record.

**Collection Schema:**
```python
collection_config = {
    "name": "regulatory_docs",
    "vectors": {
        "dense": VectorParams(size=768, distance=Distance.COSINE),
    },
    "sparse_vectors": {
        "bm25": SparseVectorParams()  # For hybrid search
    },
    "payload_schema": {
        "source": "keyword",
        "doc_id": "keyword",
        "version": "keyword",
        "jurisdiction": "keyword",
        "regulation_family": "keyword",
        "status": "keyword",
        "section_number": "text",
        "effective_date": "datetime",
        "text": "text"
    }
}
```

### 3.5 Retrieval Strategy: Hybrid Search + Reranking

**Stage 1: Hybrid Retrieval**
- Dense retrieval: cosine similarity on nomic-embed-text vectors (top-20)
- Sparse retrieval: BM25 on raw text (top-20)
- Fusion: Reciprocal Rank Fusion (RRF) combines both ranked lists
- Pre-filter: `status == "active"` by default; auditors can override to include superseded

```python
# Qdrant native hybrid search
results = client.query_points(
    collection_name="regulatory_docs",
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=20),
        Prefetch(query=sparse_vector, using="bm25", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
    query_filter=Filter(must=[FieldCondition(key="status", match=MatchValue(value="active"))])
)
```

**Stage 2: Cross-Encoder Reranking**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, 22M params, <50ms latency)
- Reranks top-10 hybrid results to top-5 for LLM context
- Rationale: bi-encoder retrieval optimizes recall; cross-encoder optimizes precision

**Stage 3: Contextual Compression**
- For long retrieved chunks: extract only the sentences most relevant to the query
- Uses `LLMLinguaCompressor` (open-source, Microsoft) — reduces token cost 3-5x with <5% quality loss on regulatory domain

### 3.6 Document Update & Version Control

```
New circular ingested
       │
       ▼
Parser extracts metadata (doc_id, supersedes_id, effective_date)
       │
       ▼
Check if supersedes_id exists in Qdrant
       │
  Yes  │  No
       ▼       ▼
Mark old     Insert new
chunks:       chunks with
status=       status=active
superseded
       │
       ▼
Impact Analysis Agent triggered (see Section 5.3)
       │
       ▼
Compliance team notified of affected policy areas
```

---

## 4. LLM Orchestration Layer

### 4.1 Foundation Model Selection

**Primary Generation Model: Mistral 7B Instruct v0.3 (self-hosted via Ollama/vLLM)**

**Routing Strategy (Multi-Model):**

| Task | Model | Justification |
|---|---|---|
| Query classification / intent | Mistral 7B (4-bit quantized) | Fast, low latency, <200ms |
| Regulatory Q&A generation | Mistral 7B Instruct | Strong instruction following, 32K context |
| Complex multi-reg synthesis | Mixtral 8x7B (MoE) | Higher accuracy for multi-hop reasoning |
| Transaction risk classification | Fine-tuned DistilBERT (custom) | <10ms, deterministic, auditable |
| Compliance report generation | Mixtral 8x7B | Long-form, structured output |
| PII detection & redaction | Presidio (rule-based + NER) | Deterministic, auditable — LLMs not used for PII |

**Why not GPT-4 / Claude?**
Proprietary APIs are referenced in the architecture as the "production upgrade path" but the prototype exclusively uses self-hosted open-source models. The primary constraint is data residency: RBI and MiFID II data cannot be processed by US-hosted third-party APIs without explicit regulatory approval. Self-hosted models eliminate this risk entirely.

### 4.2 Prompt Engineering Framework

**System Prompt Template (Regulatory Q&A):**
```
You are a regulatory compliance assistant for FinServ Global, a financial services firm 
operating under Basel III, MiFID II, and RBI regulatory frameworks.

RULES:
1. Answer ONLY from the provided regulatory context. Never speculate.
2. If the answer is not in the context, say: "This information is not found in the 
   retrieved regulatory documents. Please consult your compliance officer."
3. Always cite your sources using [DOC_ID, Section X.Y] format.
4. Flag any answer where regulations from multiple jurisdictions may conflict.
5. Never provide legal advice — provide regulatory information only.
6. Responses must be audit-ready: precise, cited, and unambiguous.

CONTEXT:
{retrieved_chunks}

QUERY: {user_query}

Respond in this format:
ANSWER: [Your cited answer]
SOURCES: [DOC_ID, Section X.Y; DOC_ID2, Section Z.A]
CONFIDENCE: [HIGH / MEDIUM / LOW] with brief reason
CROSS_JURISDICTION_FLAG: [YES/NO] — [explanation if YES]
```

**Few-Shot Examples for Transaction Screening:**
```
Example 1:
Transaction: Wire transfer of $500K to ABC Corp (KYC verified, Singapore, trade finance)
Assessment:
  Risk Rating: LOW
  Applicable Regulations: Basel III (credit exposure), MAS TF guidelines
  Required Actions: Standard due diligence, maintain transaction records 7 years
  Citations: [Basel_III_CRE, Section 4.1], [MAS_TF_2023, Section 2.3]

Example 2:
Transaction: Cross-border payment of $2M to non-KYC entity in Iran
Assessment:
  Risk Rating: CRITICAL
  Applicable Regulations: FATF Recommendation 16, RBI FEMA 2023, US OFAC SDN
  Required Actions: BLOCK transaction immediately. File SAR within 24 hours. 
                    Notify MLRO. Do not tip off counterparty.
  Citations: [FATF_R16_2023], [RBI_FEMA_2023_015, Section 5.2], [OFAC_SDN_LIST]
```

### 4.3 Guardrails

**1. Hallucination Detection**
- Every LLM response is validated: cited DOC_IDs must exist in the retrieved context
- If citation doesn't exist in retrieved chunks → response flagged, confidence set to LOW
- Implementation: post-generation parser cross-checks `SOURCES:` field against retrieved chunk metadata

**2. Output Schema Validation**
- Pydantic models enforce structured output for all agent responses
- Regex validation on citation format `[DOC_ID, Section X.Y]`
- JSON schema validation for transaction assessments

**3. PII Redaction (Pre-LLM)**
- Microsoft Presidio (open-source) scans all inputs before sending to LLM
- Entities detected: account numbers, customer names, national IDs, phone numbers
- Redacted tokens replaced with typed placeholders: `[ACCOUNT_NUMBER_1]`
- Redaction log maintained for audit trail

**4. Regulatory Accuracy Checks**
- Post-generation: extract numerical values (capital ratios, thresholds, dates) from response
- Cross-check against a structured regulatory fact database (maintained separately)
- Mismatch triggers LOW confidence flag and human review queue

**5. Jailbreak / Prompt Injection Detection**
- Input classifier (small fine-tuned model) detects instruction injection patterns
- Blocklist of known jailbreak patterns
- Any detected injection: request blocked, security event logged

---

## 5. Agentic Workflow Design

### 5.1 LangGraph Agent Architecture

**Framework: LangGraph**
Chosen over CrewAI for: stateful graph execution, native cycle support (enables reflection/re-query), first-class support for tool-calling patterns, deterministic state management critical for audit trails.

```
                    ┌─────────────┐
  User Input ──────►│   ROUTER    │
                    │   NODE      │
                    └──────┬──────┘
                           │ classify intent
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
         ┌────────┐  ┌──────────┐  ┌──────────────┐
         │  Q&A   │  │  SCREEN  │  │    IMPACT    │
         │  NODE  │  │  TXNODE  │  │  ANALYSIS    │
         └────┬───┘  └────┬─────┘  └──────┬───────┘
              │            │               │
              ▼            ▼               ▼
         ┌─────────────────────────────────────┐
         │          RAG RETRIEVAL TOOL          │
         │  hybrid_search(query, filters)       │
         └─────────────────┬───────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌──────────────────┐    ┌──────────────────────┐
    │  SYNTHESIS NODE  │    │  CROSS-REG CHECKER   │
    │  (LLM generate)  │    │  (multi-jurisdiction │
    │                  │    │   conflict detection) │
    └────────┬─────────┘    └──────────┬───────────┘
             │                         │
             └────────────┬────────────┘
                          ▼
                ┌─────────────────┐
                │  REFLECTION     │
                │  NODE           │◄─── Retry if confidence LOW
                └────────┬────────┘
                         │ confidence HIGH/MEDIUM
                         ▼
                ┌─────────────────┐
                │  GUARDRAIL      │
                │  VALIDATOR      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  FORMATTER      │
                │  NODE           │
                └────────┬────────┘
                         │
                    Final Response
```

### 5.2 Tool Definitions

```python
tools = [
    Tool(
        name="regulatory_search",
        description="""Search the regulatory knowledge base for relevant regulations,
        circulars, or guidelines. Use for any regulatory question.
        Args: query (str), jurisdiction (list[str] optional), 
              regulation_family (list[str] optional),
              include_superseded (bool, default False)""",
        func=hybrid_search
    ),
    Tool(
        name="transaction_lookup",
        description="""Look up historical transactions by type, counterparty, or 
        jurisdiction. Use for precedent-based compliance assessment.
        Args: transaction_id (str optional), filters (dict)""",
        func=transaction_db_lookup
    ),
    Tool(
        name="regulation_diff",
        description="""Compare two versions of a regulation to identify changes.
        Args: doc_id (str), version_old (str), version_new (str)""",
        func=regulation_diff
    ),
    Tool(
        name="generate_compliance_report",
        description="""Generate a structured compliance report for a set of 
        transaction IDs over a date range.
        Args: transaction_ids (list[str]), date_from (str), date_to (str),
              report_type (str): 'audit' | 'summary' | 'regulatory_submission'""",
        func=generate_report
    ),
    Tool(
        name="flag_for_human_review",
        description="""Flag a transaction or query for human compliance officer 
        review when AI confidence is low or risk is CRITICAL.
        Args: item_id (str), reason (str), priority (str): 'urgent' | 'standard'""",
        func=flag_for_review
    ),
]
```

### 5.3 State Management

```python
class ComplianceAgentState(TypedDict):
    # Input
    input: str
    intent: str  # qa | screen_transaction | impact_analysis | report
    transaction: Optional[dict]
    
    # Processing
    retrieved_chunks: List[RetrievedChunk]
    jurisdictions_triggered: List[str]
    regulations_triggered: List[str]
    
    # Output construction
    draft_response: str
    citations: List[Citation]
    confidence: str  # HIGH | MEDIUM | LOW
    
    # Control flow
    reflection_count: int  # max 2 reflection cycles
    requires_human_review: bool
    
    # Audit trail (append-only)
    audit_log: List[AuditEntry]
```

### 5.4 Error Handling Strategy

| Failure Mode | Detection | Response |
|---|---|---|
| LLM timeout (>30s) | asyncio timeout | Return cached similar query or graceful degradation message |
| Vector DB unavailable | Health check / exception | Fallback to BM25-only keyword search on document store |
| Low retrieval relevance | Score threshold <0.5 | Trigger reflection, broaden query, flag LOW confidence |
| Citation mismatch | Post-gen validator | Strip invalid citation, flag LOW confidence, queue for review |
| Ambiguous transaction | Pydantic validation | Request specific fields; return partial assessment with gaps noted |
| All models unavailable | Health check | Return maintenance page; queue request for retry |

---

## 6. Infrastructure & Non-Functional Requirements

### 6.1 Cloud Architecture (AWS)

```
┌─────────────────────────────── AWS Region: ap-south-1 (Mumbai — RBI data) ───────┐
│                                                                                    │
│  ┌──────────────────────────── EKS Cluster ───────────────────────────────────┐   │
│  │                                                                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐   │   │
│  │  │  Compliance API  │  │  LLM Service    │  │  Ingestion Workers       │   │   │
│  │  │  (FastAPI pods)  │  │  (vLLM pods)    │  │  (Celery + Airflow)     │   │   │
│  │  │  2-10 pods HPA   │  │  GPU node group │  │  2-4 pods               │   │   │
│  │  └────────┬─────────┘  └────────┬────────┘  └──────────────────────────┘   │   │
│  │           │                     │                                            │   │
│  │  ┌────────▼─────────────────────▼────────────────────────────────────────┐  │   │
│  │  │                    Internal Service Mesh (Istio)                       │  │   │
│  │  └───────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐   │   │
│  │  │  Qdrant Cluster  │  │  PostgreSQL RDS  │  │  Redis ElastiCache       │   │   │
│  │  │  (StatefulSet)   │  │  (transactions)  │  │  (query cache, sessions) │   │   │
│  │  │  3-node HA       │  │  Multi-AZ        │  │  cluster mode            │   │   │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  Network Controls: VPC private subnets only │ No public IP for data services│   │
│  │  Encryption: EBS/RDS encrypted at rest (AES-256) │ TLS 1.3 in transit      │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────┘

Additional regions:
- eu-west-1 (Ireland): MiFID II data — same topology
- us-east-1 (N. Virginia): Basel III / US data — same topology

Cross-region: No regulatory data replication across regions. 
Shared: Monitoring stack, CI/CD pipeline metadata only.
```

### 6.2 Auto-Scaling Strategy

**LLM Inference Pods (vLLM on GPU nodes):**
- Metric: GPU utilization + queue depth (custom metric via Prometheus adapter)
- Scale-out threshold: GPU util >70% OR queue depth >50 requests
- Scale-in: GPU util <20% for 10 minutes (slow scale-in to avoid cold start thrash)
- Min: 1 GPU node, Max: 5 GPU nodes (g4dn.xlarge, A10G GPU)
- Cold start mitigation: model pre-loaded in pod init; scale-out takes ~3 min

**API Pods (FastAPI):**
- Standard HPA on CPU utilization + request rate
- Min: 2 pods, Max: 10 pods, target CPU: 60%

**Ingestion Workers:**
- KEDA (Kubernetes Event-Driven Autoscaling) on Airflow queue depth
- Scale to 0 when no ingestion jobs pending

### 6.3 Cost Estimation (500 concurrent users, 10K queries/day)

| Component | Spec | Monthly Cost (AWS ap-south-1) |
|---|---|---|
| EKS Control Plane | Managed | ~$150 |
| API Pods (3x c5.xlarge avg) | 4 vCPU, 8GB | ~$180 |
| LLM GPU Nodes (2x g4dn.xlarge avg) | A10G, 16GB VRAM | ~$1,200 |
| Qdrant (3x r5.large, 200GB EBS) | 2 vCPU, 16GB | ~$420 |
| PostgreSQL RDS (db.r5.large, Multi-AZ) | 2 vCPU, 16GB | ~$350 |
| Redis ElastiCache (cache.r6g.large) | 2 vCPU, 13GB | ~$130 |
| Data Transfer, S3, CloudWatch | Misc | ~$200 |
| **Total (single region)** | | **~$2,630/month** |
| **Total (3 regions)** | | **~$7,500/month** |

*Significant cost reduction vs. OpenAI API at 10K queries/day (~$3,000-8,000/month for GPT-4 class, with data residency concerns.)*

### 6.4 Security & Compliance

**Data Residency Enforcement:**
- Kubernetes namespace per jurisdiction (`ns-in`, `ns-eu`, `ns-us`)
- NetworkPolicy: pods in `ns-in` cannot egress to `ns-eu` or `ns-us`
- Data tagged with `jurisdiction` label at ingestion; routing enforced at API gateway

**Encryption:**
- At rest: AES-256 on all EBS volumes, RDS, S3 (KMS-managed keys, per-region CMK)
- In transit: TLS 1.3 enforced; mTLS between services (Istio)
- LLM model weights: encrypted at rest; loaded into GPU memory only

**Model Access Controls:**
- vLLM endpoints accessible only from within cluster (no public endpoint)
- Service accounts with least-privilege IAM roles
- All LLM requests logged with: user_id, timestamp, query hash (not raw query for PII), response hash, latency

**Audit Logging:**
- Every compliance query logged: user_id, intent, retrieved doc_ids, model used, response hash, confidence
- Immutable audit log stored in S3 with Object Lock (WORM)
- Retention: 7 years (regulatory requirement)
- CloudTrail for all AWS API calls

**No Data Leakage to External APIs:**
- Egress firewall rules block all traffic to *.openai.com, *.anthropic.com, *.googleapis.com from data-processing namespaces
- Only exception: Prometheus telemetry to Grafana Cloud (no regulatory content — metrics only)

### 6.5 Observability Stack

**Metrics (Prometheus + Grafana):**
- LLM latency: p50, p95, p99 per model and intent type
- Token usage per query (cost tracking)
- Retrieval latency, reranking latency
- Vector DB query latency and index size
- Agent reflection cycles per query
- Human review queue depth

**LLM-Specific Evaluation Pipeline (Weekly):**
- Sample 5% of production queries (anonymized)
- Run through RAGAS evaluation against GPT-4 judge (isolated evaluation environment, no production data)
- Alert if faithfulness drops below 0.75 or relevance below 0.70

**Drift Detection:**
- Embedding distribution drift: monthly check using Maximum Mean Discrepancy (MMD) on query embeddings vs. index embeddings
- Retrieval relevance drift: track mean reranking scores over time; alert if 7-day moving average drops >15%
- Response confidence distribution: alert if LOW confidence rate exceeds 20% over 24 hours

---

## 7. Data Flow for Key Scenarios

### 7.1 Regulatory Q&A
```
User query → PII scan → Intent classification → RAG retrieval (hybrid + rerank)
→ Context compression → LLM generation (Mistral 7B) → Citation validation
→ Confidence scoring → Format response → Audit log → Return to user
Latency target: <5 seconds (p95)
```

### 7.2 Transaction Screening
```
Transaction payload → Schema validation → PII redaction → Intent: screen_transaction
→ Extract transaction attributes → Parallel RAG queries (by jurisdiction + instrument type)
→ Cross-regulation synthesis → Risk rating (DistilBERT classifier) → LLM narrative generation
→ Structured assessment output → Audit log → Return to user
Latency target: <10 seconds (p95)
```

### 7.3 Regulatory Change Impact Analysis
```
New circular detected → Parse + embed → Qdrant upsert → Mark old version superseded
→ Trigger impact agent → Query: "Which existing policies reference [old_regulation]?"
→ Retrieve affected policies → LLM: identify specific clauses affected → 
→ Generate impact report → Notify compliance team → Update knowledge base
Duration: typically 2-5 minutes (async, not real-time)
```

---

## 8. Limitations & Future Roadmap

| Current Limitation | Mitigation | Roadmap Item |
|---|---|---|
| Mistral 7B accuracy on complex multi-hop regulatory reasoning | Reflection loops, human review queue | Fine-tune on regulatory domain; evaluate Mixtral 8x7B |
| No real-time transaction feed integration | Sample payloads for prototype | Kafka consumer integration in v2 |
| English-only regulatory documents | English covers prototype scope | Add multilingual embeddings for Hindi/French RBI/EU docs |
| GPU cold start latency (~3 min) | Min 1 GPU always warm | Reserved instances + model caching |
| Manual evaluation dataset (20 QA pairs) | Covers prototype scope | Expand to 500+ pairs; expert-validated |

---

*Document version: 1.0 | Status: Submitted for evaluation*
