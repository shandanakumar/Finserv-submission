# ADR-002: Model Hosting Strategy (Self-Hosted vs. API)

**Status:** Accepted  
**Date:** June 2025  
**Deciders:** AI Architect, CISO, Compliance Head  
**Context Level:** System-wide

---

## Context

The Regulatory Compliance Assistant requires LLM inference for multiple tasks: regulatory Q&A generation, transaction risk narration, report generation, and query classification. A key architectural decision is whether to use third-party LLM APIs (OpenAI, Anthropic, Google) or self-host open-source models within FinServ's infrastructure.

This decision has significant implications for:
- **Data residency:** RBI Master Directions require financial data to be stored and processed in India. MiFID II imposes data sovereignty requirements in the EU. Sending regulatory documents or transaction data to US-hosted LLM APIs may violate these requirements.
- **Cost at scale:** API costs grow linearly with usage; self-hosted costs are primarily fixed (compute).
- **Latency:** API round-trips add network latency; self-hosted inference is local to the cluster.
- **Vendor lock-in:** API dependency creates risk if providers change pricing, terms, or deprecate models.
- **Audit requirements:** Regulators expect FinServ to demonstrate control over AI systems processing regulated data.

---

## Decision

**Self-hosted open-source models as primary inference layer, with optional API fallback for non-regulated tasks only.**

| Use Case | Primary | Fallback |
|---|---|---|
| Regulatory Q&A | Mistral 7B (self-hosted, Ollama/vLLM) | Mixtral 8x7B (self-hosted) |
| Report generation | Mixtral 8x7B (self-hosted) | None — must remain self-hosted |
| Transaction screening | Mistral 7B + DistilBERT classifier | None |
| Evaluation / quality scoring | Mistral 7B (self-hosted) | API (isolated eval environment, non-production data only) |
| Embeddings | nomic-embed-text (self-hosted) | bge-large-en (self-hosted) |

**Proprietary APIs are explicitly prohibited from receiving regulatory document text, transaction data, or any data classified as confidential under FinServ's data governance policy.**

---

## Alternatives Considered

### Option A: Fully Self-Hosted Open-Source ✓ SELECTED (for regulated data)

**Models evaluated:**

| Model | Params | VRAM (4-bit) | Benchmark (MMLU) | Regulatory Q&A (internal eval) |
|---|---|---|---|---|
| Mistral 7B Instruct v0.3 | 7B | ~5GB | 62.5 | 78/100 |
| Mixtral 8x7B Instruct | 47B (8x7B MoE) | ~26GB | 70.6 | 85/100 |
| Llama 3 8B Instruct | 8B | ~5.5GB | 68.4 | 76/100 |
| Llama 3 70B Instruct | 70B | ~42GB | 82.0 | 88/100 |
| Phi-3 Medium 14B | 14B | ~9GB | 78.0 | 80/100 |

**Selected stack:**
- **Mistral 7B Instruct** for latency-sensitive tasks (classification, short Q&A): fits on single A10G GPU (16GB VRAM), <2s inference
- **Mixtral 8x7B** for accuracy-critical tasks (report generation, complex multi-reg synthesis): requires 2x A10G or 1x A100, 4-8s inference

**Serving infrastructure:**
- **vLLM** for production (continuous batching, PagedAttention, 3-4x throughput vs naive serving)
- **Ollama** for local development and prototype (simpler setup)

**Pros:**
- Full data residency compliance — no regulatory data leaves FinServ's infrastructure
- Fixed compute cost; becomes cheaper than API at ~500+ queries/day
- No API rate limits or outages from third-party providers
- Full model version control — no surprise model changes from providers
- Auditable: exact model version and weights locked for reproducibility
- Can fine-tune on FinServ's own regulatory domain data

**Cons:**
- GPU infrastructure cost and operational complexity
- Model quality ceiling below frontier APIs (GPT-4, Claude Opus) by ~10-15% on complex reasoning
- Requires MLOps capability to manage model lifecycle, quantization, updates

**Mitigation for quality gap:**
- Hybrid retrieval + reranking improves RAG quality significantly regardless of LLM
- Structured prompts + few-shot examples narrow the gap for domain-specific tasks
- Human review queue catches low-confidence outputs
- In 12-18 months: fine-tune Mistral/Llama on FinServ's historical compliance Q&A data

---

### Option B: Fully API-Based (OpenAI / Anthropic / Google)

**Pros:**
- Frontier model quality (GPT-4, Claude Opus)
- No GPU infrastructure to manage
- Instant access to latest model versions

**Cons:**
- **Data residency violation risk:** OpenAI and Anthropic process data on US servers. Without a Business Associate Agreement and explicit regulatory carve-out, sending RBI-regulated data violates RBI's IT Framework for NBFC/Banks and potentially FEMA provisions.
- **MiFID II Article 25:** Firms must demonstrate control over algorithmic systems. Reliance on black-box third-party API with unpredictable model updates is difficult to defend to regulators.
- **Cost at scale:** 10K queries/day at ~2K tokens/query (input + output) = ~60M tokens/month. GPT-4 at $30/1M tokens = ~$1,800/month per region, growing linearly. Self-hosted is ~$1,200/month fixed across all query volumes.
- **Vendor lock-in:** OpenAI has changed pricing 3x in 2 years; model deprecations with 6-month notice create operational risk for regulated workflows.
- **Audit trail:** API providers do not provide per-query model weight snapshots required for reproducibility audits.

**Verdict: Eliminated for regulated data processing.** Permitted only for internal tooling with no regulatory data exposure (e.g., developer productivity tools, internal documentation search).

---

### Option C: Hybrid — Self-Hosted for Regulated Data, API for Non-Regulated

**The nuanced middle ground:** Use self-hosted models for anything touching regulatory documents or transaction data. Use API models for:
- Internal knowledge base queries (no regulatory content)
- Developer tooling
- Evaluation framework (in isolated environment with synthetic/public data only)

**This is the implemented approach** — self-hosted is the primary system; API access is not wired into the regulated data path.

---

## Consequences

### Positive
- Full regulatory compliance on data residency
- Cost-effective at scale
- Reproducible audit trail (model version pinned, weights stored)
- Independence from third-party API terms and pricing changes

### Negative
- 10-15% accuracy gap vs frontier models on complex multi-step reasoning
- GPU operational overhead (patching, scaling, monitoring)
- Model updates require internal testing and rollout process

### Trade-off Accepted
The 10-15% accuracy gap is accepted because:
1. The gap is in complex reasoning, not domain-specific retrieval (where RAG quality matters more)
2. Regulatory compliance requirement is non-negotiable and outweighs accuracy marginal gain
3. Human review queue provides a safety net for low-confidence outputs
4. Fine-tuning roadmap closes the gap over 12-18 months

---

## Review Triggers

This decision should be revisited if:
- Anthropic or OpenAI establish India-region data processing with regulatory certification
- A frontier API provider achieves RBI / MiFID II data residency compliance
- Open-source model quality (Llama 4, Mistral Large) closes the gap to <5% vs frontier APIs

---

*ADR-002 | Reviewed: June 2025 | Next review: December 2025*
