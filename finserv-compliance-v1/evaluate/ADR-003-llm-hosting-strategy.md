# ADR-003: LLM Hosting Strategy — Self-Hosted vs Managed API

**Status:** Accepted  
**Date:** 2024-01-15  
**Deciders:** AI Architect

---

## Context

The system requires LLM inference for:
- Query intent classification (~50 tokens)
- Regulatory answer synthesis (~2000 tokens context, ~200 tokens output)
- Quality reflection (~500 tokens)
- Report generation (~4000 tokens context, ~1000 tokens output)

Constraints:
- Open-source models only (assignment requirement)
- Data residency — regulatory document content cannot be sent to US-hosted external APIs without addressing compliance
- Prototype must run on developer hardware (Windows, no dedicated GPU available)
- Production must support 500 concurrent users, 10K queries/day

---

## Decision

**Selected: AWS Bedrock (V1 — production prototype)**  
**Alternative documented: Ollama self-hosted (V2 — full data sovereignty)**

---

## Alternatives Considered

| Option | Cost (10K q/day) | GPU Required | Data Residency | Latency | Decision |
|---|---|---|---|---|---|
| **AWS Bedrock (Mistral)** | ~$15/day | None | AWS us-east-1 | 1.5s | ✅ V1 Selected |
| **Ollama + Mistral 7B** | $0 API cost | T4 16GB | 100% local | 2-3s | ✅ V2 Alternative |
| **OpenAI GPT-4** | ~$300/day | None | US servers | 2s | ❌ Rejected — proprietary, data residency |
| **Anthropic Claude API** | ~$150/day | None | US servers | 1s | ❌ Rejected — proprietary |
| **vLLM + Mistral** | $0 API cost | A100 80GB | 100% local | 0.5s | ✅ V2 Production path |
| **HuggingFace TGI** | $0 API cost | T4 16GB | 100% local | 2s | ❌ Complex setup |

---

## Model Selection Rationale

**Why Mistral 7B (primary):**
- Apache 2.0 licence — free for commercial use, deployable on-premise
- Outperforms LLaMA 2 13B on reasoning benchmarks despite smaller size
- Fits in 4GB VRAM — runs on standard T4 GPU (g4dn.xlarge, $0.50/hr)
- Grouped Query Attention (GQA) reduces memory bandwidth requirements

**Why Mixtral 8x7B (complex tasks):**
- Mixture of Experts architecture — 45B parameter quality at 7B inference cost
- Used for report generation and multi-regulation cross-reference queries
- Apache 2.0 licence

**Why NOT LLaMA 2 70B:** Requires 80GB VRAM — A100 GPU costs $3.00/hr vs $0.50/hr for T4. 6x cost increase not justified for compliance Q&A where 7B models perform adequately.

---

## Consequences

**V1 — AWS Bedrock:**
- ✅ No GPU procurement — immediate deployment
- ✅ Auto-scales with demand — no capacity planning for spikes
- ✅ Data stays within AWS us-east-1 — addresses most data residency requirements
- ⚠️ Cost grows linearly with usage — 10K queries/day at avg 2000 tokens = ~$15/day
- ⚠️ Not suitable for RBI India regulations requiring data to stay in India (AWS India region = ap-south-1, needs separate deployment)

**V2 — Ollama Self-Hosted:**
- ✅ Zero API cost — only infrastructure cost
- ✅ 100% data sovereignty — regulatory documents never leave bank premises
- ✅ No internet dependency — works in air-gapped environments
- ⚠️ Requires GPU procurement (g4dn.xlarge minimum)
- ⚠️ Operational overhead — model updates, GPU monitoring, capacity planning

---

## Production Recommendation

For RBI India compliance specifically, **V2 with vLLM on AWS ap-south-1** satisfies both data residency (India) and throughput requirements. vLLM's continuous batching achieves 10x higher throughput than Ollama at the same GPU cost, making it viable for 500 concurrent users.

```
Cost estimate — V2 production (500 users, 10K q/day):
  GPU: 2× g4dn.2xlarge (T4, 16GB) = $1.00/hr × 24 = $720/month
  Storage: 100GB EBS for model weights = $10/month
  Total: ~$730/month vs ~$450/month Bedrock
  Break-even: ~20K queries/day (above that, self-hosted is cheaper)
```
