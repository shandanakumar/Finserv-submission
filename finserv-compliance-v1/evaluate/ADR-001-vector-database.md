# ADR-001: Vector Database Selection

**Status:** Accepted  
**Date:** 2024-01-15  
**Deciders:** AI Architect

---

## Context

The FinServ Compliance Assistant requires a vector database to store and retrieve 2,882+ embedding vectors representing chunks from regulatory documents (Basel III, MiFID II, RBI KYC, RBI PSL, FATF). The database must support:

- Cosine similarity search on 1024-dimensional vectors
- Metadata filtering (regulation_id, is_superseded, jurisdiction)
- Hybrid search compatibility (dense + BM25)
- Regulatory data residency requirements — data must not leave the bank's infrastructure
- No external API calls during inference

---

## Decision

**Selected: Qdrant (local disk mode)**

---

## Alternatives Considered

| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Qdrant (local)** | Open source, disk persistence, rich filtering, no Docker needed in local mode | Single-node only (local mode) | ✅ Selected |
| **Pinecone** | Fully managed, excellent performance | Proprietary, data leaves infrastructure, violates RBI/MiFID II data residency | ❌ Rejected |
| **Weaviate** | Open source, GraphQL API, multi-modal | Heavy resource requirements, complex setup | ❌ Rejected |
| **pgvector** | Runs in existing PostgreSQL, familiar ops | Poor performance at scale, no hybrid search | ❌ Rejected |
| **ChromaDB** | Simple Python API, easy setup | Limited metadata filtering, not production-grade | ❌ Rejected |
| **FAISS** | Extremely fast, battle-tested | No metadata filtering, no persistence without wrapper | ❌ Rejected |

---

## Consequences

**Positive:**
- Zero data egress — all vectors stored on local disk (`./qdrant_storage`)
- Rich payload filtering enables version-aware retrieval (`is_superseded=False`)
- Supports both `query_points` (new API) and `search` (legacy API) for compatibility
- Disk persistence means no re-ingestion on server restart

**Negative:**
- Local disk mode is single-process only — concurrent access from multiple processes requires Qdrant server mode (Docker)
- For production at scale (500+ concurrent users), must migrate to Qdrant Cloud or self-hosted Qdrant server with replication

**Migration path for production:**
```python
# Development (current)
QdrantClient(path="./qdrant_storage")

# Production
QdrantClient(url="http://qdrant-service:6333", api_key=settings.QDRANT_API_KEY)
```

---

## Trade-off Rationale

Data residency was the primary constraint. Pinecone, despite superior performance, was eliminated immediately because regulatory documents (RBI circulars, Basel III frameworks) cannot be sent to US-hosted external APIs under RBI data localisation guidelines and MiFID II data protection requirements.

Qdrant in local disk mode provides 95% of the production functionality needed for a prototype while maintaining full data sovereignty.
