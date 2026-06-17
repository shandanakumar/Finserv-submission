# ADR-001: Vector Database Selection

**Status:** Accepted  
**Date:** June 2025  
**Deciders:** AI Architect  
**Context Level:** System-wide

---

## Context

The Regulatory Compliance Assistant requires a vector database to store and retrieve embeddings of regulatory documents (Basel III, MiFID II, RBI circulars). The database must support:

1. **Hybrid search** — semantic (dense) + keyword (sparse/BM25) in a single query, critical for regulatory lookups where exact terms (e.g., "Tier 1 Capital Ratio", "Article 411 CRR") must be matched precisely
2. **Rich metadata filtering** — filter by jurisdiction, regulation family, version, document status without post-processing
3. **Self-hosted deployment** — regulatory data cannot be sent to third-party hosted vector DBs; must run within FinServ's data perimeter
4. **Production-grade HA** — 3-node cluster with automatic failover
5. **Versioned document management** — ability to mark documents as `active` or `superseded` and filter accordingly
6. **Scale** — 500K+ document chunks across 200+ regulatory circulars per year, growing ~15% annually

---

## Decision

**Selected: Qdrant (v1.9+)**

---

## Alternatives Considered

### Option A: Qdrant ✓ SELECTED
- **Hybrid search:** Native sparse + dense vector support; RRF fusion built-in
- **Filtering:** Rich payload filtering at query time; no post-processing needed
- **Self-hosted:** Docker/Kubernetes-native; Helm chart available; strong HA support
- **Performance:** Rust-based; benchmarks show 2-5x throughput vs Python-based alternatives at equivalent hardware
- **License:** Apache 2.0 — commercially safe
- **Maturity:** Production-grade; used by Flipkart, Meesho, and other large-scale deployments
- **Cost:** Free (self-hosted); Qdrant Cloud available as managed fallback

### Option B: Weaviate
- **Hybrid search:** Supported via BM25 + vector; good but less mature than Qdrant's RRF implementation
- **Self-hosted:** Kubernetes-ready, Helm chart available
- **Filtering:** GraphQL interface; more complex than Qdrant's REST/gRPC
- **Concern:** Higher memory footprint; Java-based modules add operational complexity
- **License:** BSD 3-Clause for core; some modules (reranking) require cloud
- **Verdict:** Strong alternative but Qdrant's Rust performance and simpler operational model wins

### Option C: pgvector (PostgreSQL extension)
- **Hybrid search:** Requires manual BM25 implementation (tsvector); not native hybrid
- **Self-hosted:** Trivially — already have PostgreSQL in the stack
- **Filtering:** Native SQL — extremely flexible
- **Concern:** Does not scale to millions of vectors without significant tuning; ANN search quality degrades without HNSW index (available but limited); not designed for high-throughput vector-first workloads
- **Verdict:** Good for <100K vectors or when operational simplicity trumps performance; not suitable as primary vector store at FinServ's scale

### Option D: ChromaDB
- **Hybrid search:** Limited — no native sparse vector support; requires external BM25
- **Self-hosted:** Python-native, easy to deploy
- **Concern:** Not production-grade for HA; no clustering in open-source version; persistent storage reliability concerns at scale
- **Verdict:** Excellent for prototyping and development; insufficient for production FinServ workloads

### Option E: Pinecone (cloud-hosted)
- **Hybrid search:** Excellent native support
- **Self-hosted:** Not available — hosted only
- **Data residency:** **DISQUALIFYING** — regulatory documents cannot be sent to Pinecone's US-hosted infrastructure without violating RBI data localization requirements
- **Verdict:** Eliminated due to data residency constraints

### Option F: Milvus
- **Hybrid search:** Supported
- **Self-hosted:** Kubernetes-ready; more complex operational footprint (etcd, MinIO dependencies)
- **Concern:** Operational complexity significantly higher than Qdrant; etcd dependency adds failure surface
- **Verdict:** Enterprise-grade but operationally heavier than needed; Qdrant preferred

---

## Consequences

### Positive
- Native hybrid search eliminates need for external BM25 service; reduces query latency
- Rust performance handles 10K queries/day with head room
- Apache 2.0 license removes licensing risk for regulated financial institution
- Rich filtering enables jurisdiction/version routing without application-layer logic
- Active development community; 1.x stable API; good Python client

### Negative / Trade-offs
- Qdrant is newer than Elasticsearch/Solr — less battle-tested in some large enterprise environments; mitigated by production references and active maintenance
- No native full-text search UI (unlike Elasticsearch's Kibana); requires Grafana + custom dashboards for observability
- Sparse vector index (for BM25) requires `fastembed` or custom tokenization; adds pipeline step

### Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Qdrant cluster failure | 3-node HA with async replication; tested failover procedure |
| Data corruption on upgrade | Snapshot to S3 before every upgrade; tested restore procedure |
| Performance degradation at scale | Load tested to 2M chunks; HNSW index parameters tuned |

---

## Implementation Notes

```yaml
# Helm values excerpt
qdrant:
  replicaCount: 3
  persistence:
    size: 200Gi
    storageClass: gp3
  resources:
    requests:
      memory: "8Gi"
      cpu: "2"
    limits:
      memory: "16Gi"
  config:
    collection:
      hnsw_config:
        m: 16
        ef_construct: 200
```

---

*ADR-001 | Reviewed: June 2025 | Next review: December 2025*
