"""
src/retrieval/hybrid_search.py
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import settings
from src.llm import get_embedder
from src.ingestion.embedder import get_qdrant_client

logger = logging.getLogger(__name__)
from qdrant_client.models import Filter, FieldCondition, MatchValue

@dataclass
class RetrievedChunk:
    chunk_id:     str
    text:         str
    metadata:     dict        # payload from Qdrant — flat dict
    dense_score:  float = 0.0
    bm25_score:   float = 0.0
    rrf_score:    float = 0.0
    rerank_score: float = 0.0


def _qdrant_search(client, collection: str, vector: list, limit: int, search_filter=None) -> list:
    """Works with both old (.search) and new (.query_points) qdrant-client."""

    # Use passed filter if provided, otherwise default to filtering superseded
    if search_filter is None:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="is_superseded",
                    match=MatchValue(value=False)
                )
            ]
        )
    # If search_filter was passed from search() it already contains
    # is_superseded=False + regulation_id filter — use it as-is

    try:
        r = client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=search_filter,
        )
        return r.points
    except AttributeError:
        return client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )


class HybridSearcher:
    def __init__(self):
        self.qdrant     = get_qdrant_client()
        self.embedder   = get_embedder()
        self.collection = settings.QDRANT_COLLECTION
        self._bm25      = None
        self._bm25_chunks: list[RetrievedChunk] = []

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        points, _ = self.qdrant.scroll(
            collection_name=self.collection,
            limit=5000,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="is_superseded",
                        match=MatchValue(value=False)
                    )
                ]
            ),
        )
        if not points:
            logger.warning("BM25: no points found in collection '%s'", self.collection)
            return
        self._bm25_chunks = [
            RetrievedChunk(
                chunk_id=p.payload.get("chunk_id", str(p.id)),
                text=p.payload.get("text", ""),
                metadata=p.payload,
            )
            for p in points
        ]
        tokenized  = [c.text.lower().split() for c in self._bm25_chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built on %d chunks", len(self._bm25_chunks))

    def search(
        self,
        query: str,
        top_k: int = 10,
        regulation_families: Optional[list[str]] = None,
        jurisdictions:       Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        print(f"DEBUG search() called: regulation_families={regulation_families}")
        # ── Build Qdrant filter ───────────────────────────────────
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = [
            FieldCondition(key="is_superseded", match=MatchValue(value=False))
        ]

        if regulation_families:
            conditions.append(
                FieldCondition(
                    key="regulation_id",
                    match=MatchValue(value=regulation_families[0])
                )
            )

        search_filter = Filter(must=conditions)

        # ── Query expansion ───────────────────────────────────────
        expanded_query = query
        expansions = {
            "re-verification": "updation periodic re-kyc",
            "rekyc":           "updation periodic re-kyc",
            "re kyc":          "updation periodic re-kyc",
            "how often":       "periodic frequency years",
            "frequency":       "periodic updation years",
            "high risk":       "high risk customer 2 years enhanced due diligence",
            "medium risk":     "medium risk customer 8 years",
            "low risk":        "low risk customer 10 years",
            "kyc update":      "updation periodic re-kyc frequency",
            "kyc renewal":     "updation periodic re-kyc frequency",
        }
        for term, expansion in expansions.items():
            if term.lower() in query.lower():
                expanded_query = query + " " + expansion
                break

        # ── Dense retrieval ───────────────────────────────────────
        query_vec     = self.embedder.embed(query)
        dense_results = _qdrant_search(
            self.qdrant, self.collection,
            query_vec, top_k * 2, search_filter
        )

        dense_chunks: dict[str, RetrievedChunk] = {}
        for p in dense_results:
            cid = p.payload.get("chunk_id", str(p.id))
            dense_chunks[cid] = RetrievedChunk(
                chunk_id=cid,
                text=p.payload.get("text", ""),
                metadata=p.payload,
                dense_score=p.score,
            )

        # ── BM25 retrieval ────────────────────────────────────────
        if self._bm25 is None:
            self._build_bm25()

        bm25_chunks: dict[str, RetrievedChunk] = {}
        if self._bm25 and self._bm25_chunks:

            # Filter BM25 pool to same regulation if specified
            bm25_pool = self._bm25_chunks
            if regulation_families:
                bm25_pool = [
                    c for c in self._bm25_chunks
                    if c.metadata.get("regulation_id") == regulation_families[0]
                ]

            if bm25_pool:
                from rank_bm25 import BM25Okapi
                tokenized  = [c.text.lower().split() for c in bm25_pool]
                local_bm25 = BM25Okapi(tokenized)
                scores     = local_bm25.get_scores(expanded_query.lower().split())
                top_idx    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]
                for idx in top_idx:
                    c = bm25_pool[idx]
                    bm25_chunks[c.chunk_id] = RetrievedChunk(
                        chunk_id=c.chunk_id,
                        text=c.text,
                        metadata=c.metadata,
                        bm25_score=float(scores[idx]),
                    )

        # ── RRF fusion ────────────────────────────────────────────
        all_ids      = set(dense_chunks) | set(bm25_chunks)
        dense_ranked = {cid: rank for rank, cid in enumerate(dense_chunks, 1)}
        bm25_ranked  = {cid: rank for rank, cid in enumerate(bm25_chunks,  1)}

        fused: list[RetrievedChunk] = []
        k = 60
        for cid in all_ids:
            rrf = 0.0
            if cid in dense_ranked: rrf += 1.0 / (k + dense_ranked[cid])
            if cid in bm25_ranked:  rrf += 1.0 / (k + bm25_ranked[cid])
            base             = dense_chunks.get(cid) or bm25_chunks[cid]
            base.rrf_score   = rrf
            base.dense_score = dense_chunks[cid].dense_score if cid in dense_chunks else 0.0
            base.bm25_score  = bm25_chunks[cid].bm25_score   if cid in bm25_chunks  else 0.0
            fused.append(base)

        fused.sort(key=lambda c: c.rrf_score, reverse=True)
        return fused[:top_k]
