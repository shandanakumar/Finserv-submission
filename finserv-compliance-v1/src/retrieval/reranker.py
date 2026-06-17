"""
src/retrieval/reranker.py — No download version
No huggingface, no internet, no SSL issues.
"""
from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.hybrid_search import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:

    def rerank(
        self,
        query: str,
        chunks: list["RetrievedChunk"],
        top_k: int = 5,
    ) -> list["RetrievedChunk"]:
        if not chunks:
            return []

        query_terms = set(re.sub(r"[^\w\s]", "", query.lower()).split())

        for chunk in chunks:
            text_lower = chunk.text.lower()
            term_hits  = sum(1 for t in query_terms if t in text_lower)
            keyword_score      = term_hits / max(len(query_terms), 1)
            chunk.rerank_score = (0.6 * chunk.rrf_score * 10) + (0.4 * keyword_score)

        reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        return reranked[:top_k]


def format_chunks_for_llm(chunks: list["RetrievedChunk"]) -> str:
    if not chunks:
        return "No relevant regulatory context found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        reg    = chunk.metadata.get("regulation_family", "Unknown")
        sec    = chunk.metadata.get("section_title", "")
        doc    = chunk.metadata.get("source_document", chunk.metadata.get("doc_id", ""))
        header = f"[{i}] {reg}"
        if sec:
            header += f" — {sec}"
        if doc:
            header += f" ({doc})"
        parts.append(f"{header}\n{chunk.text.strip()}")

    return "\n\n---\n\n".join(parts)