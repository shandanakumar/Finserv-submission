"""
src/ingestion/embedder.py
Qdrant client with disk persistence — no Docker, no re-ingestion on restart.
"""
from __future__ import annotations
import hashlib
import logging
import os
from qdrant_client.models import PointStruct, UpdateStatus
from config.settings import settings
from src.ingestion.chunker import DocumentChunk
from src.llm import get_embedder

logger = logging.getLogger(__name__)

_qdrant_client = None


def _make_qdrant_client():
    """
    Create Qdrant client.
    disk mode (default): saves vectors to ./qdrant_storage folder.
                         Persists across restarts — no re-ingestion needed.
    memory mode:         in-process only, resets on restart.
                         Only use for quick tests.

    Switch in .env:
        QDRANT_MODE=disk    ← use this always
        QDRANT_MODE=memory  ← only for quick tests
    """
    from qdrant_client import QdrantClient

    mode = getattr(settings, "QDRANT_MODE", "disk")

    if mode == "memory":
        logger.info("Qdrant: IN-MEMORY mode (data lost on restart)")
        return QdrantClient(":memory:")
    else:
        path = "./qdrant_storage"
        os.makedirs(path, exist_ok=True)
        logger.info("Qdrant: DISK mode — persisting to %s", path)
        return QdrantClient(path=path)


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = _make_qdrant_client()
    return _qdrant_client


class VectorStoreIngester:
    def __init__(self):
        self.qdrant     = get_qdrant_client()
        self.embedder   = get_embedder()
        self.collection = settings.QDRANT_COLLECTION

    def ensure_collection_exists(self) -> None:
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection '%s' dim=%d",
                        self.collection, settings.EMBEDDING_DIMENSION)
        else:
            logger.info("Collection '%s' already exists — skipping",
                        self.collection)

    def _chunk_id_to_int(self, chunk_id: str) -> int:
        return int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16) % (2**63)

    def ingest(self, chunks: list[DocumentChunk], batch_size: int = 20,version_metadata: dict = None,) -> int:
        if not chunks:
            return 0
        vm = version_metadata or {}
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch   = chunks[i: i + batch_size]
            vectors = self.embedder.embed_batch([c.text for c in batch])

            points = []
            for chunk, vector in zip(batch, vectors):
                points.append(PointStruct(
                    id=self._chunk_id_to_int(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id":          chunk.chunk_id,
                        "text":              chunk.text,
                        "doc_id":            chunk.doc_id,
                        "source_document":   chunk.source,
                        "regulation_family": chunk.regulation_family,
                        "jurisdiction":      chunk.jurisdiction,
                        "section_title":     chunk.section_title,
                        "section_number":    chunk.section_number,
                        "status":            chunk.status,
                        "chunk_index":       chunk.chunk_index,

                        
                        "regulation_id":  vm.get("regulation_id",  chunk.regulation_family),
                        "version":        vm.get("version",         "v1"),
                        "effective_from": vm.get("effective_from",  ""),
                        "effective_to":   vm.get("effective_to",    "current"),
                        "source_url":     vm.get("source_url",      ""),
                        "page_number":    chunk.chunk_index,   # best approximation without PDF page tracking
                        "is_superseded":  vm.get("status", "active") == "superseded",
                    }
                ))

            result = self.qdrant.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )
            if result.status == UpdateStatus.COMPLETED:
                total += len(points)
                logger.info("Upserted %d chunks (batch %d)",
                            len(points), i // batch_size + 1)

        return total

    def collection_stats(self) -> dict:
        info = self.qdrant.get_collection(self.collection)
        return {
            "total_points": info.points_count,
            "vector_size":  info.config.params.vectors.size,
            "status":       info.status.name,
        }