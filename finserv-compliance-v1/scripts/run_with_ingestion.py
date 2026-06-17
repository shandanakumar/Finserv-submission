"""
scripts/run_with_ingestion.py

Ingests PDFs AND starts FastAPI server in one command.
Solves the in-memory Qdrant problem — same process = shared memory.

HOW TO RUN:
    venv\Scripts\activate
    python scripts\run_with_ingestion.py

    Then open: http://localhost:8000/docs
"""

import sys
import os
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

os.environ.update({
    "LLM_BACKEND":             "bedrock",
    "AWS_REGION":              "us-east-1",
    "QDRANT_MODE":             "memory",
    "BEDROCK_PRIMARY_MODEL":   "mistral.mistral-7b-instruct-v0:2",
    "BEDROCK_COMPLEX_MODEL":   "mistral.mixtral-8x7b-instruct-v0:1",
    "BEDROCK_EMBEDDING_MODEL": "amazon.titan-embed-text-v2:0",
    "EMBEDDING_DIMENSION":     "1024",
    "QDRANT_COLLECTION":       "regulatory_docs",
})

# ── Patch Titan embed format ──────────────────────────────────
import boto3
import json as _json
from src.llm import bedrock_client as _bc

_BEDROCK = boto3.client("bedrock-runtime", region_name="us-east-1")

def _fixed_embed(self, text):
    if not text or not text.strip():
        return [0.0] * 1024
    resp = _BEDROCK.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=_json.dumps({"inputText": text[:8000]}),
        contentType="application/json",
        accept="application/json",
    )
    return _json.loads(resp["body"].read())["embedding"]

def _fixed_init(self):
    self.client     = _bc.get_bedrock_client()
    self.model_id   = "amazon.titan-embed-text-v2:0"
    self.dimensions = 1024

_bc.BedrockEmbedder.__init__ = _fixed_init
_bc.BedrockEmbedder.embed    = _fixed_embed

# ── Ingest ────────────────────────────────────────────────────
from pathlib import Path
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import SemanticChunker
from src.ingestion.embedder import VectorStoreIngester
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import src.ingestion.embedder as emb_module

DOCS_DIR = Path("real_docs")
SAMPLE_DIR = Path("sample_docs")

print(f"\n[STARTUP] Ingesting from {DOCS_DIR}/ ...")

# Fresh in-memory Qdrant
import os as _os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

_os.makedirs("./qdrant_storage", exist_ok=True)
emb_module._qdrant_client = QdrantClient(path="./qdrant_storage")

# Only create collection if it doesn't exist yet
_existing = [c.name for c in emb_module._qdrant_client.get_collections().collections]
if "regulatory_docs" not in _existing:
    emb_module._qdrant_client.create_collection(
        "regulatory_docs",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print("[STARTUP] Created Qdrant collection on disk")
else:
    print("[STARTUP] Qdrant collection already exists on disk — skipping ingestion")
# Metadata detection
def detect_metadata(filename):
    name = filename.upper()
    if "BASEL" in name or "BIS" in name or "D424" in name:
        return {"source": "BIS", "regulation_family": "Basel_III", "jurisdiction": "GLOBAL"}
    elif "RBI" in name or "KYC" in name or "MASTER" in name:
        return {"source": "RBI", "regulation_family": "RBI_KYC", "jurisdiction": "IN"}
    elif "MIFID" in name or "CELEX" in name or "ESMA" in name:
        return {"source": "ESMA", "regulation_family": "MiFID_II", "jurisdiction": "EU"}
    elif "FATF" in name or "AML" in name or "GAFI" in name:
        return {"source": "FATF", "regulation_family": "AML_CFT", "jurisdiction": "GLOBAL"}
    return {}

loader  = DocumentLoader()
chunker = SemanticChunker(max_chunk_tokens=400, overlap_tokens=50)
ingester = VectorStoreIngester()

# Build metadata overrides
seen = set()
pdf_files = []
for f in DOCS_DIR.iterdir():
    if f.suffix.lower() == ".pdf" and f.name not in seen:
        seen.add(f.name)
        pdf_files.append(f)

metadata_overrides = {f.name: detect_metadata(f.name) for f in pdf_files}

docs = loader.load_directory(str(DOCS_DIR), metadata_overrides=metadata_overrides)
rbi_docs = [d for d in loader.load_directory(str(SAMPLE_DIR))
            if "RBI" in d.source or "rbi" in d.doc_id.lower()]
docs.extend(rbi_docs)
print(f"[STARTUP] Added {len(rbi_docs)} clean RBI sample docs")
print(f"[STARTUP] Loaded {len(docs)} documents")

total = 0
for doc in docs:
    chunks = chunker.chunk_document(doc)
    n      = ingester.ingest(chunks, batch_size=20)
    total += n
    print(f"[STARTUP]   {doc.doc_id}: {n} chunks stored")

print(f"[STARTUP] Ingestion complete — {total} chunks in Qdrant")

# Save summary for Streamlit UI
os.makedirs("outputs", exist_ok=True)
summary = {
    "ingested_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_chunks": total,
    "documents": [
        {
            "doc_id":            doc.doc_id,
            "source":            doc.source,
            "regulation_family": doc.regulation_family,
            "jurisdiction":      doc.jurisdiction,
            "title":             doc.title[:100],
            "char_count":        len(doc.content),
            "chunk_count":       len(chunker.chunk_document(doc)),
        }
        for doc in docs
    ],
}
with open("outputs/ingestion_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("[STARTUP] Saved outputs/ingestion_summary.json")

# ── Start API server ──────────────────────────────────────────
print(f"[STARTUP] Starting API on http://localhost:8000 ...")
print(f"[STARTUP] Swagger UI: http://localhost:8000/docs")
print(f"[STARTUP] Streamlit:  streamlit run app.py  (new terminal)")

import uvicorn
uvicorn.run(
    "src.api.main:app",
    host="0.0.0.0",
    port=8000,
    reload=False,
    log_level="info",
)