"""
scripts/ingest_pdfs.py  — v4 fixed
Version-aware ingestion with disk persistence.
"""

import sys
import os
import json
import time

# ── Fix path FIRST before any project imports ─────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ── Env vars SECOND ───────────────────────────────────────────
os.environ.update({
    "LLM_BACKEND":             "bedrock",
    "AWS_REGION":              "us-east-1",
    "QDRANT_MODE":             "disk",        # ← disk not memory
    "BEDROCK_PRIMARY_MODEL":   "mistral.mistral-7b-instruct-v0:2",
    "BEDROCK_COMPLEX_MODEL":   "mistral.mixtral-8x7b-instruct-v0:1",
    "BEDROCK_EMBEDDING_MODEL": "amazon.titan-embed-text-v2:0",
    "EMBEDDING_DIMENSION":     "1024",
    "QDRANT_COLLECTION":       "regulatory_docs",
})

# ── Project imports THIRD ─────────────────────────────────────
import boto3
import json as _json
from pathlib import Path
from src.llm import bedrock_client as _bc
from src.ingestion.version_registry import get_version, REGULATION_REGISTRY


def sep(n, title): print(f"\n{'='*62}\n  Step {n}: {title}\n{'='*62}")
def ok(m):   print(f"  [OK]   {m}")
def info(m): print(f"         {m}")
def warn(m): print(f"  [WARN] {m}")
def fail(m): print(f"  [FAIL] {m}")


# ── Patch Titan embed ─────────────────────────────────────────
BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name="us-east-1")

def titan_embed(text: str) -> list:
    resp = BEDROCK_CLIENT.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=_json.dumps({"inputText": text[:8000]}),
        contentType="application/json",
        accept="application/json",
    )
    return _json.loads(resp["body"].read())["embedding"]

def _fixed_embed(self, text):
    if not text or not text.strip():
        return [0.0] * 1024
    return titan_embed(text)

def _fixed_init(self):
    self.client     = _bc.get_bedrock_client()
    self.model_id   = "amazon.titan-embed-text-v2:0"
    self.dimensions = 1024

_bc.BedrockEmbedder.__init__ = _fixed_init
_bc.BedrockEmbedder.embed    = _fixed_embed


# ═══════════════════════════════════════════════════════════════
sep(1, "Scan real_docs/ folder")
# ═══════════════════════════════════════════════════════════════
REAL_DOCS_DIR = Path("real_docs")

if not REAL_DOCS_DIR.exists():
    fail("real_docs/ folder not found")
    sys.exit(1)

# Deduplicate — Windows glob finds same file for *.pdf and *.PDF
seen = set()
pdf_files = []
for f in REAL_DOCS_DIR.iterdir():
    if f.suffix.lower() in (".pdf", ".txt") and f.name not in seen:

        seen.add(f.name)
        pdf_files.append(f)

pdf_files.sort(key=lambda f: f.name)

ok(f"Found {len(pdf_files)} unique PDF files:")
for f in pdf_files:
    reg = get_version(f.name)
    reg_info = f"{reg.regulation_id} {reg.version} ({reg.status})" if reg else "NOT IN REGISTRY"
    info(f"  {f.name}  ({f.stat().st_size//1024} KB)  →  {reg_info}")


# ═══════════════════════════════════════════════════════════════
sep(2, "Build metadata from version registry")
# ═══════════════════════════════════════════════════════════════
# Build METADATA_OVERRIDES from registry — single source of truth
METADATA_OVERRIDES = {}
for f in pdf_files:
    reg = get_version(f.name)
    if reg:
        METADATA_OVERRIDES[f.name] = {
            "source":            reg.jurisdiction if reg.jurisdiction != "GLOBAL"
                                 else reg.regulation_id.split("_")[0],
            "regulation_family": reg.regulation_id,
            "jurisdiction":      reg.jurisdiction,
            "version":           reg.version,
            "effective_from":    reg.effective_from,
            "effective_to":      reg.effective_to or "current",
            "source_url":        reg.source_url,
            "status":            reg.status,
            "regulation_id":     reg.regulation_id,
            "is_superseded":     reg.status == "superseded",
        }
        ok(f"  {f.name} → {reg.regulation_id} {reg.version} [{reg.status}]")
    else:
        warn(f"  {f.name} → NOT IN REGISTRY — add to version_registry.py")
        # Fallback detection
        name = f.name.upper()
        if "BASEL" in name or "BIS" in name:
            METADATA_OVERRIDES[f.name] = {
                "source": "BIS", "regulation_family": "BASEL_III",
                "jurisdiction": "GLOBAL", "version": "unknown",
                "status": "active", "regulation_id": "BASEL_III",
                "is_superseded": False,
            }
        elif "RBI" in name or "KYC" in name:
            METADATA_OVERRIDES[f.name] = {
                "source": "RBI", "regulation_family": "RBI_KYC",
                "jurisdiction": "IN", "version": "unknown",
                "status": "active", "regulation_id": "RBI_KYC",
                "is_superseded": False,
            }
        elif "MIFID" in name or "CELEX" in name:
            METADATA_OVERRIDES[f.name] = {
                "source": "ESMA", "regulation_family": "MIFID_II",
                "jurisdiction": "EU", "version": "unknown",
                "status": "active", "regulation_id": "MIFID_II",
                "is_superseded": False,
            }
        elif "FATF" in name:
            METADATA_OVERRIDES[f.name] = {
                "source": "FATF", "regulation_family": "FATF_AML",
                "jurisdiction": "GLOBAL", "version": "unknown",
                "status": "active", "regulation_id": "FATF_AML",
                "is_superseded": False,
            }


# ═══════════════════════════════════════════════════════════════
sep(3, "Parse PDFs → extract text")
# ═══════════════════════════════════════════════════════════════
info("Using pdfminer.six with CID font cleanup")

from src.ingestion.document_loader import DocumentLoader
loader = DocumentLoader()

docs = loader.load_directory(
    str(REAL_DOCS_DIR),
    metadata_overrides=METADATA_OVERRIDES,
)

if not docs:
    fail("No documents loaded — check PDFs are readable")
    sys.exit(1)

ok(f"Parsed {len(docs)} documents:")
for doc in docs:
    meta = METADATA_OVERRIDES.get(doc.doc_id + ".pdf") or \
           METADATA_OVERRIDES.get(doc.doc_id.lower() + ".pdf") or {}
    ok(f"  {doc.doc_id}")
    info(f"    Regulation: {doc.regulation_family}  "
         f"Version: {meta.get('version','?')}  "
         f"Status: {meta.get('status','?')}  "
         f"Chars: {len(doc.content):,}")


# ═══════════════════════════════════════════════════════════════
sep(4, "Chunk documents")
# ═══════════════════════════════════════════════════════════════
info("Hierarchical semantic chunking — headings → size → overlap")

from src.ingestion.chunker import SemanticChunker
chunker = SemanticChunker(max_chunk_tokens=400, overlap_tokens=50)

all_chunks = []
for doc in docs:
    chunks = chunker.chunk_document(doc)
    all_chunks.extend(chunks)
    ok(f"  {doc.doc_id}: {len(chunks)} chunks")
    # Preview first chunk
    if chunks:
        c = chunks[0]
        info(f"    First chunk: [{c.section_number}] {c.section_title}")
        info(f"    Preview: {c.text[:100].strip()}...")

ok(f"Total: {len(all_chunks)} chunks | "
   f"Avg size: {sum(len(c.text) for c in all_chunks)//max(len(all_chunks),1)} chars")


# ═══════════════════════════════════════════════════════════════
sep(5, "Setup Qdrant (disk persistence)")
# ═══════════════════════════════════════════════════════════════
import src.ingestion.embedder as emb_module
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

os.makedirs("./qdrant_storage", exist_ok=True)
emb_module._qdrant_client = QdrantClient(path="./qdrant_storage")

existing_cols = [c.name for c in emb_module._qdrant_client.get_collections().collections]
if "regulatory_docs" in existing_cols:
    ok("Existing Qdrant collection found on disk")
    # Delete and recreate to ensure fresh versioned data
    emb_module._qdrant_client.delete_collection("regulatory_docs")
    info("Deleted old collection for fresh versioned re-ingestion")

emb_module._qdrant_client.create_collection(
    "regulatory_docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
ok("Created fresh Qdrant collection (dim=1024, metric=COSINE)")


# ═══════════════════════════════════════════════════════════════
sep(6, "Embed + store with version metadata")
# ═══════════════════════════════════════════════════════════════
info(f"Embedding {len(all_chunks)} chunks via Titan v2")
info(f"Estimated time: {len(all_chunks) * 200 // 1000}s at 200ms/chunk")

from src.ingestion.embedder import VectorStoreIngester
ingester = VectorStoreIngester()

t0 = time.time()
total_stored = 0

for doc in docs:
    doc_chunks = [c for c in all_chunks if c.doc_id == doc.doc_id]
    if not doc_chunks:
        warn(f"  No chunks for {doc.doc_id} — skipping")
        continue

    # Get version metadata — match by filename
    # Match doc_id to filename more reliably
    reg = None
    for fname in METADATA_OVERRIDES:
        # Remove extension and compare case-insensitively
        fname_clean = fname.upper().replace(".PDF", "")
        doc_clean   = doc.doc_id.upper()
        if fname_clean == doc_clean or doc_clean in fname_clean or fname_clean in doc_clean:
            reg = get_version(fname)
            if reg:
                break

    version_meta = {}
    if reg:
        version_meta = {
            "regulation_id":  reg.regulation_id,
            "version":        reg.version,
            "effective_from": reg.effective_from,
            "effective_to":   reg.effective_to or "current",
            "source_url":     reg.source_url,
            "status":         reg.status,
            "is_superseded":  reg.status == "superseded",
        }

    n = ingester.ingest(doc_chunks, batch_size=20, version_metadata=version_meta)
    total_stored += n
    ok(f"  {doc.doc_id}: {n} chunks "
       f"| version={version_meta.get('version','unknown')} "
       f"| status={version_meta.get('status','unknown')} "
       f"| superseded={version_meta.get('is_superseded', False)}")

elapsed = int(time.time() - t0)
ok(f"Total: {total_stored} chunks stored in {elapsed}s")
ok(f"Qdrant: {ingester.collection_stats()}")


# ═══════════════════════════════════════════════════════════════
sep(7, "Test retrieval with version awareness")
# ═══════════════════════════════════════════════════════════════
from src.retrieval.hybrid_search import HybridSearcher
searcher = HybridSearcher()

test_questions = [
    "What is the minimum CET1 capital ratio under Basel III?",
    "What are KYC requirements for high risk customers?",
    "What is the PSL target for agriculture?",
]

for q in test_questions:
    print(f"\n  Q: {q}")
    results = searcher.search(q, top_k=3)
    for i, r in enumerate(results[:2], 1):
        print(f"  {i}. [{r.metadata.get('regulation_id','?')}] "
              f"v={r.metadata.get('version','?')} "
              f"superseded={r.metadata.get('is_superseded','?')}")
        print(f"     {r.text[:100].strip()}...")


# ═══════════════════════════════════════════════════════════════
sep(8, "Save summary")
# ═══════════════════════════════════════════════════════════════
os.makedirs("outputs", exist_ok=True)

summary = {
    "ingested_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_chunks": total_stored,
    "documents": [
        {
            "doc_id":            doc.doc_id,
            "source":            doc.source,
            "regulation_family": doc.regulation_family,
            "regulation_id":     METADATA_OVERRIDES.get(
                                     next((k for k in METADATA_OVERRIDES
                                           if doc.doc_id.upper() in k.upper()), ""),
                                     {}).get("regulation_id", doc.regulation_family),
            "version":           METADATA_OVERRIDES.get(
                                     next((k for k in METADATA_OVERRIDES
                                           if doc.doc_id.upper() in k.upper()), ""),
                                     {}).get("version", "unknown"),
            "status":            METADATA_OVERRIDES.get(
                                     next((k for k in METADATA_OVERRIDES
                                           if doc.doc_id.upper() in k.upper()), ""),
                                     {}).get("status", "active"),
            "jurisdiction":      doc.jurisdiction,
            "title":             doc.title[:100],
            "char_count":        len(doc.content),
            "chunk_count":       len([c for c in all_chunks if c.doc_id == doc.doc_id]),
        }
        for doc in docs
    ],
}

with open("outputs/ingestion_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

ok("Saved outputs/ingestion_summary.json")

print(f"""
{'='*62}
  Ingestion complete!
  Documents: {len(docs)} | Chunks: {total_stored} | Time: {elapsed}s
{'='*62}

Next steps:
  uvicorn src.api.main:app --reload --port 8000
  streamlit run app.py
""")

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"Working dir: {os.getcwd()}")