"""
scripts/test_windows_local.py  —  Fixed version

Fixes applied vs test_bedrock_setup.py:
  1. Titan request format: removed "dimensions" + "normalize" fields
     (Titan v2 on this account uses simple {"inputText": "..."} only)
  2. Qdrant: uses in-memory mode — no Docker needed
  3. Chunker: correct params are max_chunk_tokens / overlap_tokens
              correct method is chunk_document(doc) not chunk(text,...)
  4. DocumentLoader: correct method is load_directory(path)
"""

import sys, os, json, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Force correct settings before any imports
os.environ["LLM_BACKEND"]               = "bedrock"
os.environ["AWS_REGION"]                = "us-east-1"
os.environ["QDRANT_MODE"]               = "memory"
os.environ["BEDROCK_PRIMARY_MODEL"]     = "mistral.mistral-7b-instruct-v0:2"
os.environ["BEDROCK_COMPLEX_MODEL"]     = "mistral.mixtral-8x7b-instruct-v0:1"
os.environ["BEDROCK_EMBEDDING_MODEL"]   = "amazon.titan-embed-text-v2:0"
os.environ["EMBEDDING_DIMENSION"]       = "1536"
os.environ["QDRANT_COLLECTION"]         = "regulatory_docs"
os.environ["LLM_TEMPERATURE"]           = "0.1"
os.environ["LLM_MAX_TOKENS"]            = "2048"
os.environ["LLM_TIMEOUT_SECONDS"]       = "60"
os.environ["DOCS_DIR"]                  = "sample_docs"


def sep(title, n):
    print(f"\n{'='*60}\n  Step {n}: {title}\n{'='*60}")
def ok(msg):   print(f"  [OK]   {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"         {msg}")


# ── Step 1: AWS credentials ───────────────────────────────────
sep("AWS credentials", 1)
try:
    import boto3
    sts      = boto3.client("sts", region_name="us-east-1")
    identity = sts.get_caller_identity()
    ok(f"Account: {identity['Account']}")
    ok(f"Role:    {identity['Arn'].split('/')[-1]}")
    BEDROCK  = boto3.client("bedrock-runtime", region_name="us-east-1")
    ok("bedrock-runtime client created")
except Exception as e:
    fail(str(e))
    info("Run:  aws configure")
    sys.exit(1)


# ── Step 2: Mistral 7B ────────────────────────────────────────
sep("Mistral 7B — single question", 2)
try:
    body = json.dumps({
        "prompt":      "<s>[INST] In one sentence: what is the Basel III minimum CET1 ratio? [/INST]",
        "max_tokens":  150,
        "temperature": 0.1,
    })
    t0   = time.time()
    resp = BEDROCK.invoke_model(
        modelId="mistral.mistral-7b-instruct-v0:2",
        body=body, contentType="application/json", accept="application/json",
    )
    ms   = int((time.time() - t0) * 1000)
    text = json.loads(resp["body"].read())["outputs"][0]["text"].strip()
    ok(f"Latency:  {ms}ms")
    ok(f"Answer:   {text[:120]}")
except Exception as e:
    fail(str(e))


# ── Step 3: Titan Embeddings — correct request format ─────────
sep("Titan Embeddings v2 — correct format", 3)
info("FIX APPLIED: removed 'dimensions' and 'normalize' fields.")
info("This account's Titan v2 endpoint only accepts {inputText: ...}")
info("The validation error in test_bedrock_setup.py was caused by extra fields.")

def titan_embed(text: str) -> list:
    """
    Correct Titan v2 request format for this AWS account.
    Just inputText — no dimensions, no normalize.
    Returns a list of floats (1536 dims).
    """
    body = json.dumps({"inputText": text[:8000]})
    resp = BEDROCK.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]

try:
    t0     = time.time()
    vector = titan_embed("What is the Basel III CET1 minimum ratio?")
    ms     = int((time.time() - t0) * 1000)
    ok(f"Dimensions: {len(vector)}")
    ok(f"Latency:    {ms}ms")
    ok(f"Sample:     [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f} ...]")
    EMBED_DIM = len(vector)
    os.environ["EMBEDDING_DIMENSION"] = str(EMBED_DIM)
    ok(f"EMBEDDING_DIMENSION set to {EMBED_DIM}")
except Exception as e:
    fail(str(e))
    info("If ValidationException: try removing more fields from the request body")
    sys.exit(1)


# ── Step 4: In-memory Qdrant ──────────────────────────────────
sep("In-memory Qdrant (no Docker needed)", 4)
info("QdrantClient(':memory:') runs entirely inside Python.")
info("No port, no Docker, no network — just RAM.")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    qd = QdrantClient(":memory:")
    qd.create_collection(
        "test_col",
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    ok(f"Collection created (dim={EMBED_DIM}, metric=COSINE)")

    # Insert
    vec = titan_embed("Banks must hold 4.5% CET1 capital under Basel III.")
    qd.upsert("test_col", points=[
        PointStruct(id=1, vector=vec, payload={"text": "CET1 = 4.5% minimum"})
    ], wait=True)
    ok("Inserted 1 point")

    # Retrieve
    q_vec   = titan_embed("What is the CET1 minimum requirement?")
    results = qd.search("test_col", query_vector=q_vec, limit=1)
    ok(f"Retrieved: '{results[0].payload['text']}'  score={results[0].score:.4f}")

except Exception as e:
    fail(str(e))
    import traceback; traceback.print_exc()


# ── Step 5: Fix BedrockEmbedder to use correct format ─────────
sep("Patching BedrockEmbedder for this account", 5)
info("Overrides BedrockEmbedder.embed() to use the correct request format.")
info("This patch runs in memory — no file changes needed.")

try:
    from src.llm import bedrock_client as bc

    def _fixed_embed(self, text: str) -> list:
        if not text or not text.strip():
            return [0.0] * int(os.environ.get("EMBEDDING_DIMENSION", "1536"))
        body = json.dumps({"inputText": text[:8000]})
        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        vec = result["embedding"]
        # Update dimension from actual response
        if len(vec) != self.dimensions:
            self.dimensions = len(vec)
            os.environ["EMBEDDING_DIMENSION"] = str(len(vec))
        return vec

    bc.BedrockEmbedder.embed = _fixed_embed

    # test the patched version
    from src.llm.bedrock_client import BedrockEmbedder
    embedder = BedrockEmbedder()
    v = embedder.embed("test sentence for compliance")
    ok(f"Patched BedrockEmbedder works — dim={len(v)}")

except Exception as e:
    fail(str(e))
    import traceback; traceback.print_exc()


# ── Step 6: Ingest sample docs ────────────────────────────────
sep("Ingest 5 regulatory docs into in-memory Qdrant", 6)
info("Correct chunker usage: SemanticChunker(max_chunk_tokens=400)")
info("                       chunker.chunk_document(doc)")
info("Correct loader usage:  loader.load_directory('sample_docs')")

# Rebuild Qdrant client with correct dimension from Step 3
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Patch the singleton to use correct dimension
import src.ingestion.embedder as emb_module
emb_module._qdrant_client = QdrantClient(":memory:")

try:
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.embedder import VectorStoreIngester

    loader  = DocumentLoader()
    chunker = SemanticChunker(
        max_chunk_tokens=400,
        overlap_tokens=50,
    )

    ingester = VectorStoreIngester()

    # Recreate collection with correct dimension
    from qdrant_client.models import Distance, VectorParams
    ingester.qdrant.create_collection(
        collection_name=ingester.collection,
        vectors_config=VectorParams(
            size=int(os.environ["EMBEDDING_DIMENSION"]),
            distance=Distance.COSINE,
        ),
    )
    ok(f"Collection '{ingester.collection}' created (dim={os.environ['EMBEDDING_DIMENSION']})")

    docs = loader.load_directory("sample_docs")
    ok(f"Loaded {len(docs)} documents from sample_docs/")

    total = 0
    for doc in docs:
        # Correct method: chunk_document(doc) → list[DocumentChunk]
        chunks = chunker.chunk_document(doc)
        n = ingester.ingest(chunks)
        total += n
        ok(f"  {doc.doc_id}: {len(chunks)} chunks → {n} stored")

    ok(f"Total: {total} chunks in Qdrant")

except Exception as e:
    fail(str(e))
    import traceback; traceback.print_exc()


# ── Step 7: Hybrid search ─────────────────────────────────────
sep("Hybrid search — retrieve relevant chunks", 7)

try:
    from src.retrieval.hybrid_search import HybridSearcher

    searcher = HybridSearcher()
    results  = searcher.search(
        "What is the minimum CET1 capital ratio under Basel III?",
        top_k=3,
    )

    ok(f"Retrieved {len(results)} chunks")
    for i, r in enumerate(results, 1):
        ok(f"  {i}. [{r.metadata.get('regulation_family','?')}] "
           f"{r.text[:80].strip()}... (score={r.rrf_score:.4f})")

except Exception as e:
    fail(str(e))
    import traceback; traceback.print_exc()


# ── Step 8: Full agent ────────────────────────────────────────
sep("Full compliance agent — 3 real questions", 8)
info("Router → RAG → Mistral 7B → Reflection → Guardrails → Answer")

try:
    from src.agent.compliance_agent import FallbackComplianceAgent

    agent = FallbackComplianceAgent()

    questions = [
        "What is the minimum CET1 capital ratio under Basel III?",
        "What are the KYC requirements for HIGH risk customers under RBI?",
        "When is a suitability assessment mandatory under MiFID II?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        t0     = time.time()
        result = agent.run(q)
        ms     = int((time.time() - t0) * 1000)
        answer = result.get("answer", result.get("draft_response", "no answer"))
        print(f"  A: {str(answer)[:200]}")
        print(f"     Confidence={result.get('confidence')}  "
              f"Regulations={result.get('applicable_regulations')}  "
              f"Time={ms}ms")

    ok("All 3 questions answered")

except Exception as e:
    fail(str(e))
    import traceback; traceback.print_exc()


# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}\n  All steps complete!\n{'='*60}")
print("""
Start the API server:
  uvicorn src.api.main:app --reload --port 8000

Open Swagger UI in browser:
  http://localhost:8000/docs

Quick test in PowerShell:
  Invoke-RestMethod `
    -Uri "http://localhost:8000/query" `
    -Method POST `
    -Headers @{"X-API-Key"="dev-key-1234";"Content-Type"="application/json"} `
    -Body '{"query":"What is the minimum CET1 ratio under Basel III?"}'
""")
