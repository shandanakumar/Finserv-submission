"""
scripts/test_windows_local.py  —  v3 (all fixes applied)

Run from project root:
    venv\Scripts\activate
    python scripts\test_windows_local.py
"""

import sys, os, json, time, importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Set env before any project imports
os.environ.update({
    "LLM_BACKEND":             "bedrock",
    "AWS_REGION":              "us-east-1",
    "QDRANT_MODE":             "memory",
    "BEDROCK_PRIMARY_MODEL":   "mistral.mistral-7b-instruct-v0:2",
    "BEDROCK_COMPLEX_MODEL":   "mistral.mixtral-8x7b-instruct-v0:1",
    "BEDROCK_EMBEDDING_MODEL": "amazon.titan-embed-text-v2:0",
    "EMBEDDING_DIMENSION":     "1024",
    "QDRANT_COLLECTION":       "regulatory_docs",
    "LLM_TEMPERATURE":         "0.1",
    "LLM_MAX_TOKENS":          "2048",
    "LLM_TIMEOUT_SECONDS":     "60",
    "DOCS_DIR":                "sample_docs",
})


def sep(n, title):
    print(f"\n{'='*60}\n  Step {n}: {title}\n{'='*60}")
def ok(m):   print(f"  [OK]   {m}")
def fail(m): print(f"  [FAIL] {m}")
def info(m): print(f"         {m}")

import boto3
BEDROCK = boto3.client("bedrock-runtime", region_name="us-east-1")

# ── helpers ───────────────────────────────────────────────────
def titan_embed(text: str) -> list:
    """Embed using whichever Titan model works on this account."""
    for model_id in [
        "amazon.titan-embed-text-v2:0",
        "amazon.titan-embed-text-v1:2",
        "amazon.titan-embed-text-v1:0",
    ]:
        try:
            resp = BEDROCK.invoke_model(
                modelId=model_id,
                body=json.dumps({"inputText": text[:8000]}),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            if "embedding" in result:
                os.environ["TITAN_MODEL_ID"]       = model_id
                os.environ["EMBEDDING_DIMENSION"]  = str(len(result["embedding"]))
                return result["embedding"]
        except Exception:
            continue
    raise RuntimeError("No Titan embedding model accessible on this account")

def qdrant_search(client, collection, vector, limit=3):
    """Works with both old (.search) and new (.query_points) qdrant-client."""
    try:
        r = client.query_points(
            collection_name=collection, query=vector,
            limit=limit, with_payload=True,
        )
        return r.points
    except AttributeError:
        return client.search(
            collection_name=collection, query_vector=vector,
            limit=limit, with_payload=True,
        )

# ─────────────────────────────────────────────────────────────
sep(1, "AWS credentials")
try:
    identity = boto3.client("sts", region_name="us-east-1").get_caller_identity()
    ok(f"Account: {identity['Account']}")
    ok(f"Role:    {identity['Arn'].split('/')[-1]}")
except Exception as e:
    fail(str(e)); sys.exit(1)

# ─────────────────────────────────────────────────────────────
sep(2, "Mistral 7B — single question")
try:
    t0 = time.time()
    resp = BEDROCK.invoke_model(
        modelId="mistral.mistral-7b-instruct-v0:2",
        body=json.dumps({
            "prompt": "<s>[INST] In one sentence: what is the Basel III CET1 minimum ratio? [/INST]",
            "max_tokens": 100, "temperature": 0.1,
        }),
        contentType="application/json", accept="application/json",
    )
    text = json.loads(resp["body"].read())["outputs"][0]["text"].strip()
    ok(f"Latency: {int((time.time()-t0)*1000)}ms")
    ok(f"Answer:  {text[:120]}")
except Exception as e:
    fail(str(e))

# ─────────────────────────────────────────────────────────────
sep(3, "Titan Embeddings — auto-detect model")
info("Tries titan-embed-text-v2:0, v1:2, v1:0 until one works")
try:
    t0  = time.time()
    vec = titan_embed("What is the Basel III CET1 minimum ratio?")
    ok(f"Model:      {os.environ['TITAN_MODEL_ID']}")
    ok(f"Dimensions: {len(vec)}  (stored as EMBEDDING_DIMENSION)")
    ok(f"Latency:    {int((time.time()-t0)*1000)}ms")
    ok(f"Sample:     [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f} ...]")
    EMBED_DIM = len(vec)
except Exception as e:
    fail(str(e)); sys.exit(1)

# ─────────────────────────────────────────────────────────────
sep(4, "In-memory Qdrant — store and retrieve")
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    qd = QdrantClient(":memory:")
    qd.create_collection("test",
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))
    ok(f"Collection created (dim={EMBED_DIM})")

    v1 = titan_embed("Banks must hold 4.5% CET1 capital under Basel III.")
    qd.upsert("test", points=[
        PointStruct(id=1, vector=v1, payload={"text": "CET1 = 4.5% minimum"})
    ], wait=True)
    ok("Point inserted")

    v2      = titan_embed("What is the minimum CET1 requirement?")
    results = qdrant_search(qd, "test", v2, limit=1)
    ok(f"Retrieved: '{results[0].payload['text']}'  score={results[0].score:.4f}")

except Exception as e:
    fail(str(e)); import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
sep(5, "Patch BedrockEmbedder with working model + format")
info(f"Using Titan model: {os.environ.get('TITAN_MODEL_ID','unknown')}")
info(f"Using dimension:   {os.environ.get('EMBEDDING_DIMENSION','unknown')}")
try:
    from src.llm import bedrock_client as bc

    _titan_id  = os.environ["TITAN_MODEL_ID"]
    _titan_dim = int(os.environ["EMBEDDING_DIMENSION"])

    def _fixed_embed(self, text: str) -> list:
        if not text or not text.strip():
            return [0.0] * _titan_dim
        resp = self.client.invoke_model(
            modelId=_titan_id,
            body=json.dumps({"inputText": text[:8000]}),
            contentType="application/json", accept="application/json",
        )
        return json.loads(resp["body"].read())["embedding"]

    bc.BedrockEmbedder.embed = _fixed_embed
    # Also override model_id and dimensions on any future instances
    _orig_init = bc.BedrockEmbedder.__init__
    def _fixed_init(self):
        from src.llm.bedrock_client import get_bedrock_client
        self.client     = get_bedrock_client()
        self.model_id   = _titan_id
        self.dimensions = _titan_dim
    bc.BedrockEmbedder.__init__ = _fixed_init

    # test it
    from src.llm.bedrock_client import BedrockEmbedder
    embedder = BedrockEmbedder()
    v = embedder.embed("compliance test")
    ok(f"BedrockEmbedder works — dim={len(v)}")

except Exception as e:
    fail(str(e)); import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
sep(6, "Ingest 5 regulatory docs")
info("Basel III · MiFID II · RBI KYC · RBI PSL · FATF AML")
info(f"Using Titan ({os.environ.get('TITAN_MODEL_ID')}, {os.environ.get('EMBEDDING_DIMENSION')} dims)")
info("Takes 3-5 minutes — ~50 chunks × 200ms per Titan call")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    import src.ingestion.embedder as emb_module

    # Give all modules a fresh in-memory client with correct dimension
    emb_module._qdrant_client = QdrantClient(":memory:")
    emb_module._qdrant_client.create_collection(
        "regulatory_docs",
        vectors_config=VectorParams(
            size=int(os.environ["EMBEDDING_DIMENSION"]),
            distance=Distance.COSINE,
        ),
    )

    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.embedder import VectorStoreIngester

    loader   = DocumentLoader()
    chunker  = SemanticChunker(max_chunk_tokens=400, overlap_tokens=50)
    ingester = VectorStoreIngester()

    docs = loader.load_directory("sample_docs")
    ok(f"Loaded {len(docs)} documents")

    total = 0
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        n      = ingester.ingest(chunks)
        total += n
        ok(f"  {doc.doc_id}: {len(chunks)} chunks → {n} stored")

    stats = ingester.collection_stats()
    ok(f"Total: {total} chunks  |  Collection: {stats}")

except Exception as e:
    fail(str(e)); import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
sep(7, "Hybrid search — retrieve relevant chunks")
try:
    from src.retrieval.hybrid_search import HybridSearcher

    searcher = HybridSearcher()
    results  = searcher.search(
        "What is the minimum CET1 capital ratio under Basel III?",
        top_k=3,
    )
    ok(f"Retrieved {len(results)} chunks")
    for i, r in enumerate(results, 1):
        reg = r.metadata.get("regulation_family", "?")
        ok(f"  {i}. [{reg}] {r.text[:80].strip()}...")

except Exception as e:
    fail(str(e)); import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
sep(8, "Full compliance agent — 3 questions")
info("Router → RAG retrieval → Mistral 7B synthesis → Reflection → Answer")
try:
    from src.agent.compliance_agent import FallbackComplianceAgent
    agent = FallbackComplianceAgent()

    questions = [
        "What is the minimum CET1 capital ratio under Basel III?",
        "What are KYC requirements for HIGH risk customers under RBI?",
        "When is a suitability assessment mandatory under MiFID II?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        t0     = time.time()
        result = agent.run(q)
        ms     = int((time.time() - t0) * 1000)
        answer = result.get("answer", result.get("draft_response", ""))
        print(f"  A: {str(answer)[:250]}")
        print(f"     Confidence={result.get('confidence')} | "
              f"Regs={result.get('applicable_regulations')} | "
              f"Time={ms}ms")

    ok("All 3 questions answered — pipeline complete!")

except Exception as e:
    fail(str(e)); import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}\n  Complete! Start the API server:\n{'='*60}")
print("""
  uvicorn src.api.main:app --reload --port 8000

  Then open:  http://localhost:8000/docs
""")
