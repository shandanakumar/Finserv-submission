"""
scripts/test_local_setup.py

Run this to verify every component of the LOCAL (Ollama) stack.
Mirror of test_bedrock_setup.py — same 8 steps, same output format.

HOW TO RUN:
    cd finserv-compliance
    source venv/bin/activate
    python scripts/test_local_setup.py

PREREQUISITES:
    1. ollama serve              (running in a separate terminal)
    2. ollama pull mistral       (downloaded)
    3. ollama pull nomic-embed-text (downloaded)
    4. docker compose up -d qdrant postgres redis
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force local backend regardless of .env
os.environ.setdefault("LLM_BACKEND", "ollama")


def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def ok(msg):   print(f"  [OK]   {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"         {msg}")


# ─── Step 1: Ollama server running ────────────────────────────
sep("Step 1 — Ollama server reachable")
info("""
Ollama must be running as a background server.
  macOS/Linux: ollama serve          (new terminal, leave it open)
  Windows:     Ollama app in system tray, or: ollama serve
  Check port:  curl http://localhost:11434
""")

try:
    import requests
    resp = requests.get("http://localhost:11434", timeout=5)
    ok(f"Ollama server responding at http://localhost:11434")
except Exception as e:
    fail(f"Cannot reach Ollama: {e}")
    info("Start Ollama with:  ollama serve")
    info("Then re-run this script.")
    sys.exit(1)


# ─── Step 2: Models pulled ────────────────────────────────────
sep("Step 2 — Models downloaded (ollama list)")
info("""
Required models:
  mistral           4.1 GB   fast Q&A and routing
  nomic-embed-text  274 MB   embeddings
Optional (needs 32 GB RAM):
  mixtral           26 GB    complex reasoning

Pull missing ones with:
  ollama pull mistral
  ollama pull nomic-embed-text
  ollama pull mixtral       # skip if RAM < 32 GB
""")

try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = resp.json().get("models", [])
    model_names = [m["name"].split(":")[0] for m in models]

    ok(f"Models on disk: {model_names}")

    for required in ["mistral", "nomic-embed-text"]:
        if required in model_names:
            size = next(
                (f"{m['size'] / 1e9:.1f} GB" for m in models if m["name"].startswith(required)),
                "?"
            )
            ok(f"  {required:25s}  {size}")
        else:
            fail(f"  {required} not found — run: ollama pull {required}")

    if "mixtral" in model_names:
        ok(f"  {'mixtral':25s}  (complex model available)")
    else:
        info("  mixtral not pulled — complex tasks will fall back to mistral")

except Exception as e:
    fail(str(e))


# ─── Step 3: Mistral 7B generate ──────────────────────────────
sep("Step 3 — Mistral 7B text generation")
info("""
Sends a single compliance question to Mistral 7B.
Expected latency:
  GPU (NVIDIA):  ~1–3 seconds
  Apple Silicon: ~3–8 seconds
  CPU only:      ~30–120 seconds  (slow but works)
""")

try:
    from src.llm.ollama_client import OllamaLLMClient

    llm = OllamaLLMClient(model_id="mistral")
    t0 = time.time()
    response = llm.invoke(
        system_prompt="You are a concise compliance assistant. Answer in one sentence.",
        user_message="What is the minimum CET1 capital ratio under Basel III?",
    )
    ok(f"Latency:  {response.latency_ms}ms")
    ok(f"Tokens:   {response.input_tokens} in / {response.output_tokens} out")
    ok(f"Response: {response.text[:150]}")

except Exception as e:
    fail(str(e))


# ─── Step 4: Mixtral (if available) ──────────────────────────
sep("Step 4 — Mixtral 8×7B (skip if not pulled)")

try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    pulled = [m["name"].split(":")[0] for m in resp.json().get("models", [])]

    if "mixtral" not in pulled:
        info("Mixtral not pulled — skipping.")
        info("To pull (needs 26 GB disk + 32 GB RAM):  ollama pull mixtral")
    else:
        llm_mx = OllamaLLMClient(model_id="mixtral")
        t0 = time.time()
        response = llm_mx.invoke(
            system_prompt="You are a FinServ compliance expert.",
            user_message=(
                "Explain the difference between LCR and NSFR under Basel III "
                "and which ratio regulators check more frequently."
            ),
        )
        ok(f"Latency:  {response.latency_ms}ms")
        ok(f"Response: {response.text[:150]}")

except Exception as e:
    fail(str(e))


# ─── Step 5: nomic-embed-text embeddings ──────────────────────
sep("Step 5 — nomic-embed-text embeddings (768 dims)")
info("""
nomic-embed-text produces 768-dimensional vectors.
EMBEDDING_DIMENSION in settings.py must equal 768 for local.
(For Bedrock/Titan it's 1536 — they're different.)
""")

try:
    from src.llm.ollama_client import OllamaEmbedder
    from config.settings import settings

    embedder = OllamaEmbedder()
    t0 = time.time()
    vector = embedder.embed("What is the Basel III CET1 minimum capital ratio?")
    latency_ms = int((time.time() - t0) * 1000)

    ok(f"Dimensions: {len(vector)}  (expected: {settings.EMBEDDING_DIMENSION})")
    ok(f"Latency:    {latency_ms}ms")
    ok(f"Sample:     [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}, ...]")

    if len(vector) != settings.EMBEDDING_DIMENSION:
        fail(f"Mismatch: got {len(vector)}, EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION}")
        info("Fix: set EMBEDDING_DIMENSION=768 in your .env file")

except Exception as e:
    fail(str(e))


# ─── Step 6: Qdrant health ────────────────────────────────────
sep("Step 6 — Qdrant Docker health")
info("""
Qdrant must be running in Docker.
  Start:  docker compose up -d qdrant
  Check:  docker compose ps
  UI:     http://localhost:6333/dashboard
""")

try:
    from qdrant_client import QdrantClient
    from config.settings import settings

    qdrant = QdrantClient(url=settings.QDRANT_URL, timeout=10)
    collections = qdrant.get_collections().collections
    ok(f"Qdrant at {settings.QDRANT_URL} — healthy")
    ok(f"Existing collections: {[c.name for c in collections] or '(none yet)'}")

except Exception as e:
    fail(str(e))
    info("Run: docker compose up -d qdrant")


# ─── Step 7: Embed → store → retrieve ────────────────────────
sep("Step 7 — Integration: embed → Qdrant → retrieve")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from src.llm.ollama_client import OllamaEmbedder
    from config.settings import settings

    TEST_COLL = "test_local_integration"
    qdrant = QdrantClient(url=settings.QDRANT_URL)
    embedder = OllamaEmbedder()

    # Clean slate
    existing = [c.name for c in qdrant.get_collections().collections]
    if TEST_COLL in existing:
        qdrant.delete_collection(TEST_COLL)

    qdrant.create_collection(
        collection_name=TEST_COLL,
        vectors_config=VectorParams(
            size=settings.EMBEDDING_DIMENSION,  # 768
            distance=Distance.COSINE,
        ),
    )
    ok("Test collection created (dim=768, metric=COSINE)")

    # Insert a known sentence
    text = "Banks must maintain a minimum CET1 capital ratio of 4.5% of risk-weighted assets."
    vector = embedder.embed(text)
    qdrant.upsert(
        collection_name=TEST_COLL,
        points=[PointStruct(id=1, vector=vector, payload={"text": text})],
        wait=True,
    )
    ok(f"Inserted 1 point (dim={len(vector)})")

    # Retrieve with similar query
    query_vec = embedder.embed("What is the minimum CET1 requirement?")
    results = qdrant.search(collection_name=TEST_COLL, query_vector=query_vec, limit=1)

    if results:
        ok(f"Retrieved: '{results[0].payload['text'][:80]}...'")
        ok(f"Score:     {results[0].score:.4f}")
    else:
        fail("No results — check Qdrant and embedding dimension")

    qdrant.delete_collection(TEST_COLL)
    ok("Test collection cleaned up")

except Exception as e:
    fail(str(e))
    import traceback
    traceback.print_exc()


# ─── Step 8: Full pipeline ────────────────────────────────────
sep("Step 8 — Full pipeline: ingest docs → compliance query")
info("""
This ingests all 5 sample regulatory docs and runs a real compliance query.
Will take 1–3 minutes (embedding ~50 chunks via nomic-embed-text).
""")

try:
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.embedder import VectorStoreIngester
    from src.agent.compliance_agent import FallbackComplianceAgent
    from config.settings import settings

    loader = DocumentLoader()
    chunker = SemanticChunker(max_tokens=400, overlap_pct=0.1)
    ingester = VectorStoreIngester()
    ingester.ensure_collection_exists()

    docs = loader.load_directory(settings.DOCS_DIR)
    ok(f"Loaded {len(docs)} documents from {settings.DOCS_DIR}/")

    total = 0
    for doc in docs:
        chunks = chunker.chunk(doc.text, doc.doc_id, doc.metadata)
        n = ingester.ingest(chunks)
        total += n
        ok(f"  {doc.metadata.get('source_document', doc.doc_id)}: {len(chunks)} chunks")

    ok(f"Total ingested: {total} chunks")
    ok(f"Collection stats: {ingester.collection_stats()}")

    print("\n  Running compliance query (Mistral 7B)...")
    agent = FallbackComplianceAgent()
    result = agent.run(
        "What is the minimum CET1 ratio under Basel III and "
        "what triggers automatic distribution restrictions?"
    )
    print(f"\n  ANSWER:     {result.get('answer', 'ERROR')[:300]}")
    print(f"  CONFIDENCE: {result.get('confidence')}")
    print(f"  REGULATIONS:{result.get('applicable_regulations')}")
    ok("Full pipeline completed")

except Exception as e:
    fail(str(e))
    import traceback
    traceback.print_exc()


# ─── Summary ──────────────────────────────────────────────────
sep("Done — next steps")
print("""
All green? Start the API server:

  uvicorn src.api.main:app --reload --port 8000

Test with curl:

  curl -X POST http://localhost:8000/query \\
    -H "X-API-Key: dev-key-1234" \\
    -H "Content-Type: application/json" \\
    -d '{"query": "What is the minimum LCR under Basel III?"}'

Test transaction screening:

  curl -X POST http://localhost:8000/screen-transaction \\
    -H "X-API-Key: dev-key-1234" \\
    -H "Content-Type: application/json" \\
    -d '{"transaction_id": "TXN-XBORDER-001"}'

Switch to Bedrock when ready:
  1. In .env set:  LLM_BACKEND=bedrock
  2. Add AWS credentials and model IDs
  3. Set:          EMBEDDING_DIMENSION=1536
  4. Delete Qdrant collection (vector size changes):
       curl -X DELETE http://localhost:6333/collections/regulatory_docs
  5. Re-run ingestion:  python scripts/test_bedrock_setup.py
""")
