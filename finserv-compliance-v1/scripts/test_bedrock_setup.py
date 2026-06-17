"""
scripts/test_bedrock_setup.py

Run this FIRST before touching anything else.
Each test is independent — comment out ones that pass to move faster.

HOW TO RUN:
    cd finserv-compliance
    python scripts/test_bedrock_setup.py

WHAT EACH STEP TESTS:
    Step 1 — AWS credentials are configured and reachable
    Step 2 — Bedrock service is accessible in your region
    Step 3 — Mistral 7B responds to a single question
    Step 4 — Mixtral 8×7B responds to a single question
    Step 5 — Titan embeddings produce a 1536-dim vector
    Step 6 — Qdrant (Docker) is running and accepting connections
    Step 7 — End-to-end: embed a text → store in Qdrant → retrieve it
    Step 8 — Full compliance agent: ingest docs → ask a question
"""

import json
import sys
import time
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg):
    print(f"  [OK]  {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


# ─── Step 1: AWS credentials ──────────────────────────────────────────────────
separator("Step 1 — AWS credentials")
print("""
WHAT THIS CHECKS:
  boto3 can find your AWS credentials via one of:
  (a) AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY env vars
  (b) ~/.aws/credentials file (from `aws configure`)
  (c) IAM role (if running on EC2/ECS)

IF THIS FAILS:
  Run:  aws configure
  Enter your Access Key ID, Secret Access Key, region (us-east-1), format (json)
  Or set env vars:
    export AWS_ACCESS_KEY_ID=AKIA...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_REGION=us-east-1
""")

try:
    import boto3
    sts = boto3.client("sts", region_name=os.getenv("AWS_REGION", "us-east-1"))
    identity = sts.get_caller_identity()
    ok(f"Account:  {identity['Account']}")
    ok(f"ARN:      {identity['Arn']}")
    ok(f"Region:   {os.getenv('AWS_REGION', 'us-east-1')}")
except Exception as e:
    fail(str(e))
    print("\n  Cannot continue without valid credentials. Fix Step 1 first.")
    sys.exit(1)


# ─── Step 2: Bedrock service access ──────────────────────────────────────────
separator("Step 2 — Bedrock service accessible")
print("""
WHAT THIS CHECKS:
  The bedrock control-plane (not runtime) can list available models.
  This confirms your IAM user/role has bedrock:ListFoundationModels permission.

IF THIS FAILS with AccessDeniedException:
  Attach this IAM policy to your user/role:
  {
    "Effect": "Allow",
    "Action": ["bedrock:ListFoundationModels", "bedrock:InvokeModel"],
    "Resource": "*"
  }
""")

try:
    bedrock = boto3.client("bedrock", region_name=os.getenv("AWS_REGION", "us-east-1"))
    models = bedrock.list_foundation_models(byProvider="Mistral AI")
    mistral_models = [m["modelId"] for m in models["modelSummaries"]]
    ok(f"Bedrock reachable. Mistral models visible: {len(mistral_models)}")
    for m in mistral_models:
        print(f"         {m}")
except Exception as e:
    fail(str(e))


# ─── Step 3: Mistral 7B invoke ────────────────────────────────────────────────
separator("Step 3 — Mistral 7B Instruct v0.2")
print("""
WHAT THIS CHECKS:
  You have Model access enabled for Mistral 7B in the Bedrock console.

IF THIS FAILS with AccessDeniedException:
  Go to: AWS Console → Bedrock → Model access (left sidebar)
  Find "Mistral 7B Instruct" → click "Manage model access" → enable it
  Takes 1-2 minutes to activate.
""")

try:
    from src.llm.bedrock_client import BedrockLLMClient
    from config.settings import settings

    llm = BedrockLLMClient(model_id=settings.LLM_PRIMARY_MODEL)
    t0 = time.time()
    response = llm.invoke(
        system_prompt="You are a concise assistant. Answer in one sentence.",
        user_message="What is the minimum CET1 capital ratio under Basel III?",
    )
    latency_ms = int((time.time() - t0) * 1000)
    ok(f"Model:    {settings.LLM_PRIMARY_MODEL}")
    ok(f"Latency:  {latency_ms}ms")
    ok(f"Response: {response.text[:120]}...")
    ok(f"Est cost: ${response.estimated_cost_usd:.6f}")
except Exception as e:
    fail(str(e))


# ─── Step 4: Mixtral 8×7B invoke ─────────────────────────────────────────────
separator("Step 4 — Mixtral 8×7B Instruct v0.1")
print("""
WHAT THIS CHECKS:
  Same as Step 3 but for Mixtral. Also needs model access enabled.

EXPECTED DIFFERENCE FROM MISTRAL 7B:
  Mixtral is slower (800ms–2s) but produces more structured answers.
  For compliance reports and multi-step reasoning this is worth it.
""")

try:
    llm_complex = BedrockLLMClient(model_id=settings.LLM_COMPLEX_MODEL)
    t0 = time.time()
    response = llm_complex.invoke(
        system_prompt="You are a FinServ compliance expert. Answer precisely.",
        user_message=(
            "Compare the large exposure limits under Basel III "
            "and explain what happens when a bank breaches the 25% Tier 1 limit."
        ),
    )
    latency_ms = int((time.time() - t0) * 1000)
    ok(f"Model:    {settings.LLM_COMPLEX_MODEL}")
    ok(f"Latency:  {latency_ms}ms")
    ok(f"Response: {response.text[:120]}...")
except Exception as e:
    fail(str(e))


# ─── Step 5: Titan embeddings ─────────────────────────────────────────────────
separator("Step 5 — Amazon Titan Embeddings v2")
print("""
WHAT THIS CHECKS:
  Titan Text Embeddings v2 produces a 1536-dimensional vector.
  This model is available by default in Bedrock — no access request needed.

WHY 1536 DIMENSIONS:
  More dimensions = richer semantic space = better retrieval accuracy.
  The cost is slightly larger Qdrant storage (1536 * 4 bytes = 6KB per vector).
  For 100K regulatory chunks that's ~600MB — well within Qdrant's capacity.
""")

try:
    from src.llm.bedrock_client import BedrockEmbedder

    embedder = BedrockEmbedder()
    t0 = time.time()
    vector = embedder.embed("What is the Basel III CET1 minimum capital ratio?")
    latency_ms = int((time.time() - t0) * 1000)

    ok(f"Dimensions: {len(vector)} (expected: {settings.EMBEDDING_DIMENSION})")
    ok(f"Latency:    {latency_ms}ms")
    ok(f"Sample:     [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}, ...]")

    # Confirm the vector is normalised (magnitude ≈ 1.0 when normalize=True)
    magnitude = sum(x**2 for x in vector) ** 0.5
    ok(f"Magnitude:  {magnitude:.4f} (should be ≈ 1.0 — confirms normalize=True)")

    if len(vector) != settings.EMBEDDING_DIMENSION:
        fail(f"Dimension mismatch! Got {len(vector)}, expected {settings.EMBEDDING_DIMENSION}")
        fail("Update EMBEDDING_DIMENSION in settings.py to match Titan output")
except Exception as e:
    fail(str(e))


# ─── Step 6: Qdrant health ────────────────────────────────────────────────────
separator("Step 6 — Qdrant (Docker) health check")
print("""
WHAT THIS CHECKS:
  Qdrant is running in Docker and the REST API responds.

IF THIS FAILS:
  Run:  docker compose up -d qdrant
  Wait 10 seconds, then try again.
  Check: docker compose ps   (should show qdrant as "healthy")
  Logs:  docker compose logs qdrant
""")

try:
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(url=settings.QDRANT_URL, timeout=10)
    collections = qdrant.get_collections().collections
    ok(f"Qdrant reachable at {settings.QDRANT_URL}")
    ok(f"Existing collections: {[c.name for c in collections] or '(none yet)'}")
except Exception as e:
    fail(str(e))
    print("  Start Qdrant with:  docker compose up -d qdrant")


# ─── Step 7: Embed → store → retrieve ────────────────────────────────────────
separator("Step 7 — End-to-end: embed → Qdrant → retrieve")
print("""
WHAT THIS CHECKS:
  The full storage pipeline:
    1. Embed a test sentence via Titan
    2. Insert into Qdrant under a test collection
    3. Query with a semantically similar sentence
    4. Confirm the original text comes back

THIS IS THE CRITICAL INTEGRATION TEST.
  If Steps 5 and 6 pass but Step 7 fails, the issue is usually:
  - EMBEDDING_DIMENSION mismatch (settings vs Qdrant collection)
  - Qdrant collection created with wrong vector size (delete and recreate)
""")

try:
    from qdrant_client.models import Distance, VectorParams, PointStruct

    TEST_COLLECTION = "test_integration"
    qdrant = QdrantClient(url=settings.QDRANT_URL)
    embedder = BedrockEmbedder()

    # Create a temporary test collection
    existing = [c.name for c in qdrant.get_collections().collections]
    if TEST_COLLECTION in existing:
        qdrant.delete_collection(TEST_COLLECTION)

    qdrant.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config=VectorParams(
            size=settings.EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
    )
    ok("Test collection created")

    # Embed and insert a known regulatory sentence
    text = "Banks must maintain a minimum CET1 capital ratio of 4.5% of risk-weighted assets."
    vector = embedder.embed(text)
    qdrant.upsert(
        collection_name=TEST_COLLECTION,
        points=[PointStruct(id=1, vector=vector, payload={"text": text, "regulation": "Basel III"})],
    )
    ok(f"Inserted 1 point (dim={len(vector)})")

    # Query with semantically similar text
    query = "What is the CET1 minimum ratio requirement?"
    query_vector = embedder.embed(query)
    results = qdrant.search(
        collection_name=TEST_COLLECTION,
        query_vector=query_vector,
        limit=1,
    )

    if results:
        ok(f"Retrieved: '{results[0].payload['text'][:80]}...'")
        ok(f"Score:     {results[0].score:.4f} (higher = more similar)")
    else:
        fail("No results returned from Qdrant search")

    # Clean up test collection
    qdrant.delete_collection(TEST_COLLECTION)
    ok("Test collection cleaned up")

except Exception as e:
    fail(str(e))


# ─── Step 8: Ingest sample docs + ask a question ─────────────────────────────
separator("Step 8 — Full pipeline: ingest docs → compliance query")
print("""
WHAT THIS CHECKS:
  The complete end-to-end flow:
    1. Load .txt sample docs from sample_docs/
    2. Chunk them with SemanticChunker
    3. Embed chunks via Titan → store in Qdrant
    4. Run a compliance query through the full agent
    5. Print the structured response

THIS WILL TAKE 2-3 MINUTES because of embedding all chunks.
  (Each Titan call = ~200ms, with ~50 chunks that's ~10 seconds.
   Plus Bedrock LLM calls for synthesis = another ~2 seconds.)
""")

try:
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import SemanticChunker
    from src.ingestion.embedder import VectorStoreIngester

    loader = DocumentLoader()
    chunker = SemanticChunker(max_tokens=400, overlap_pct=0.1)
    ingester = VectorStoreIngester()

    ingester.ensure_collection_exists()

    docs_dir = settings.DOCS_DIR
    docs = loader.load_directory(docs_dir)
    ok(f"Loaded {len(docs)} documents from {docs_dir}/")

    total_chunks = 0
    for doc in docs:
        chunks = chunker.chunk(doc.text, doc.doc_id, doc.metadata)
        n = ingester.ingest(chunks)
        total_chunks += n
        ok(f"  {doc.metadata.get('source_document', doc.doc_id)}: {len(chunks)} chunks → {n} upserted")

    ok(f"Total chunks ingested: {total_chunks}")

    stats = ingester.collection_stats()
    ok(f"Qdrant collection stats: {stats}")

    # Run a real compliance question
    print("\n  Running compliance query...")
    from src.agent.compliance_agent import FallbackComplianceAgent

    agent = FallbackComplianceAgent()
    result = agent.run("What is the minimum CET1 capital ratio under Basel III and what triggers the conservation buffer restrictions?")

    print(f"\n  ANSWER:     {result.get('answer', '')[:300]}...")
    print(f"  CONFIDENCE: {result.get('confidence')}")
    print(f"  REGULATIONS:{result.get('applicable_regulations')}")
    print(f"  HUMAN REVIEW NEEDED: {result.get('requires_human_review')}")
    ok("Full pipeline completed successfully")

except Exception as e:
    fail(str(e))
    import traceback
    traceback.print_exc()


# ─── Summary ──────────────────────────────────────────────────────────────────
separator("Setup verification complete")
print("""
NEXT STEPS:

  1. If all steps passed:
     Start the FastAPI server:
       uvicorn src.api.main:app --reload --port 8000

     Test via curl:
       curl -X POST http://localhost:8000/query \\
         -H "X-API-Key: dev-key-1234" \\
         -H "Content-Type: application/json" \\
         -d '{"query": "What is the minimum LCR under Basel III?"}'

  2. Compare Mistral 7B vs Ministral 8B:
     In .env, change LLM_PRIMARY_MODEL to ministral.ministral-8b-instruct-v3:0
     Re-run Step 3 and compare response quality and latency.

  3. Run the full evaluation:
     python src/evaluation/evaluator.py
     (Runs all 20 questions, produces outputs/eval_report.json)
""")
