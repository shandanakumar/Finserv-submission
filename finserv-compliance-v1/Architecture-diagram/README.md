# FinServ Global — Regulatory Compliance Assistant

AI-powered regulatory compliance assistant for FinServ Global. Helps compliance officers navigate Basel III, MiFID II, RBI KYC, RBI PSL, and FATF AML regulations using a RAG pipeline with version-aware document ingestion and a LangGraph agentic workflow.

---

## Quick Start

### Prerequisites

- Python 3.13+
- AWS account with Bedrock access (Mistral 7B + Mixtral 8×7B + Titan Embeddings)
- AWS credentials configured (`~/.aws/credentials` or environment variables)

### Installation

```bash
git clone <repository-url>
cd finserv-compliance

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Configuration

Copy and edit the environment file:

```bash
copy .env.example .env
```

Required settings in `.env`:

```
LLM_BACKEND=bedrock
AWS_REGION=us-east-1
QDRANT_MODE=disk
BEDROCK_PRIMARY_MODEL=mistral.mistral-7b-instruct-v0:2
BEDROCK_COMPLEX_MODEL=mistral.mixtral-8x7b-instruct-v0:1
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
EMBEDDING_DIMENSION=1024
QDRANT_COLLECTION=regulatory_docs
```

### Running the system

**Step 1 — Add regulatory PDFs to `real_docs/`:**

Download these documents and save to `real_docs/`:

| File | Source |
|---|---|
| `BIS_Basel3_d424_v2.pdf` | https://www.bis.org/bcbs/publ/d424.pdf |
| `BIS_Basel3_2010_v1.pdf` | https://www.bis.org/publ/bcbs189.pdf |
| `MiFID CELEX_32014L0065_EN_TXT.pdf` | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014L0065 |
| `FATF Recommendations 2012.pdf` | https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/FATF%20Recommendations%202012.pdf |
| `RBI_KYC_CommercialBanks_2025_v2.pdf` | https://rbidocs.rbi.org.in/rdocs/notification/PDFs/169MD.pdf |
| `RBI_KYC_2016_v1.pdf` | RBI website |
| `RBI_PSL_2025_v2.pdf` | https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12799 |
| `RBI_PSL_2020_v1.pdf` | RBI website |

Also copy sample reference documents:

```bash
copy sample_docs\RBI_KYC_2024_003_v5.txt real_docs\
copy sample_docs\RBI_PSL_2024_004_v3.txt real_docs\
```

**Step 2 — Ingest documents (one-time, ~7 minutes):**

```bash
python scripts\ingest_pdfs.py
```

This parses all PDFs, extracts text and tables, chunks with heading detection, embeds via Titan, and stores 2,932 vectors to disk. Only needed once — vectors persist across restarts.

**Step 3 — Start the API server:**

```bash
uvicorn src.api.main:app --reload --port 8000
```

**Step 4 — Start the Streamlit UI (new terminal):**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
finserv-compliance/
├── config/
│   ├── settings.py          # Pydantic settings — all config via .env
│   └── prompts.py           # System prompts and user templates
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py    # PDF parsing (pdfminer + camelot + pdfplumber)
│   │   ├── chunker.py            # Hierarchical semantic chunker
│   │   ├── embedder.py           # Titan embeddings + Qdrant ingestion
│   │   ├── version_registry.py   # Document version lifecycle metadata
│   │   └── table_chunker.py      # Camelot table extraction
│   ├── retrieval/
│   │   ├── hybrid_search.py      # Dense + BM25 + RRF fusion
│   │   └── reranker.py           # Keyword overlap reranker
│   ├── agent/
│   │   ├── compliance_agent.py   # LangGraph 6-node agent
│   │   ├── tools.py              # 5 compliance tools
│   │   └── guardrails.py         # PII redaction + citation validation
│   ├── llm/
│   │   ├── bedrock_client.py     # Mistral + Mixtral + Titan via Bedrock
│   │   └── ollama_client.py      # Local Ollama (V2 alternative)
│   ├── evaluation/
│   │   └── run_eval.py           # RAGAS-style evaluation (20 Q&A pairs)
│   └── api/
│       └── main.py               # FastAPI endpoints
├── scripts/
│   ├── ingest_pdfs.py            # Full ingestion pipeline
│   └── run_with_ingestion.py     # Ingest + start server combined
├── docs/
│   ├── architecture.md           # Full architecture document
│   └── adr/
│       ├── ADR-001-vector-database.md
│       ├── ADR-002-orchestration-framework.md
│       └── ADR-003-llm-hosting-strategy.md
├── sample_docs/                  # Clean reference text documents
├── real_docs/                    # Your downloaded regulatory PDFs
├── outputs/                      # Eval reports, ingestion summary
├── app.py                        # Streamlit UI
└── README.md
```

---

## API Reference

### POST /query

Answer a regulatory compliance question.

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: dev-key-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the minimum CET1 capital ratio under Basel III?",
    "regulation_families": ["BASEL_III"]
  }'
```

Response:

```json
{
  "request_id": "85ea49d4-...",
  "query": "What is the minimum CET1 capital ratio under Basel III?",
  "answer": "ANSWER: The minimum CET1 capital ratio is 4.5% of risk-weighted assets.\nSOURCES: Basel III d424, Section 2\nCONFIDENCE: HIGH",
  "confidence": "HIGH",
  "applicable_regulations": ["BASEL_III"],
  "requires_human_review": false,
  "processing_time_ms": 4539.0
}
```

### POST /screen-transaction

Screen a financial transaction against applicable regulations.

```bash
curl -X POST http://localhost:8000/screen-transaction \
  -H "X-API-Key: dev-key-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-XBORDER-001",
    "description": "Cross-border payment of $2M to non-KYC verified entity in high-risk jurisdiction"
  }'
```

### GET /health

```bash
curl http://localhost:8000/health
```

---

## Running the Evaluation

```bash
# Stop the API server first (Qdrant single-process lock)
# Ctrl+C in the server terminal, then:

python src\evaluation\run_eval.py
```

Results saved to:
- `outputs/eval_report.json` — full per-question results
- `outputs/eval_report.md` — submission-ready markdown report

**Evaluation results (prototype):**

| Metric | Score | Target |
|---|---|---|
| Faithfulness | 0.632 | >= 0.80 |
| Answer Relevance | 0.516 | >= 0.75 |
| Context Precision | 0.715 | >= 0.75 |
| Context Recall | 0.715 | >= 0.70 |
| Avg Latency | 3,712ms | <= 10,000ms |
| HIGH confidence | 85% (17/20) | — |

---

## Architecture Highlights

**Version-aware ingestion:** Every chunk carries `regulation_id`, `version`, `effective_from`, `is_superseded` as Qdrant payload. Retrieval filters superseded chunks by default. Historical queries and regulatory comparison supported.

**Hybrid retrieval:** Titan dense vectors (semantic) + BM25 keyword index + RRF(k=60) fusion. Regulation-specific filtering eliminates cross-regulation noise.

**LangGraph agent:** 6-node deterministic graph with typed state, append-only audit log, reflection retry loop (max 2 cycles), and FallbackComplianceAgent for resilience.

**Table extraction:** camelot-py lattice mode for RBI/BIS structured tables. Converts row data to natural language sentences for better RAG retrieval of numeric regulatory targets.

**Two deployment variants:**
- V1 (Bedrock): no GPU required, pay-per-token, data stays in AWS
- V2 (Ollama): self-hosted, zero API cost, 100% data sovereignty, GPU required

---

## Known Limitations

1. **Table extraction:** Some RBI PDFs use complex multi-column tables where column associations are lost during text extraction. Mitigated with camelot-py and clean reference documents. Production fix: multimodal vision model (Pixtral on Bedrock).

2. **Hindi text encoding:** RBI 2025 PDFs contain Devanagari characters that pdfminer cannot extract. Supplemented with English reference documents.

3. **Qdrant single-process:** Local disk mode allows only one process at a time. Running evaluation requires stopping the API server. Production fix: Qdrant server mode with Docker.

4. **RAGAS metrics:** Custom keyword-overlap metrics penalise correct answers using different phrasing than ground truth. Production fix: LLM-as-judge via the RAGAS library.

---

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| Python | CPython | 3.13.2 |
| API framework | FastAPI | 0.115+ |
| LLM orchestration | LangGraph | 0.2+ |
| Vector store | Qdrant | 1.9+ |
| PDF extraction | pdfminer.six | 20260107 |
| Table extraction | camelot-py | 0.11+ |
| LLM inference | AWS Bedrock (Mistral AI) | — |
| Embeddings | Amazon Titan Embed v2 | — |
| UI | Streamlit | 1.35+ |
| Evaluation | Custom RAGAS-style | — |

---

## License

This project uses only open-source libraries with Apache 2.0 or MIT licences. Mistral 7B and Mixtral 8×7B are Apache 2.0. Amazon Titan Embeddings is a managed AWS service.
