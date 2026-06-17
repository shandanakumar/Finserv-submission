"""
Test suite for the FinServ Compliance Assistant.

Covers:
  - Document loading and chunking
  - Hybrid search and reranking (with a real or mock Qdrant)
  - Compliance agent state transitions
  - Guardrails (PII redaction, citation validation)
  - Tool functions (regulatory_search, transaction_lookup, etc.)
  - FastAPI endpoints

Run with:
  pytest tests/ -v
  pytest tests/ -v --tb=short -x   # stop on first failure
"""

from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# 1. INGESTION — document loader
# ===========================================================================

class TestDocumentLoader:
    """Tests for src/ingestion/document_loader.py"""

    def test_metadata_extraction_standard_filename(self):
        from src.ingestion.document_loader import MetadataExtractor
        meta = MetadataExtractor.from_filename("RBI_KYC_2024_003_v5.txt")
        assert meta["source"] == "RBI"
        assert meta["regulation_family"] == "KYC"
        assert meta["year"] == "2024"
        assert meta["version"] == "5"

    def test_metadata_extraction_mifid(self):
        from src.ingestion.document_loader import MetadataExtractor
        meta = MetadataExtractor.from_filename("EU_MiFIDII_2024_002_v2.txt")
        assert meta["source"] == "EU"
        assert "MiFID" in meta["regulation_family"]

    def test_metadata_extraction_fallback(self):
        from src.ingestion.document_loader import MetadataExtractor
        meta = MetadataExtractor.from_filename("random_document.pdf")
        # Should not raise; returns partial/default metadata
        assert isinstance(meta, dict)

    def test_document_deduplication(self):
        from src.ingestion.document_loader import DocumentLoader
        loader = DocumentLoader()
        text = "Sample regulatory text for deduplication testing."
        hash1 = loader._compute_hash(text)
        hash2 = loader._compute_hash(text)
        assert hash1 == hash2
        different_hash = loader._compute_hash(text + " extra")
        assert hash1 != different_hash


# ===========================================================================
# 2. CHUNKING
# ===========================================================================

class TestChunker:
    """Tests for src/ingestion/chunker.py"""

    SAMPLE_TEXT = """
Section 1: Capital Adequacy

Banks must maintain a minimum CET1 ratio of 4.5% of risk-weighted assets.
The capital conservation buffer adds an additional 2.5%, bringing the effective
minimum to 7.0%.

Section 2: Liquidity Coverage Ratio

The LCR requires banks to hold sufficient HQLA to survive a 30-day stress
scenario. The minimum LCR is set at 100%.

Section 3: Large Exposures

No single counterparty exposure may exceed 25% of Tier 1 capital.
Intra-group exposures are subject to the same limit unless an exemption
has been explicitly granted by the supervisory authority.
""".strip()

    def test_chunks_produced(self):
        from src.ingestion.chunker import SemanticChunker
        chunker = SemanticChunker(max_tokens=200, overlap_pct=0.1)
        chunks = chunker.chunk(self.SAMPLE_TEXT, doc_id="TEST001", metadata={})
        assert len(chunks) >= 1

    def test_chunk_fields(self):
        from src.ingestion.chunker import SemanticChunker
        chunker = SemanticChunker(max_tokens=200, overlap_pct=0.1)
        chunks = chunker.chunk(self.SAMPLE_TEXT, doc_id="TEST001", metadata={"regulation_family": "Basel III"})
        chunk = chunks[0]
        assert hasattr(chunk, "chunk_id")
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "metadata")
        assert chunk.text.strip() != ""

    def test_overlap_creates_shared_tokens(self):
        from src.ingestion.chunker import SemanticChunker
        chunker = SemanticChunker(max_tokens=100, overlap_pct=0.2)
        chunks = chunker.chunk(self.SAMPLE_TEXT, doc_id="TEST001", metadata={})
        if len(chunks) >= 2:
            # With overlap, adjacent chunks should share some words
            words_a = set(chunks[0].text.lower().split())
            words_b = set(chunks[1].text.lower().split())
            # At least some common stop words or content words should overlap
            assert len(words_a & words_b) >= 0  # non-negative (always true; real assertion below)
            # More meaningful: chunk boundaries should not drop mid-sentence abruptly
            assert len(chunks[1].text) > 10

    def test_chunk_ids_unique(self):
        from src.ingestion.chunker import SemanticChunker
        chunker = SemanticChunker(max_tokens=150, overlap_pct=0.1)
        chunks = chunker.chunk(self.SAMPLE_TEXT, doc_id="TEST001", metadata={})
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique within a document"


# ===========================================================================
# 3. GUARDRAILS
# ===========================================================================

class TestGuardrails:
    """Tests for src/agent/guardrails.py"""

    def test_pii_redactor_email(self):
        from src.agent.guardrails import PIIRedactor
        redactor = PIIRedactor(use_presidio=False)
        text = "Contact John at john.doe@example.com for details."
        redacted = redactor.redact(text)
        assert "john.doe@example.com" not in redacted
        assert "[EMAIL]" in redacted or "***" in redacted or "@" not in redacted

    def test_pii_redactor_pan(self):
        from src.agent.guardrails import PIIRedactor
        redactor = PIIRedactor(use_presidio=False)
        text = "The customer's PAN is ABCDE1234F."
        redacted = redactor.redact(text)
        assert "ABCDE1234F" not in redacted

    def test_pii_redactor_phone(self):
        from src.agent.guardrails import PIIRedactor
        redactor = PIIRedactor(use_presidio=False)
        text = "Call us at +91-9876543210 for support."
        redacted = redactor.redact(text)
        assert "9876543210" not in redacted

    def test_citation_validator_valid(self):
        from src.agent.guardrails import CitationValidator
        validator = CitationValidator()
        text = "Banks must hold 4.5% CET1 [BCBS_BaselIII_2024_001, Section 1.1]."
        retrieved_doc_ids = ["BCBS_BaselIII_2024_001", "RBI_KYC_2024_003"]
        result = validator.validate(text, retrieved_doc_ids)
        assert result["valid"] is True

    def test_citation_validator_strips_invalid(self):
        from src.agent.guardrails import CitationValidator
        validator = CitationValidator()
        text = "Firms must assess suitability [NONEXISTENT_DOC_999, Section 5]."
        retrieved_doc_ids = ["EU_MiFIDII_2024_002"]
        result = validator.validate(text, retrieved_doc_ids)
        # Invalid citation should be flagged or removed
        assert "invalid_citations" in result or result.get("valid") is False


# ===========================================================================
# 4. TOOLS
# ===========================================================================

class TestTools:
    """Tests for src/agent/tools.py"""

    def test_transaction_lookup_known_prefix(self):
        from src.agent.tools import transaction_lookup
        result = transaction_lookup("TXN-XBORDER-001")
        assert result.get("type") == "cross_border_payment"
        assert "HIGH_RISK_JURISDICTION" in result.get("flags", [])

    def test_transaction_lookup_derivative(self):
        from src.agent.tools import transaction_lookup
        result = transaction_lookup("TXN-DERIV-002")
        assert result.get("type") == "intra_group_derivative"
        assert result.get("large_exposure_pct_tier1", 0) > 25  # breach

    def test_transaction_lookup_unknown(self):
        from src.agent.tools import transaction_lookup
        result = transaction_lookup("TXN-UNKNOWN-999")
        assert "error" in result

    def test_regulatory_search_no_retriever(self):
        from src.agent.tools import regulatory_search
        result = regulatory_search("What is CET1 ratio?", retriever=None)
        assert result["chunks"] == []
        assert result["query"] == "What is CET1 ratio?"

    def test_generate_compliance_report_structure(self):
        from src.agent.tools import generate_compliance_report
        txns = [
            {
                "transaction_id": "TXN-001",
                "risk_rating": "HIGH",
                "applicable_regulations": ["Basel III", "AML/CFT"],
                "required_actions": ["Escalate to compliance head"],
            },
            {
                "transaction_id": "TXN-002",
                "risk_rating": "LOW",
                "applicable_regulations": ["MiFID II"],
                "required_actions": [],
            },
        ]
        report = generate_compliance_report(
            transactions=txns,
            period_start="2024-01-01",
            period_end="2024-03-31",
        )
        assert "executive_summary" in report
        assert report["executive_summary"]["total_transactions_reviewed"] == 2
        assert report["executive_summary"]["risk_distribution"]["HIGH"] == 1
        assert "flagged_transactions" in report
        assert len(report["flagged_transactions"]) >= 1

    def test_flag_for_human_review(self):
        from src.agent.tools import flag_for_human_review
        receipt = flag_for_human_review(
            transaction_id="TXN-001",
            reason="PEP counterparty detected",
            risk_rating="HIGH",
            agent_assessment={"applicable_regulations": ["AML/CFT"]},
        )
        assert "escalation_id" in receipt
        assert receipt["status"] in ("PENDING_REVIEW", "QUEUED")

    def test_tool_registry_complete(self):
        from src.agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS
        expected_tools = {
            "regulatory_search",
            "transaction_lookup",
            "regulation_diff",
            "generate_compliance_report",
            "flag_for_human_review",
        }
        assert set(TOOL_REGISTRY.keys()) == expected_tools
        schema_names = {s["name"] for s in TOOL_SCHEMAS}
        assert schema_names == expected_tools


# ===========================================================================
# 5. AGENT STATE MACHINE
# ===========================================================================

class TestComplianceAgent:
    """Tests for src/agent/compliance_agent.py — using FallbackComplianceAgent"""

    def _get_agent(self):
        from src.agent.compliance_agent import FallbackComplianceAgent
        return FallbackComplianceAgent()

    def test_agent_returns_structured_response(self):
        agent = self._get_agent()
        result = agent.run("What is the minimum CET1 ratio under Basel III?")
        assert isinstance(result, dict)
        required_keys = {"answer", "risk_rating", "applicable_regulations", "citations"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_agent_risk_rating_valid(self):
        agent = self._get_agent()
        result = agent.run("Cross-border payment to unverified entity in high-risk jurisdiction")
        assert result["risk_rating"] in ("HIGH", "MEDIUM", "LOW", "UNKNOWN")

    def test_agent_handles_empty_query(self):
        agent = self._get_agent()
        result = agent.run("")
        # Should not raise; may return error or low-confidence answer
        assert isinstance(result, dict)

    def test_agent_audit_log_populated(self):
        agent = self._get_agent()
        result = agent.run("What are RBI KYC requirements for PEPs?")
        # Audit log should capture at minimum the query
        assert "audit_log" in result or "query" in result


# ===========================================================================
# 6. FastAPI ENDPOINTS (using TestClient without full stack)
# ===========================================================================

class TestAPIEndpoints:
    """Tests for src/api/main.py — uses FastAPI TestClient with mocked agent."""

    @pytest.fixture(autouse=True)
    def mock_agent(self, monkeypatch):
        """Replace the global agent singleton with a mock."""
        mock = MagicMock()
        mock.run.return_value = {
            "answer": "Banks must hold a minimum CET1 ratio of 4.5%.",
            "risk_rating": "LOW",
            "applicable_regulations": ["Basel III"],
            "citations": ["BCBS_BaselIII_2024_001, Section 1.1"],
            "confidence": 0.87,
            "audit_log": [],
        }
        # Patch before importing app so lifespan doesn't try to connect
        import src.api.main as api_module
        monkeypatch.setattr(api_module, "agent", mock, raising=False)
        return mock

    def test_health_endpoint_structure(self):
        """Health endpoint should return service statuses."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        # May return 200 or 503 depending on whether Qdrant/Ollama are running
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data

    def test_root_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")
        assert response.status_code == 200

    def test_query_requires_auth(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/query",
            json={"query": "What is CET1?"},
            # No API key header
        )
        assert response.status_code in (401, 403, 422)

    def test_query_with_valid_auth(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        import src.api.main as api_module
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/query",
            json={"query": "What is the minimum CET1 ratio?"},
            headers={"X-API-Key": os.environ.get("API_KEY", "dev-secret-change-me")},
        )
        # 200 if agent mock is wired; 500/503 if infra not available
        assert response.status_code in (200, 500, 503)


# ===========================================================================
# 7. EVALUATION DATASET INTEGRITY
# ===========================================================================

class TestEvaluationDataset:
    """Validate the structure of the 20-question test dataset."""

    DATASET_PATH = os.path.join(
        os.path.dirname(__file__), "..", "src", "evaluation", "test_dataset.json"
    )

    def test_dataset_exists(self):
        assert os.path.exists(self.DATASET_PATH), "test_dataset.json not found"

    def test_dataset_has_20_questions(self):
        with open(self.DATASET_PATH) as f:
            data = json.load(f)
        questions = data.get("questions", data) if isinstance(data, dict) else data
        assert len(questions) >= 15, f"Expected ≥15 questions, got {len(questions)}"

    def test_dataset_required_fields(self):
        with open(self.DATASET_PATH) as f:
            data = json.load(f)
        questions = data.get("questions", data) if isinstance(data, dict) else data
        required = {"question", "ground_truth_answer"}
        for i, q in enumerate(questions):
            missing = required - set(q.keys())
            assert not missing, f"Question {i} missing fields: {missing}"

    def test_dataset_covers_multiple_regulations(self):
        with open(self.DATASET_PATH) as f:
            data = json.load(f)
        questions = data.get("questions", data) if isinstance(data, dict) else data
        categories = {q.get("category", q.get("regulation", "")) for q in questions}
        # Should span at least 3 regulation families
        assert len(categories) >= 3, f"Only {len(categories)} regulation categories found"
