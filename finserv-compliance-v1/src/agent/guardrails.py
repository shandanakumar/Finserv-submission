"""
src/agent/guardrails.py

Output validation, citation verification, and PII redaction.

Guardrail layers (in order of application):
1. PII redaction (pre-LLM): scan inputs, redact before sending to model
2. Citation validation (post-LLM): verify cited DOC_IDs exist in retrieved context
3. Schema validation (post-LLM): ensure response conforms to required format
4. Confidence downgrade: flag and downgrade if validation fails

Note on PII approach:
Using Microsoft Presidio (open-source, self-hosted) rather than an LLM for PII detection.
Rationale: LLMs are probabilistic — a deterministic rule-based system is more auditable
and reliable for PII redaction in a regulated environment. Every redaction is logged.
"""
import re
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    citations_valid: bool
    valid_citations: list[dict]
    invalid_citations: list[str]
    pii_detected: bool
    pii_entities: list[dict]
    cleaned_response: str
    confidence_adjustment: str  # "none" | "downgrade_to_medium" | "downgrade_to_low"


class PIIRedactor:
    """
    Rule-based PII detection and redaction using pattern matching + optional Presidio.
    Financial domain PII includes: account numbers, customer IDs, national IDs, phone numbers.
    """

    # Patterns for financial-domain PII
    PII_PATTERNS = [
        # Account numbers (10-18 digits, common in banking)
        (r'\b\d{10,18}\b(?!\s*(?:crore|lakh|million|billion))', '[ACCOUNT_NUMBER]'),
        # SWIFT/BIC codes: 8 or 11 alphanumeric chars
        (r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b', '[SWIFT_BIC]'),
        # Indian PAN: 5 letters, 4 digits, 1 letter
        (r'\b[A-Z]{5}\d{4}[A-Z]\b', '[PAN_NUMBER]'),
        # Aadhaar: 12 digits with optional spaces
        (r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[AADHAAR]'),
        # Email addresses
        (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        # Phone numbers (international format)
        (r'(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4,6}', '[PHONE]'),
        # Customer/entity names preceded by "customer:", "entity:", "counterparty:"
        # (Heuristic — not always reliable; Presidio handles this better)
        # IBAN: 2 letters + 2 digits + up to 30 alphanumeric
        (r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b', '[IBAN]'),
    ]

    def __init__(self):
        self._presidio_available = self._check_presidio()

    def _check_presidio(self) -> bool:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            return True
        except ImportError:
            logger.info("Presidio not available — using regex-based PII redaction")
            return False

    def redact(self, text: str) -> tuple[str, list[dict]]:
        """
        Redact PII from text.
        Returns (redacted_text, list of detected PII entities with positions).
        """
        if self._presidio_available:
            return self._redact_presidio(text)
        return self._redact_regex(text)

    def _redact_presidio(self, text: str) -> tuple[str, list[dict]]:
        """Use Microsoft Presidio for NLP-powered PII detection."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            analyzer = AnalyzerEngine()
            anonymizer = AnonymizerEngine()

            results = analyzer.analyze(text=text, language="en")
            entities = [{"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score} for r in results]
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized.text, entities
        except Exception as e:
            logger.error(f"Presidio redaction failed: {e} — falling back to regex")
            return self._redact_regex(text)

    def _redact_regex(self, text: str) -> tuple[str, list[dict]]:
        """Regex-based PII redaction."""
        entities = []
        redacted = text

        for pattern, replacement in self.PII_PATTERNS:
            matches = list(re.finditer(pattern, redacted))
            for match in reversed(matches):  # Reverse to preserve positions
                entities.append({
                    "type": replacement.strip("[]"),
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(),
                })
                redacted = redacted[:match.start()] + replacement + redacted[match.end():]

        return redacted, entities


class CitationValidator:
    """
    Validates that citations in LLM responses correspond to actually retrieved documents.
    Prevents hallucinated citations — a critical compliance concern.
    """

    CITATION_PATTERN = re.compile(
        r'\[([A-Z0-9_\-\.]+),\s*(?:Section|Article|Para(?:graph)?|Part|Chapter)?\s*([\d\.A-Z]+)\]',
        re.IGNORECASE
    )

    def validate(
        self,
        response: str,
        retrieved_chunks: list[dict]
    ) -> tuple[bool, list[dict], list[str]]:
        """
        Extract all citations from response and verify they appear in retrieved chunks.
        
        Returns:
            (all_valid: bool, valid_citations: list, invalid_citations: list)
        """
        retrieved_doc_ids = {c.get("doc_id", "").upper() for c in retrieved_chunks}
        
        found_citations = self.CITATION_PATTERN.findall(response)
        
        valid_citations = []
        invalid_citations = []

        for doc_id, section in found_citations:
            doc_id_upper = doc_id.upper().strip()
            
            # Check if this DOC_ID exists in retrieved chunks
            if doc_id_upper in retrieved_doc_ids:
                # Find the matching chunk for full metadata
                matching_chunk = next(
                    (c for c in retrieved_chunks if c.get("doc_id", "").upper() == doc_id_upper),
                    None
                )
                valid_citations.append({
                    "doc_id": doc_id,
                    "section": section,
                    "regulation_family": matching_chunk.get("regulation_family", "") if matching_chunk else "",
                    "jurisdiction": matching_chunk.get("jurisdiction", "") if matching_chunk else "",
                })
            else:
                invalid_citations.append(f"[{doc_id}, {section}]")
                logger.warning(f"Hallucinated citation detected: [{doc_id}, {section}]")

        # If no citations found in response but we have retrieved chunks, that's also a concern
        if not found_citations and retrieved_chunks:
            logger.info("No citations found in response — may indicate citation format issue")

        all_valid = len(invalid_citations) == 0
        return all_valid, valid_citations, invalid_citations

    def strip_invalid_citations(self, response: str, invalid_citations: list[str]) -> str:
        """Remove invalid/hallucinated citations from response text."""
        cleaned = response
        for citation in invalid_citations:
            # Escape special regex chars in citation
            escaped = re.escape(citation)
            cleaned = re.sub(escaped, '[CITATION_REMOVED]', cleaned)
        return cleaned


class GuardrailValidator:
    """
    Orchestrates all guardrail checks on LLM-generated responses.
    """

    def __init__(self):
        self.pii_redactor = PIIRedactor()
        self.citation_validator = CitationValidator()

    def redact_input(self, text: str) -> tuple[str, list[dict]]:
        """Redact PII from user input before sending to LLM."""
        return self.pii_redactor.redact(text)

    def validate(
        self,
        response: str,
        retrieved_chunks: list[dict],
    ) -> dict:
        """
        Run all post-generation guardrail checks.
        
        Returns validation result dict with:
        - citations_valid: bool
        - valid_citations: list
        - invalid_citations: list
        - pii_detected: bool
        - pii_entities: list
        - cleaned_response: str (with invalid citations stripped, PII removed)
        - confidence_adjustment: str
        """
        # 1. Citation validation
        citations_valid, valid_citations, invalid_citations = self.citation_validator.validate(
            response, retrieved_chunks
        )

        # 2. Strip invalid citations from response
        cleaned_response = response
        if invalid_citations:
            cleaned_response = self.citation_validator.strip_invalid_citations(
                cleaned_response, invalid_citations
            )

        # 3. Scan response for PII (LLM should not repeat PII from context)
        cleaned_response, pii_entities = self.pii_redactor.redact(cleaned_response)
        pii_detected = len(pii_entities) > 0

        if pii_detected:
            logger.warning(f"PII detected in LLM response: {[e['type'] for e in pii_entities]}")

        # 4. Determine confidence adjustment
        if invalid_citations:
            confidence_adjustment = "downgrade_to_low"
        elif pii_detected:
            confidence_adjustment = "downgrade_to_medium"
        else:
            confidence_adjustment = "none"

        return {
            "citations_valid": citations_valid,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "pii_detected": pii_detected,
            "pii_entities": pii_entities,
            "cleaned_response": cleaned_response,
            "confidence_adjustment": confidence_adjustment,
        }
