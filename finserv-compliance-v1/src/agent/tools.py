"""
Tool definitions for the FinServ Compliance Agent.
Each tool wraps a capability (retrieval, transaction lookup, diff, report) that the
LangGraph agent can invoke during its reasoning loop.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool input/output schemas (plain dicts — no LangChain dependency required)
# ---------------------------------------------------------------------------

def regulatory_search(
    query: str,
    regulation_families: list[str] | None = None,
    jurisdictions: list[str] | None = None,
    top_k: int = 5,
    retriever: Any = None,
) -> dict[str, Any]:
    """
    Search the regulatory knowledge base with hybrid retrieval + reranking.

    Args:
        query:               Natural-language question or keyword query.
        regulation_families: Optional filter e.g. ["Basel III", "MiFID II"].
        jurisdictions:       Optional filter e.g. ["EU", "IN"].
        top_k:               Number of chunks to return after reranking.
        retriever:           Injected HybridSearcher instance (set at agent init).

    Returns:
        {
            "chunks": [{"chunk_id", "text", "source", "regulation", "section",
                        "effective_date", "score"}, ...],
            "query":  str
        }
    """
    if retriever is None:
        logger.warning("regulatory_search called without a retriever — returning empty")
        return {"chunks": [], "query": query}

    try:
        chunks = retriever.search(
            query=query,
            regulation_families=regulation_families,
            jurisdictions=jurisdictions,
            top_k=top_k,
        )
        return {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source": c.metadata.get("source_document", ""),
                    "regulation": c.metadata.get("regulation_family", ""),
                    "section": c.metadata.get("section_title", ""),
                    "effective_date": c.metadata.get("effective_date", ""),
                    "score": round(c.rrf_score or 0.0, 4),
                }
                for c in chunks
            ],
            "query": query,
        }
    except Exception as exc:
        logger.error("regulatory_search error: %s", exc)
        return {"chunks": [], "query": query, "error": str(exc)}


def transaction_lookup(transaction_id: str, db_client: Any = None) -> dict[str, Any]:
    """
    Fetch a transaction record from the operational data store.

    In the prototype this returns a synthetic payload; in production it would
    call the transaction DB (PostgreSQL / core banking API).

    Args:
        transaction_id: Unique transaction reference.
        db_client:      Injected database client (optional in prototype).

    Returns:
        Transaction payload dict or error.
    """
    # ---- Prototype: return canned payloads keyed by prefix ----------------
    SYNTHETIC: dict[str, dict] = {
        "TXN-XBORDER": {
            "transaction_id": "TXN-XBORDER-001",
            "type": "cross_border_payment",
            "amount_usd": 2_000_000,
            "currency": "USD",
            "originator": {"name": "FinServ Global Ltd", "jurisdiction": "IN"},
            "beneficiary": {
                "name": "Unknown Entity LLC",
                "jurisdiction": "VU",           # Vanuatu — FATF grey list
                "kyc_verified": False,
            },
            "instrument": "SWIFT wire",
            "timestamp": "2024-03-15T09:32:00Z",
            "flags": ["HIGH_RISK_JURISDICTION", "KYC_UNVERIFIED"],
        },
        "TXN-DERIV": {
            "transaction_id": "TXN-DERIV-002",
            "type": "intra_group_derivative",
            "amount_usd": 450_000_000,
            "notional_ccy": "EUR",
            "instrument": "interest_rate_swap",
            "counterparty": {"name": "FinServ EU GmbH", "group_entity": True},
            "large_exposure_pct_tier1": 28.5,   # threshold: 25 % (Basel III)
            "timestamp": "2024-03-15T11:00:00Z",
            "flags": ["LARGE_EXPOSURE_BREACH"],
        },
        "TXN-RETAIL": {
            "transaction_id": "TXN-RETAIL-003",
            "type": "retail_investment",
            "product": "structured_note_capital_at_risk",
            "complexity_rating": "complex",
            "client": {
                "segment": "retail",
                "mifid_category": "non-professional",
                "suitability_assessed": False,
            },
            "amount_eur": 50_000,
            "timestamp": "2024-03-15T14:20:00Z",
            "flags": ["SUITABILITY_NOT_ASSESSED", "COMPLEX_PRODUCT_RETAIL"],
        },
        "TXN-NBFC": {
            "transaction_id": "TXN-NBFC-004",
            "type": "nbfc_lending",
            "amount_inr": 5_000_000,
            "borrower_category": "small_farmer",
            "priority_sector_eligible": True,
            "psl_sub_category": "agriculture",
            "reported_to_rbi": False,
            "timestamp": "2024-03-15T16:00:00Z",
            "flags": ["PSL_REPORTING_PENDING"],
        },
    }

    for prefix, payload in SYNTHETIC.items():
        if transaction_id.upper().startswith(prefix):
            return payload

    if db_client:
        try:
            return db_client.get_transaction(transaction_id)
        except Exception as exc:
            return {"error": str(exc), "transaction_id": transaction_id}

    return {
        "error": "Transaction not found in prototype dataset",
        "transaction_id": transaction_id,
        "available_test_ids": list(SYNTHETIC.keys()),
    }


def regulation_diff(
    regulation: str,
    old_version: str,
    new_version: str,
    retriever: Any = None,
) -> dict[str, Any]:
    """
    Identify what changed between two versions of a regulation.

    Fetches chunks from both versions and uses keyword diffing to surface
    sections that were added, modified, or removed.

    Args:
        regulation:  e.g. "RBI Master Direction KYC"
        old_version: e.g. "2022-v3"
        new_version: e.g. "2024-v5"
        retriever:   Injected HybridSearcher.

    Returns:
        {
            "regulation": str,
            "old_version": str,
            "new_version": str,
            "changes": [{"section", "change_type", "summary"}, ...]
        }
    """
    if retriever is None:
        return {
            "regulation": regulation,
            "old_version": old_version,
            "new_version": new_version,
            "changes": [],
            "error": "No retriever available",
        }

    try:
        old_chunks = retriever.search(
            query=regulation,
            top_k=20,
            version_filter=old_version,
        )
        new_chunks = retriever.search(
            query=regulation,
            top_k=20,
            version_filter=new_version,
        )

        old_sections = {c.metadata.get("section_title", ""): c.text for c in old_chunks}
        new_sections = {c.metadata.get("section_title", ""): c.text for c in new_chunks}

        changes = []
        all_sections = set(old_sections) | set(new_sections)
        for section in sorted(all_sections):
            if section not in old_sections:
                changes.append({"section": section, "change_type": "ADDED", "summary": "New section introduced"})
            elif section not in new_sections:
                changes.append({"section": section, "change_type": "REMOVED", "summary": "Section removed"})
            elif old_sections[section] != new_sections[section]:
                changes.append({"section": section, "change_type": "MODIFIED", "summary": "Content updated"})

        return {
            "regulation": regulation,
            "old_version": old_version,
            "new_version": new_version,
            "changes": changes,
        }

    except Exception as exc:
        logger.error("regulation_diff error: %s", exc)
        return {
            "regulation": regulation,
            "old_version": old_version,
            "new_version": new_version,
            "changes": [],
            "error": str(exc),
        }


def generate_compliance_report(
    transactions: list[dict],
    period_start: str,
    period_end: str,
    report_type: str = "audit_committee",
) -> dict[str, Any]:
    """
    Generate a structured compliance report for a set of transactions.

    Args:
        transactions: List of transaction assessment dicts (output of agent runs).
        period_start: ISO date string.
        period_end:   ISO date string.
        report_type:  "audit_committee" | "regulatory_submission" | "internal".

    Returns:
        Structured report dict suitable for serialisation to PDF/Markdown.
    """
    total = len(transactions)
    risk_counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    violations_by_regulation: dict[str, int] = {}
    flagged: list[dict] = []

    for txn in transactions:
        risk = txn.get("risk_rating", "UNKNOWN").upper()
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

        for reg in txn.get("applicable_regulations", []):
            violations_by_regulation[reg] = violations_by_regulation.get(reg, 0) + 1

        if risk in ("HIGH", "MEDIUM"):
            flagged.append({
                "transaction_id": txn.get("transaction_id", "N/A"),
                "risk_rating": risk,
                "primary_concern": txn.get("required_actions", ["—"])[0] if txn.get("required_actions") else "—",
                "regulations": txn.get("applicable_regulations", []),
            })

    return {
        "report_metadata": {
            "title": f"Compliance Assessment Report — {report_type.replace('_', ' ').title()}",
            "period": {"start": period_start, "end": period_end},
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "report_type": report_type,
        },
        "executive_summary": {
            "total_transactions_reviewed": total,
            "risk_distribution": risk_counts,
            "high_risk_count": risk_counts.get("HIGH", 0),
            "compliance_rate_pct": round(
                (risk_counts.get("LOW", 0) / total * 100) if total else 0, 1
            ),
        },
        "violations_by_regulation": violations_by_regulation,
        "flagged_transactions": flagged,
        "recommendations": _generate_recommendations(risk_counts, violations_by_regulation),
    }


def _generate_recommendations(
    risk_counts: dict[str, int],
    violations_by_regulation: dict[str, int],
) -> list[str]:
    recs = []
    if risk_counts.get("HIGH", 0) > 0:
        recs.append(
            f"Escalate {risk_counts['HIGH']} HIGH-risk transaction(s) to Compliance Head for "
            "immediate review and regulatory notification where required."
        )
    top_reg = max(violations_by_regulation, key=violations_by_regulation.get, default=None)
    if top_reg:
        recs.append(
            f"Conduct targeted training on {top_reg} — highest frequency of flagged issues "
            f"({violations_by_regulation[top_reg]} occurrence(s) in period)."
        )
    recs.append("Schedule quarterly regulatory change impact assessment as new circulars are ingested.")
    return recs


def flag_for_human_review(
    transaction_id: str,
    reason: str,
    risk_rating: str,
    agent_assessment: dict,
    reviewer_queue: Any = None,
) -> dict[str, Any]:
    """
    Escalate a transaction to the human compliance review queue.

    Args:
        transaction_id:   Transaction reference.
        reason:           Short explanation of why human review is needed.
        risk_rating:      Agent's assigned risk level.
        agent_assessment: Full structured assessment from the agent.
        reviewer_queue:   Injected queue client (Redis/SQS in production).

    Returns:
        Escalation receipt dict.
    """
    escalation = {
        "escalation_id": f"ESC-{transaction_id}-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}",
        "transaction_id": transaction_id,
        "reason": reason,
        "risk_rating": risk_rating,
        "escalated_at": datetime.utcnow().isoformat() + "Z",
        "status": "PENDING_REVIEW",
        "agent_assessment_summary": {
            "applicable_regulations": agent_assessment.get("applicable_regulations", []),
            "required_actions": agent_assessment.get("required_actions", []),
            "confidence": agent_assessment.get("confidence", "N/A"),
        },
    }

    if reviewer_queue:
        try:
            reviewer_queue.enqueue(escalation)
            escalation["status"] = "QUEUED"
        except Exception as exc:
            logger.error("Failed to enqueue escalation: %s", exc)
            escalation["queue_error"] = str(exc)
    else:
        logger.info(
            "Human review escalation (prototype — no queue): %s",
            json.dumps(escalation, indent=2),
        )

    return escalation


# ---------------------------------------------------------------------------
# Tool registry — maps tool names to callables for agent dispatch
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "regulatory_search": regulatory_search,
    "transaction_lookup": transaction_lookup,
    "regulation_diff": regulation_diff,
    "generate_compliance_report": generate_compliance_report,
    "flag_for_human_review": flag_for_human_review,
}

TOOL_SCHEMAS = [
    {
        "name": "regulatory_search",
        "description": (
            "Search the regulatory knowledge base for relevant rules, circulars, "
            "and guidelines. Use for any question about regulatory requirements."
        ),
        "parameters": {
            "query": "string — natural-language or keyword query",
            "regulation_families": "list[string] | null — e.g. ['Basel III', 'MiFID II']",
            "jurisdictions": "list[string] | null — e.g. ['EU', 'IN', 'US']",
            "top_k": "int (default 5) — number of chunks to return",
        },
    },
    {
        "name": "transaction_lookup",
        "description": (
            "Retrieve a structured transaction payload by ID. "
            "Use when the user provides a transaction reference."
        ),
        "parameters": {"transaction_id": "string"},
    },
    {
        "name": "regulation_diff",
        "description": (
            "Compare two versions of a regulation to identify what changed. "
            "Use for impact analysis when a new circular is ingested."
        ),
        "parameters": {
            "regulation": "string — e.g. 'RBI Master Direction KYC'",
            "old_version": "string",
            "new_version": "string",
        },
    },
    {
        "name": "generate_compliance_report",
        "description": (
            "Produce a structured compliance report for a set of transactions "
            "over a given period. Use for audit committee or regulatory submissions."
        ),
        "parameters": {
            "transactions": "list[dict] — transaction assessment dicts",
            "period_start": "string — ISO date",
            "period_end": "string — ISO date",
            "report_type": "string — 'audit_committee' | 'regulatory_submission' | 'internal'",
        },
    },
    {
        "name": "flag_for_human_review",
        "description": (
            "Escalate a transaction to the human compliance review queue. "
            "Use when the agent is uncertain or the risk is HIGH."
        ),
        "parameters": {
            "transaction_id": "string",
            "reason": "string — short explanation",
            "risk_rating": "string — HIGH | MEDIUM | LOW",
            "agent_assessment": "dict — full assessment output",
        },
    },
]
