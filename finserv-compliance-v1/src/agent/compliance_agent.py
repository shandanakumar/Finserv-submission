"""
src/agent/compliance_agent.py
"""
import json
import logging
import time
from datetime import datetime
from typing import Optional, TypedDict, Annotated
import operator
import re

from config.settings import settings
from config.prompts import (
    REGULATORY_QA_SYSTEM_PROMPT,
    REGULATORY_QA_USER_TEMPLATE,
    REFLECTION_PROMPT,
)
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import CrossEncoderReranker, format_chunks_for_llm
from src.agent.guardrails import GuardrailValidator
from src.llm import get_llm_client

logger = logging.getLogger(__name__)


class AuditEntry(TypedDict):
    timestamp: str
    node: str
    action: str
    details: dict


class ComplianceAgentState(TypedDict):
    input: str
    intent: str
    transaction: Optional[dict]
    search_query: str
    retrieved_chunks: list
    jurisdictions_triggered: list
    regulations_triggered: list
    context: str
    draft_response: str
    citations: list
    confidence: str
    reflection_count: int
    requires_human_review: bool
    broader_search_needed: bool
    broader_search_terms: list
    error: Optional[str]
    final_response: dict
    audit_log: Annotated[list, operator.add]


def _audit(node: str, action: str, details: dict = None) -> list:
    return [{
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "node": node,
        "action": action,
        "details": details or {},
    }]


def router_node(state: ComplianceAgentState) -> dict:
    query = state["input"].lower()
    transaction = state.get("transaction")

    if transaction:
        intent = "screen_transaction"
    elif any(k in query for k in ["report", "summary", "audit committee"]):
        intent = "report"
    elif any(k in query for k in ["impact", "change", "new circular", "amendment"]):
        intent = "impact_analysis"
    else:
        intent = "qa"

    return {
        "intent": intent,
        "search_query": state["input"],
        "audit_log": _audit("router", "intent_classified", {"intent": intent}),
    }


def rag_retrieval_node(state: ComplianceAgentState) -> dict:
    try:
        print(f"DEBUG rag_retrieval: regulations_triggered={state.get('regulations_triggered')}")

        searcher = HybridSearcher()
        reranker = CrossEncoderReranker()

        chunks = searcher.search(
            query=state["search_query"],
            top_k=settings.RETRIEVAL_TOP_K,
            regulation_families=state.get("regulations_triggered") or None,
        )

        if not chunks:
            return {
                "retrieved_chunks": [],
                "context": "No relevant regulatory documents found.",
                "jurisdictions_triggered": [],
                "regulations_triggered": [],
                "audit_log": _audit("rag_retrieval", "no_results"),
            }
        reranked = reranker.rerank(state["search_query"], chunks, top_k=settings.RERANK_TOP_K)
        context  = format_chunks_for_llm(reranked)
        print("DEBUG RERANKED COUNT:", len(reranked))
        for i, c in enumerate(reranked, 1):
            print(f"  {i}. ver={c.metadata.get('version')} | {c.text[:80].strip()}")
        print("DEBUG CONTEXT LENGTH:", len(context))
        jurisdictions = list({c.metadata.get("jurisdiction", "") for c in reranked if c.metadata.get("jurisdiction")})
        regulations   = list({c.metadata.get("regulation_family", "") for c in reranked if c.metadata.get("regulation_family")})

        return {
            "retrieved_chunks": [c.metadata for c in reranked],
            "context": context,
            "jurisdictions_triggered": jurisdictions,
            "regulations_triggered": regulations,
            "audit_log": _audit("rag_retrieval", "retrieved", {"n_chunks": len(reranked)}),
        }
    except Exception as e:
        logger.error("rag_retrieval_node error: %s", e)
        return {
            "retrieved_chunks": [],
            "context": "Retrieval error.",
            "jurisdictions_triggered": [],
            "regulations_triggered": [],
            "error": str(e),
            "audit_log": _audit("rag_retrieval", "error", {"error": str(e)}),
        }


def synthesis_node(state: ComplianceAgentState) -> dict:
    intent = state.get("intent", "qa")
    use_complex = intent in ("report", "impact_analysis")
    llm = get_llm_client(use_complex=use_complex)
    print("DEBUG CONTEXT:\n" + state.get("context", "")[:500])
   # For better Mistral performance — use focused context (first 2000 chars only)
    full_context = state.get("context", "No context available.")
    focused_context = full_context[:3000]  # Mistral 7B works better with focused context

    user_message = REGULATORY_QA_USER_TEMPLATE.format(
        query=state["input"],
        context=focused_context,
        regulations=", ".join(state.get("regulations_triggered", ["Unknown"])),
        jurisdictions=", ".join(state.get("jurisdictions_triggered", ["Unknown"])),
    )

    try:
        t0       = time.time()
        response = llm.invoke(
            system_prompt=REGULATORY_QA_SYSTEM_PROMPT,
            user_message=user_message,
        )
        latency_ms = int((time.time() - t0) * 1000)

        confidence = "MEDIUM"
        for line in response.text.split("\n"):
            if line.startswith("CONFIDENCE:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in ("HIGH", "MEDIUM", "LOW"):
                    confidence = val
                break

        return {
            "draft_response": response.text,
            "confidence": confidence,
            "audit_log": _audit("synthesis", "generated", {
                "model": getattr(response, "model_id", "unknown"),
                "latency_ms": latency_ms,
                "confidence": confidence,
            }),
        }

    except Exception as e:
        logger.error("synthesis_node error: %s", e)
        return {
            "draft_response": f"Unable to generate response: {e}",
            "confidence": "LOW",
            "error": str(e),
            "audit_log": _audit("synthesis", "error", {"error": str(e)}),
        }


def reflection_node(state: ComplianceAgentState) -> dict:
    count = state.get("reflection_count", 0)

    if count >= settings.MAX_REFLECTION_CYCLES:
        return {
            "broader_search_needed": False,
            "audit_log": _audit("reflection", "max_cycles_reached", {"count": count}),
        }

    llm = get_llm_client()

    reflection_prompt = REFLECTION_PROMPT.format(
        query=state["input"],
        response=state.get("draft_response", ""),
        context=state.get("context", ""),
    )

    try:
        response = llm.invoke(
            system_prompt="You are a quality reviewer. Return only JSON.",
            user_message=reflection_prompt,
        )

        try:
            raw = response.text.strip()
            # Strip markdown code blocks if present
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            raw = raw.strip()
            # Take only first JSON object if multiple
            if raw.count("{") > 1:
                raw = raw[:raw.index("}", raw.index("{")) + 1]
            evaluation = json.loads(raw)
            quality    = evaluation.get("quality", "good")
        except Exception as parse_err:
            logger.warning(f"reflection JSON parse failed: {parse_err} — defaulting to good")
            quality = "good"
            evaluation = {}

        return {
            "reflection_count": count + 1,
            "broader_search_needed": quality == "poor",
            "broader_search_terms": evaluation.get("suggested_terms", []),
            "audit_log": _audit("reflection", f"quality_{quality}", {"cycle": count + 1}),
        }

    except Exception as e:
        logger.warning("reflection_node failed: %s", e)
        return {
            "reflection_count": count + 1,
            "broader_search_needed": False,
            "audit_log": _audit("reflection", "parse_error", {"error": str(e)}),
        }


def guardrail_validator_node(state: ComplianceAgentState) -> dict:
    try:
        validator = GuardrailValidator()
        result    = validator.validate(
            response=state.get("draft_response", ""),
            retrieved_chunks=state.get("retrieved_chunks", []),
        )

        requires_human_review = (
            state.get("confidence", "MEDIUM") == "LOW"
            or result.get("pii_detected", False)
            or not result.get("citations_valid", True)
        )

        return {
            "draft_response": result.get("cleaned_response", state.get("draft_response", "")),
            "requires_human_review": requires_human_review,
            "audit_log": _audit("guardrail_validator", "validated", {
                "pii_detected":          result.get("pii_detected", False),
                "citations_valid":       result.get("citations_valid", True),
                "requires_human_review": requires_human_review,
            }),
        }
    except Exception as e:
        logger.error("guardrail_validator_node error: %s", e)
        return {
            "requires_human_review": False,
            "audit_log": _audit("guardrail_validator", "error", {"error": str(e)}),
        }


def formatter_node(state: ComplianceAgentState) -> dict:
    return {
        "final_response": {
            "answer":                state.get("draft_response", ""),
            "confidence":            state.get("confidence", "LOW"),
            "applicable_regulations":state.get("regulations_triggered", []),
            "jurisdictions":         state.get("jurisdictions_triggered", []),
            "requires_human_review": state.get("requires_human_review", False),
            "audit_log":             state.get("audit_log", []),
            "error":                 state.get("error"),
        },
        "audit_log": _audit("formatter", "response_packaged"),
    }


def build_compliance_agent():
    try:
        from langgraph.graph import StateGraph, END

        def should_retry(state: ComplianceAgentState) -> str:
            if (state.get("broader_search_needed", False)
                    and state.get("reflection_count", 0) < settings.MAX_REFLECTION_CYCLES):
                return "rag_retrieval"
            return "guardrail_validator"

        graph = StateGraph(ComplianceAgentState)
        graph.add_node("router",              router_node)
        graph.add_node("rag_retrieval",       rag_retrieval_node)
        graph.add_node("synthesis",           synthesis_node)
        graph.add_node("reflection",          reflection_node)
        graph.add_node("guardrail_validator", guardrail_validator_node)
        graph.add_node("formatter",           formatter_node)

        graph.set_entry_point("router")
        graph.add_edge("router",              "rag_retrieval")
        graph.add_edge("rag_retrieval",       "synthesis")
        graph.add_edge("synthesis",           "reflection")
        graph.add_conditional_edges("reflection", should_retry, {
            "rag_retrieval":       "rag_retrieval",
            "guardrail_validator": "guardrail_validator",
        })
        graph.add_edge("guardrail_validator", "formatter")
        graph.add_edge("formatter",           END)

        return graph.compile()

    except ImportError:
        return FallbackComplianceAgent()


class FallbackComplianceAgent:

    def run(self, query: str, transaction: Optional[dict] = None,regulation_filter: str = None) -> dict:
        state: ComplianceAgentState = {
            "input":                  query,
            "intent":                 "",
            "transaction":            transaction,
            "search_query":           query,
            "retrieved_chunks":       [],
            "jurisdictions_triggered":[],
            "regulations_triggered": [regulation_filter] if regulation_filter else [],
            "context":                "",
            "draft_response":         "",
            "citations":              [],
            "confidence":             "MEDIUM",
            "reflection_count":       0,
            "requires_human_review":  False,
            "broader_search_needed":  False,
            "broader_search_terms":   [],
            "error":                  None,
            "final_response":         {},
            "audit_log":              [],
        }

        def merge(s, updates):
            s = dict(s)
            for k, v in updates.items():
                if k == "audit_log":
                    s["audit_log"] = s.get("audit_log", []) + v
                else:
                    s[k] = v
            return s

        state = merge(state, router_node(state))
        state = merge(state, rag_retrieval_node(state))
        state = merge(state, synthesis_node(state))
        state = merge(state, reflection_node(state))
        state = merge(state, guardrail_validator_node(state))
        state = merge(state, formatter_node(state))

        return state["final_response"]