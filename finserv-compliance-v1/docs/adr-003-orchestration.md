# ADR-003: Agentic Orchestration Framework (Build vs. Buy)

**Status:** Accepted  
**Date:** June 2025  
**Deciders:** AI Architect, Engineering Lead  
**Context Level:** Agent layer

---

## Context

The Regulatory Compliance Assistant requires an agentic orchestration layer that can:
1. Route user intent to the appropriate processing pathway (Q&A, transaction screening, impact analysis, report generation)
2. Invoke tools (RAG retrieval, transaction DB, regulation diff) in sequence or parallel
3. Maintain state across multi-step reasoning (e.g., cross-reference 3 regulations before generating a final assessment)
4. Support reflection/retry cycles when confidence is low
5. Produce a complete, append-only audit trail of every decision step — non-negotiable for regulatory compliance
6. Handle errors gracefully without losing state

The two primary build-vs-buy questions:
- **Framework choice:** LangChain vs. LangGraph vs. CrewAI vs. custom state machine
- **Build depth:** Use a framework's high-level abstractions vs. only its low-level primitives

---

## Decision

**Selected: LangGraph with low-level graph primitives (not LangChain LCEL chains)**

---

## Alternatives Considered

### Option A: LangGraph ✓ SELECTED

LangGraph models agent behavior as a directed (cyclic) graph of nodes and edges, where each node is a function and edges are conditional transitions based on state.

**Why it fits the compliance use case:**
- **Stateful execution:** `ComplianceAgentState` TypedDict persists all inputs, intermediate results, citations, and audit entries across the full execution graph — ideal for audit trail requirements
- **Cyclic graph support:** Native support for reflection loops (low confidence → re-query → re-generate → re-validate) without hacks
- **Deterministic routing:** Conditional edges make routing logic explicit and testable — critical for an auditable system
- **Tool calling:** First-class support for tool invocation with structured input/output schemas
- **Human-in-the-loop:** Built-in interrupt points for routing CRITICAL-risk transactions to human review queue
- **Streaming:** Native streaming of intermediate node outputs for responsive UX
- **Persistence:** LangGraph's checkpointer (SQLite for dev, PostgreSQL for prod) enables resumable workflows — if a pod restarts mid-workflow, the agent can resume from the last checkpoint

**Build approach:**
- Use LangGraph's `StateGraph` and `add_node`/`add_edge`/`add_conditional_edges` primitives
- Do NOT use LangChain's high-level `create_react_agent` or `AgentExecutor` — these hide state management and make audit trail generation difficult
- Write each node as a plain Python function with typed inputs/outputs — fully testable in isolation

**Pros:**
- Explicit, auditable workflow graph
- State management built-in; no custom state machine code
- Active development (LangChain Inc); strong community
- Integrates with LangSmith for tracing (optional; can be disabled for data residency)
- Apache 2.0 license

**Cons:**
- Steeper learning curve than simple LangChain chains
- LangGraph API has evolved rapidly; pin version carefully
- Slight overhead vs. raw Python for simple linear workflows

---

### Option B: CrewAI

CrewAI models agents as role-based "crew members" that collaborate on tasks.

**Evaluation:**
- **Strengths:** Natural abstraction for multi-agent collaboration; easy to define agent roles and goals; fast to prototype
- **Weakness — audit trail:** CrewAI's task/crew abstraction hides internal state; generating a step-by-step audit trail of every tool call, retrieval, and generation decision requires significant custom instrumentation
- **Weakness — determinism:** Role-based agents negotiate tasks in ways that are harder to reason about and test than a deterministic graph
- **Weakness — compliance workflows:** The "crew collaboration" metaphor is designed for open-ended tasks; compliance workflows have fixed, auditable pathways that fit a graph model better
- **License:** MIT

**Verdict:** Good for exploratory multi-agent tasks; not the right fit for deterministic, audit-mandatory compliance workflows. Would require rebuilding much of what LangGraph provides natively.

---

### Option C: Pure LangChain (LCEL Chains + AgentExecutor)

**Evaluation:**
- **Strengths:** Familiar to most ML engineers; extensive integrations; fast to build linear pipelines
- **Weakness — state:** LCEL chains pass data through as dicts; no native persistent state graph; complex workflows require nested chain composition that becomes hard to debug
- **Weakness — cycles:** Reflection loops require awkward workarounds (recursion, while loops outside the chain)
- **Weakness — audit trail:** AgentExecutor logs are string-based; structured audit trail requires significant post-processing
- **When to use:** Simple linear RAG pipelines (retrieval → generate → return) — we use LCEL for the RAG sub-pipeline within LangGraph nodes

**Verdict:** Used for internal RAG pipeline components; not suitable as the top-level orchestration layer.

---

### Option D: Custom State Machine (Python + asyncio)

**Evaluation:**
- **Strengths:** Full control; no framework dependencies; predictable behavior; minimal overhead
- **Weakness — build cost:** Implementing persistent state, retry logic, tool calling schemas, streaming, and human-in-the-loop interrupts from scratch is 4-6 weeks of engineering for a team of 2
- **Weakness — maintenance:** Framework updates (new models, tool schemas) require custom code changes
- **When appropriate:** When framework overhead is unacceptable (ultra-low latency path) or when framework licensing conflicts with enterprise requirements
- **Verdict:** Not justified when LangGraph provides the required primitives under Apache 2.0. Revisit if LangGraph licensing changes or if latency profiling reveals framework overhead is a bottleneck (currently estimated at <50ms overhead per query — acceptable).

---

### Option E: Semantic Kernel (Microsoft)

**Evaluation:**
- **Strengths:** Enterprise-grade, strong .NET/Python support, Microsoft backing, good Azure integration
- **Weakness:** Python SDK is less mature than LangGraph for complex cyclic agents; primary strength is Azure OpenAI integration — not helpful in a self-hosted model environment
- **Verdict:** Better suited for Azure-native, Microsoft-stack teams. LangGraph is more Pythonic and model-agnostic.

---

## Implementation Details

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

# Graph definition — every transition is explicit and auditable
workflow = StateGraph(ComplianceAgentState)

workflow.add_node("router", route_intent)
workflow.add_node("rag_retrieval", retrieve_regulatory_context)
workflow.add_node("cross_reg_checker", check_cross_jurisdiction_conflicts)
workflow.add_node("synthesis", generate_compliance_response)
workflow.add_node("reflection", evaluate_and_reflect)
workflow.add_node("guardrail_validator", validate_output)
workflow.add_node("formatter", format_final_response)
workflow.add_node("human_review", flag_for_human_review)

workflow.set_entry_point("router")

workflow.add_conditional_edges("router", route_by_intent, {
    "qa": "rag_retrieval",
    "screen_transaction": "rag_retrieval",
    "impact_analysis": "rag_retrieval",
    "report": "rag_retrieval",
})

workflow.add_edge("rag_retrieval", "cross_reg_checker")
workflow.add_edge("cross_reg_checker", "synthesis")
workflow.add_edge("synthesis", "reflection")

workflow.add_conditional_edges("reflection", check_confidence, {
    "high": "guardrail_validator",
    "medium": "guardrail_validator",
    "low_retry": "rag_retrieval",    # Re-query with broader search
    "low_human": "human_review",     # Escalate after 2 retries
})

workflow.add_edge("guardrail_validator", "formatter")
workflow.add_edge("formatter", END)
workflow.add_edge("human_review", END)

# PostgreSQL checkpointer for production — enables workflow resumability
checkpointer = PostgresSaver.from_conn_string(settings.POSTGRES_URL)
app = workflow.compile(checkpointer=checkpointer)
```

**Version pinning:**
```
langgraph==0.2.x  # Pin minor version; test before upgrading
langchain-core==0.2.x
```

---

## Consequences

### Positive
- Explicit, testable, auditable workflow graph — each node is a pure function
- Built-in state persistence enables workflow resumability and audit trail
- Reflection loops improve response quality without custom retry logic
- Human-in-the-loop interrupts for CRITICAL risk transactions
- Framework maintenance by LangChain Inc; community support

### Negative
- LangGraph has evolved rapidly; API stability risk on minor version updates (mitigated by version pinning and staging environment)
- Framework adds ~30-50ms overhead vs raw Python (acceptable for our latency targets)
- Team must learn LangGraph graph primitives (estimated 1-week ramp-up)

### Risk Register
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LangGraph API breaking change | Medium | Medium | Pin version; dedicated upgrade cycle in sprint |
| Framework performance overhead | Low | Low | Profiled at <50ms; within budget |
| LangChain Inc business risk | Low | Medium | Apache 2.0 fork is always possible; low-level primitives are stable |

---

*ADR-003 | Reviewed: June 2025 | Next review: December 2025*
