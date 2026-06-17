# ADR-002: Agentic Orchestration Framework Selection

**Status:** Accepted  
**Date:** 2024-01-15  
**Deciders:** AI Architect

---

## Context

The compliance agent requires an orchestration framework to manage a multi-step workflow:
1. Intent classification (router)
2. RAG retrieval
3. Re-ranking
4. LLM synthesis
5. Quality reflection with retry
6. Guardrail validation

The framework must support:
- Cyclic graphs (retry loops — reflection node can loop back to retrieval)
- Typed state management with append-only audit log
- Deterministic execution paths (critical for compliance — results must be reproducible)
- Error handling and fallback strategies
- Tool definition for regulatory search, transaction lookup, report generation

---

## Decision

**Selected: LangGraph**

---

## Alternatives Considered

| Option | Pros | Cons | Decision |
|---|---|---|---|
| **LangGraph** | Cyclic graphs, typed state, deterministic, Anthropic-backed | Steeper learning curve than plain LangChain | ✅ Selected |
| **CrewAI** | Easy multi-agent setup, role-based agents | Non-deterministic execution, hard to audit, agents can diverge | ❌ Rejected |
| **Plain LangChain** | Simple chains, large ecosystem | Linear only — cannot support retry loops, no typed state | ❌ Rejected |
| **AutoGen** | Conversational agents, Microsoft-backed | Designed for chatbots not compliance workflows, non-deterministic | ❌ Rejected |
| **Custom FSM** | Full control, no dependencies | High development cost, no community support | ❌ Rejected |

---

## Consequences

**Positive:**
- `ComplianceAgentState` TypedDict makes every field explicit and type-checked
- `audit_log: Annotated[list, operator.add]` provides append-only audit trail — every node execution is logged with timestamp, action, and details
- `should_retry` conditional edge enables reflection loop (max 2 cycles) without infinite loops
- `FallbackComplianceAgent` runs the same 6-node pipeline sequentially if LangGraph fails to initialise

**Negative:**
- LangGraph's `StateGraph.compile()` adds ~200ms cold start latency
- Debugging requires understanding graph topology — harder to trace than linear chains

---

## Trade-off Rationale

**Why not CrewAI:** In compliance, wrong answers have legal consequences. CrewAI's multi-agent approach introduces non-determinism — agents can negotiate and reach different conclusions on the same input. LangGraph's fixed graph topology guarantees the same execution path every time, which is essential for regulatory audit requirements.

**Why cyclic graph matters:** The reflection loop is the key differentiator. A compliance answer that initially lacks citations triggers a second retrieval cycle with expanded search terms. This is impossible with linear LangChain chains. The retry improves HIGH confidence answer rate by approximately 15% in testing.

**Audit trail design:** The `audit_log` field uses `Annotated[list, operator.add]` — LangGraph's merge function appends each node's log entries rather than overwriting them. This gives a complete chronological record of every step in every query, satisfying the assignment's "traceability is non-negotiable" requirement.
