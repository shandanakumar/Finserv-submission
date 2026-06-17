"""
app.py — FinServ Compliance Assistant — Streamlit UI
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:8000"
API_KEY = "dev-key-1234"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

st.set_page_config(
    page_title="FinServ Compliance Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
    st.title("FinServ Compliance")
    st.caption("AI-powered regulatory assistant")
    st.divider()
    page = st.radio(
        "Navigate",
        ["💬 Ask a Question", "📄 Document Explorer", "⚙️ System Info"],
        label_visibility="collapsed",
    )
    st.divider()
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("API Server: Online", icon="✅")
        else:
            st.error("API Server: Error")
    except Exception:
        st.error("API Server: Offline", icon="🔴")
        st.caption("Start with: uvicorn src.api.main:app --port 8000")
    st.divider()
    st.caption("Regulations covered:")
    st.caption("🌍 Basel III (BIS)")
    st.caption("🇪🇺 MiFID II (ESMA)")
    st.caption("🇮🇳 RBI KYC / PSL")
    st.caption("🌍 FATF AML/CFT")


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — Ask a Question
# ═══════════════════════════════════════════════════════════════
if page == "💬 Ask a Question":
    st.title("💬 Regulatory Compliance Q&A")
    st.caption("Ask any question about Basel III, MiFID II, RBI KYC, RBI PSL, or FATF guidelines")

    # ── Sample question buttons ───────────────────────────────
    st.subheader("Quick questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏦 Basel III CET1 ratio", use_container_width=True):
            st.session_state["last_query"] = "What is the minimum CET1 capital ratio under Basel III?"
            st.session_state["last_reg"]   = "BASEL_III — Capital Requirements"
        if st.button("💧 Liquidity LCR", use_container_width=True):
            st.session_state["last_query"] = "What is the minimum Liquidity Coverage Ratio under Basel III?"
            st.session_state["last_reg"]   = "BASEL_III — Capital Requirements"
    with col2:
        if st.button("🔍 RBI KYC high risk", use_container_width=True):
            st.session_state["last_query"] = "What is the periodic KYC updation frequency for high risk customers?"
            st.session_state["last_reg"]   = "RBI_KYC — Know Your Customer"
        if st.button("🌾 PSL agriculture", use_container_width=True):
            st.session_state["last_query"] = "What is the agriculture PSL target for commercial banks?"
            st.session_state["last_reg"]   = "RBI_PSL — Priority Sector Lending"
    with col3:
        if st.button("📋 MiFID II suitability", use_container_width=True):
            st.session_state["last_query"] = "When is a suitability assessment mandatory under MiFID II?"
            st.session_state["last_reg"]   = "MIFID_II — Markets in Financial Instruments"
        if st.button("🚨 FATF suspicious tx", use_container_width=True):
            st.session_state["last_query"] = "What are FATF requirements for suspicious transaction reporting?"
            st.session_state["last_reg"]   = "FATF_AML — Anti-Money Laundering"

    st.divider()

    # ── Regulation filter ─────────────────────────────────────
    reg_options = [
        "Auto-detect",
        "BASEL_III — Capital Requirements",
        "RBI_KYC — Know Your Customer",
        "RBI_PSL — Priority Sector Lending",
        "MIFID_II — Markets in Financial Instruments",
        "FATF_AML — Anti-Money Laundering",
    ]

    last_reg = st.session_state.get("last_reg", "Auto-detect")
    default_reg_index = reg_options.index(last_reg) if last_reg in reg_options else 0

    col_reg, col_ver = st.columns(2)
    with col_reg:
        regulation = st.selectbox(
            "Filter by regulation",
            reg_options,
            index=default_reg_index,
            help="Select a regulation to narrow search, or Auto-detect",
        )
    with col_ver:
        st.selectbox(
            "Version",
            ["Active only (recommended)", "All versions including superseded"],
        )

    # Parse regulation selection
    selected_reg = None
    if regulation != "Auto-detect":
        selected_reg = regulation.split(" — ")[0]

    # ── Query input ───────────────────────────────────────────
    query = st.text_area(
        "Your question",
        value=st.session_state.get("last_query", ""),
        height=100,
        placeholder="e.g. What is the minimum CET1 capital ratio under Basel III?",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        submit = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col_info:
        if selected_reg:
            st.caption(f"🎯 Searching: **{selected_reg}** · Powered by Mistral 7B on AWS Bedrock")
        else:
            st.caption("Powered by Mistral 7B on AWS Bedrock · Retrieved from real regulatory PDFs")

    # ── Submit ────────────────────────────────────────────────
    if submit and query.strip():
        st.session_state["last_query"] = query

        with st.spinner("Searching regulatory documents and generating answer..."):
            t0 = time.time()
            try:
                request_body = {"query": query}
                if selected_reg:
                    request_body["regulation_families"] = [selected_reg]

                response = requests.post(
                    f"{API_URL}/query",
                    headers=HEADERS,
                    json=request_body,
                    timeout=120,
                )
                elapsed = time.time() - t0

                if response.status_code == 200:
                    data = response.json()

                    st.divider()
                    conf       = data.get("confidence", "MEDIUM")
                    conf_icon  = {"HIGH": "✅", "MEDIUM": "⚠️", "LOW": "❌"}.get(conf, "❓")

                    col_conf, col_time, col_review = st.columns(3)
                    with col_conf:
                        st.metric("Confidence", f"{conf_icon} {conf}")
                    with col_time:
                        st.metric("Response time", f"{data.get('processing_time_ms', elapsed*1000):.0f}ms")
                    with col_review:
                        review = data.get("requires_human_review", False)
                        st.metric("Human Review", "Required ⚠️" if review else "Not needed ✅")

                    st.subheader("Answer")
                    st.markdown(data.get("answer", "No answer returned"))

                    regs = data.get("applicable_regulations", [])
                    if regs:
                        st.subheader("Regulations referenced")
                        for reg in regs:
                            st.badge(reg)

                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].append({
                        "time":       datetime.now().strftime("%H:%M:%S"),
                        "query":      query[:80],
                        "regulation": selected_reg or "Auto",
                        "confidence": conf,
                        "ms":         f"{elapsed*1000:.0f}ms",
                    })

                else:
                    st.error(f"API error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Run: uvicorn src.api.main:app --port 8000")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Session history ───────────────────────────────────────
    if st.session_state.get("history"):
        st.divider()
        st.subheader("Session history")
        for h in reversed(st.session_state["history"][-5:]):
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            with c1:
                st.caption(f"🕐 {h['time']} — {h['query']}")
            with c2:
                st.caption(h.get("regulation", "Auto"))
            with c3:
                st.caption(h["confidence"])
            with c4:
                st.caption(h["ms"])


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — Document Explorer
# ═══════════════════════════════════════════════════════════════
elif page == "📄 Document Explorer":
    st.title("📄 Ingested Regulatory Documents")
    st.caption("Real PDFs parsed, chunked, and embedded into Qdrant vector store")

    try:
        with open("outputs/ingestion_summary.json") as f:
            summary = json.load(f)

        total_chunks = summary.get("total_chunks", 0)
        docs         = summary.get("documents", [])
        ingested_at  = summary.get("ingested_at", "Unknown")

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Documents", len(docs))
        with col2: st.metric("Total Chunks", f"{total_chunks:,}")
        with col3: st.metric("Embedding Model", "Titan v2")
        with col4: st.metric("Vector Dimensions", "1024")

        st.divider()
        st.subheader("Documents in knowledge base")

        for doc in docs:
            version = doc.get("version", "unknown")
            status  = doc.get("status", "active")
            status_icon = "✅" if status == "active" else "⚠️ superseded"
            with st.expander(
                f"📘 {doc['doc_id']} — {doc['regulation_family']} "
                f"v{version} {status_icon} ({doc['chunk_count']} chunks)",
                expanded=False,
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Source:** {doc['source']}")
                    st.write(f"**Regulation:** {doc['regulation_family']}")
                    st.write(f"**Jurisdiction:** {doc['jurisdiction']}")
                    st.write(f"**Version:** {version}")
                    st.write(f"**Status:** {status}")
                with c2:
                    st.write(f"**Characters:** {doc['char_count']:,}")
                    st.write(f"**Chunks:** {doc['chunk_count']}")
                    st.write(f"**Title:** {doc['title'][:60]}...")

        st.divider()
        st.subheader("Chunk distribution")
        try:
            import plotly.express as px
            import pandas as pd

            df = pd.DataFrame([
                {
                    "Document":   d["doc_id"][:25],
                    "Chunks":     d["chunk_count"],
                    "Regulation": d["regulation_family"],
                    "Status":     d.get("status", "active"),
                }
                for d in docs
            ])
            fig = px.bar(
                df, x="Document", y="Chunks",
                color="Regulation",
                pattern_shape="Status",
                title="Chunks per document (striped = superseded)",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("pip install plotly pandas")

        st.caption(f"Last ingested: {ingested_at}")

    except FileNotFoundError:
        st.warning("No ingestion summary found. Run: python scripts\\ingest_pdfs.py")


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — System Info
# ═══════════════════════════════════════════════════════════════
elif page == "⚙️ System Info":
    st.title("⚙️ System Architecture")

    st.subheader("Pipeline overview")
    st.code("""
User Question
     ↓
FastAPI (/query endpoint) — API key auth
     ↓
LangGraph Agent
  ├─ Router Node      — classifies intent (Q&A / screening / report)
  ├─ RAG Retrieval    — hybrid search: dense (Titan) + BM25 → RRF fusion
  │                     version-aware filter: is_superseded=False
  │                     regulation filter: passed from UI selectbox
  ├─ Reranker         — keyword overlap scoring → top 5 chunks
  ├─ Synthesis        — Mistral 7B on Bedrock generates cited answer
  ├─ Reflection       — self-evaluates quality, retries up to 2x
  └─ Guardrails       — PII redaction + citation validation
     ↓
Structured JSON response
    """, language="text")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Components")
        st.markdown("""
| Component | Technology |
|---|---|
| LLM (primary) | Mistral 7B — AWS Bedrock |
| LLM (complex) | Mixtral 8×7B — AWS Bedrock |
| Embeddings | Amazon Titan v2 (1024 dim) |
| Vector store | Qdrant (disk persistence) |
| Orchestration | LangGraph |
| API | FastAPI |
| UI | Streamlit |
| Chunking | Hierarchical Semantic |
        """)

    with col2:
        st.subheader("Knowledge base")
        st.markdown("""
| Regulation | Versions | Status |
|---|---|---|
| Basel III | v1_2010 + v2_2017 | v2 active |
| MiFID II | v1_2014 | active |
| RBI KYC | v1_2016 + v2_2025 | v2 active |
| RBI PSL | v1_2020 + v2_2025 | v2 active |
| FATF AML | v1_2012 | active |
        """)

    st.divider()
    st.subheader("Evaluation metrics (RAGAS)")
    st.markdown("""
| Metric | Target | Meaning |
|---|---|---|
| Faithfulness | ≥ 0.80 | No hallucination — answer from context only |
| Answer Relevance | ≥ 0.75 | Answer addresses the question |
| Context Precision | ≥ 0.75 | Retrieved chunks are relevant |
| Context Recall | ≥ 0.70 | All needed info was retrieved |
    """)