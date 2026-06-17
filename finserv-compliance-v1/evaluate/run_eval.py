"""
src/evaluation/run_eval.py

RAGAS-style evaluation framework for FinServ Compliance Assistant.
Measures: Faithfulness, Answer Relevance, Context Precision, Context Recall

HOW TO RUN:
    # Make sure server is running first:
    uvicorn src.api.main:app --reload --port 8000

    # Then run evaluation:
    python src/evaluation/run_eval.py

OUTPUT:
    outputs/eval_report.json  — full results
    outputs/eval_report.md    — submission-ready report
"""

import sys
import os
import json
import time
import re
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

os.environ.update({
    "LLM_BACKEND":             "bedrock",
    "AWS_REGION":              "us-east-1",
    "QDRANT_MODE":             "disk",
    "BEDROCK_PRIMARY_MODEL":   "mistral.mistral-7b-instruct-v0:2",
    "BEDROCK_EMBEDDING_MODEL": "amazon.titan-embed-text-v2:0",
    "EMBEDDING_DIMENSION":     "1024",
    "QDRANT_COLLECTION":       "regulatory_docs",
})

# ── Test dataset — 20 Q&A pairs with ground truth ────────────
TEST_DATASET = [
    # Basel III — 5 questions
    {
        "id": "BASEL_001",
        "regulation": "BASEL_III",
        "question": "What is the minimum CET1 capital ratio under Basel III?",
        "ground_truth": "The minimum Common Equity Tier 1 (CET1) capital ratio under Basel III is 4.5% of risk-weighted assets.",
        "key_terms": ["4.5", "CET1", "Common Equity Tier 1", "risk-weighted"],
    },
    {
        "id": "BASEL_002",
        "regulation": "BASEL_III",
        "question": "What is the Capital Conservation Buffer requirement under Basel III?",
        "ground_truth": "The Capital Conservation Buffer is set at 2.5% of risk-weighted assets, in addition to the minimum CET1 ratio of 4.5%.",
        "key_terms": ["2.5", "conservation buffer", "risk-weighted"],
    },
    {
        "id": "BASEL_003",
        "regulation": "BASEL_III",
        "question": "What is the minimum Tier 1 capital ratio under Basel III?",
        "ground_truth": "The minimum Tier 1 capital ratio is 6% of risk-weighted assets under Basel III.",
        "key_terms": ["6%", "Tier 1", "risk-weighted"],
    },
    {
        "id": "BASEL_004",
        "regulation": "BASEL_III",
        "question": "What is the leverage ratio requirement under Basel III?",
        "ground_truth": "The minimum leverage ratio requirement under Basel III is 3% for internationally active banks.",
        "key_terms": ["3%", "leverage ratio"],
    },
    {
        "id": "BASEL_005",
        "regulation": "BASEL_III",
        "question": "What triggers capital distribution restrictions under Basel III conservation buffer?",
        "ground_truth": "Breaching the Capital Conservation Buffer triggers automatic restrictions on dividends, share buybacks, and discretionary bonus payments.",
        "key_terms": ["dividend", "restrictions", "conservation buffer", "buyback"],
    },
    # RBI KYC — 5 questions
    {
        "id": "KYC_001",
        "regulation": "RBI_KYC",
        "question": "What is the periodic KYC updation frequency for high risk customers under RBI?",
        "ground_truth": "High-risk customers must have their KYC records updated at least once every 2 years.",
        "key_terms": ["2 years", "high risk", "periodic", "updation"],
    },
    {
        "id": "KYC_002",
        "regulation": "RBI_KYC",
        "question": "What is the KYC updation frequency for medium risk customers?",
        "ground_truth": "Medium-risk customers must have their KYC records updated at least once every 8 years.",
        "key_terms": ["8 years", "medium risk", "periodic"],
    },
    {
        "id": "KYC_003",
        "regulation": "RBI_KYC",
        "question": "What is Enhanced Due Diligence and when is it required under RBI KYC?",
        "ground_truth": "Enhanced Due Diligence (EDD) is mandatory for HIGH-risk customers including Politically Exposed Persons (PEPs) and entities in high-risk jurisdictions.",
        "key_terms": ["Enhanced Due Diligence", "EDD", "high risk", "PEP"],
    },
    {
        "id": "KYC_004",
        "regulation": "RBI_KYC",
        "question": "What are the four key elements of KYC policy under RBI directions?",
        "ground_truth": "The four key elements of KYC policy are: Customer Acceptance Policy, Customer Identification Procedures, Monitoring of Transactions, and Risk Management.",
        "key_terms": ["Customer Acceptance", "Customer Identification", "Monitoring", "Risk Management"],
    },
    {
        "id": "KYC_005",
        "regulation": "RBI_KYC",
        "question": "What is the KYC updation frequency for low risk customers under RBI?",
        "ground_truth": "Low-risk customers must have their KYC records updated at least once every 10 years.",
        "key_terms": ["10 years", "low risk", "updation"],
    },
    # RBI PSL — 3 questions
    {
        "id": "PSL_001",
        "regulation": "RBI_PSL",
        "question": "What is the agriculture PSL target for domestic commercial banks?",
        "ground_truth": "The agriculture Priority Sector Lending target is 18% of ANBC or CEOBSE, whichever is higher.",
        "key_terms": ["18", "agriculture", "ANBC", "CEOBSE"],
    },
    {
        "id": "PSL_002",
        "regulation": "RBI_PSL",
        "question": "What is the overall Priority Sector Lending target for domestic commercial banks?",
        "ground_truth": "The overall Priority Sector Lending target is 40% of ANBC or CEOBSE, whichever is higher.",
        "key_terms": ["40", "overall", "priority sector", "ANBC"],
    },
    {
        "id": "PSL_003",
        "regulation": "RBI_PSL",
        "question": "What is the Micro Enterprises PSL sub-target?",
        "ground_truth": "The Micro Enterprises sub-target is 7.5% of ANBC or CEOBSE, whichever is higher.",
        "key_terms": ["7.5", "micro enterprises", "ANBC"],
    },
    # MiFID II — 4 questions
    {
        "id": "MIFID_001",
        "regulation": "MIFID_II",
        "question": "When is a suitability assessment mandatory under MiFID II?",
        "ground_truth": "A suitability assessment is mandatory under MiFID II when an investment firm provides a personal recommendation or portfolio management service to a client.",
        "key_terms": ["suitability", "personal recommendation", "portfolio management"],
    },
    {
        "id": "MIFID_002",
        "regulation": "MIFID_II",
        "question": "What is the best execution obligation under MiFID II?",
        "ground_truth": "Investment firms must take all sufficient steps to obtain the best possible result for clients when executing orders, taking into account price, costs, speed, likelihood of execution and settlement.",
        "key_terms": ["best execution", "best possible result", "price", "costs"],
    },
    {
        "id": "MIFID_003",
        "regulation": "MIFID_II",
        "question": "What are the requirements for the management body of a market operator under MiFID II?",
        "ground_truth": "Members of the management body must be of sufficiently good repute, possess sufficient knowledge, skills and experience to perform their duties.",
        "key_terms": ["management body", "repute", "knowledge", "skills", "experience"],
    },
    {
        "id": "MIFID_004",
        "regulation": "MIFID_II",
        "question": "What client categories exist under MiFID II?",
        "ground_truth": "MiFID II categorises clients into three categories: Retail clients, Professional clients, and Eligible counterparties, each with different levels of protection.",
        "key_terms": ["retail", "professional", "eligible counterparties", "client categories"],
    },
    # FATF — 3 questions
    {
        "id": "FATF_001",
        "regulation": "FATF_AML",
        "question": "What are FATF requirements for suspicious transaction reporting?",
        "ground_truth": "Under FATF Recommendation 20, financial institutions must report suspicious transactions to the Financial Intelligence Unit when they suspect funds are proceeds of crime or linked to terrorist financing.",
        "key_terms": ["suspicious transaction", "FIU", "Recommendation 20", "terrorist financing"],
    },
    {
        "id": "FATF_002",
        "regulation": "FATF_AML",
        "question": "What is the FATF risk-based approach to AML?",
        "ground_truth": "The FATF risk-based approach requires countries and financial institutions to identify, assess and understand their money laundering and terrorist financing risks and take measures proportionate to those risks.",
        "key_terms": ["risk-based", "identify", "assess", "money laundering", "terrorist financing"],
    },
    {
        "id": "FATF_003",
        "regulation": "FATF_AML",
        "question": "What customer due diligence measures are required under FATF?",
        "ground_truth": "FATF requires customer due diligence including: identifying and verifying customer identity, identifying beneficial ownership, understanding business relationships, and ongoing monitoring.",
        "key_terms": ["customer due diligence", "CDD", "beneficial ownership", "ongoing monitoring"],
    },
]


# ── Evaluation metrics ────────────────────────────────────────

def compute_faithfulness(answer: str, context: str, key_terms: list) -> float:
    """
    Faithfulness: does the answer only use information from the context?
    Measures: what fraction of key terms in the answer appear in the context.
    Score 0.0 - 1.0
    """
    if not answer or not context:
        return 0.0

    answer_lower  = answer.lower()
    context_lower = context.lower()

    # Check how many claims in answer are supported by context
    supported = 0
    total     = 0

    # Split answer into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]

    for sentence in sentences[:5]:  # check first 5 sentences
        total += 1
        sent_lower = sentence.lower()
        # Check if any meaningful words from sentence appear in context
        words = [w for w in sent_lower.split() if len(w) > 4]
        matches = sum(1 for w in words if w in context_lower)
        if matches >= len(words) * 0.4:  # 40% word overlap = supported
            supported += 1

    if total == 0:
        return 0.5

    return round(supported / total, 3)


def compute_answer_relevance(answer: str, question: str, key_terms: list) -> float:
    """
    Answer Relevance: does the answer address the question?
    Measures: what fraction of key terms from question appear in answer.
    Score 0.0 - 1.0
    """
    if not answer or not question:
        return 0.0

    answer_lower = answer.lower()

    # Check key terms from ground truth
    found = sum(1 for term in key_terms if term.lower() in answer_lower)

    if not key_terms:
        return 0.5

    base_score = found / len(key_terms)

    # Penalise if answer says "insufficient context" or "not provided"
    penalty_phrases = [
        "insufficient_context", "not provided", "does not provide",
        "cannot answer", "not mentioned", "no information",
        "context does not", "not explicitly"
    ]
    if any(p in answer_lower for p in penalty_phrases):
        base_score *= 0.3

    return round(min(base_score, 1.0), 3)


def compute_context_precision(answer: str, retrieved_context: str, key_terms: list) -> float:
    """
    Context Precision: were the retrieved chunks actually relevant?
    Measures: what fraction of retrieved context is relevant to the question.
    Score 0.0 - 1.0
    """
    if not retrieved_context:
        return 0.0

    context_lower = retrieved_context.lower()
    found = sum(1 for term in key_terms if term.lower() in context_lower)

    if not key_terms:
        return 0.5

    return round(min(found / len(key_terms), 1.0), 3)


def compute_context_recall(ground_truth: str, retrieved_context: str, key_terms: list) -> float:
    """
    Context Recall: did retrieved chunks contain all info needed to answer?
    Measures: what fraction of ground truth key terms appear in retrieved context.
    Score 0.0 - 1.0
    """
    if not retrieved_context or not ground_truth:
        return 0.0

    context_lower = retrieved_context.lower()
    found = sum(1 for term in key_terms if term.lower() in context_lower)

    if not key_terms:
        return 0.5

    return round(min(found / len(key_terms), 1.0), 3)


# ── API call ──────────────────────────────────────────────────

def query_api(question: str, regulation: str, timeout: int = 60) -> dict:
    """Call agent directly — no API server needed."""
    try:
        from src.agent.compliance_agent import FallbackComplianceAgent
        agent  = FallbackComplianceAgent()
        result = agent.run(question, regulation_filter=regulation)
        return {
            "answer":     result.get("answer", ""),
            "confidence": result.get("confidence", "LOW"),
        }
    except Exception as e:
        return {"error": str(e), "answer": "", "confidence": "LOW"}

def get_retrieved_context(question: str, regulation: str) -> str:
    """Get the raw context that was retrieved for scoring."""
    try:
        import boto3
        import json as _json

        os.environ.setdefault("QDRANT_MODE", "disk")
        os.environ.setdefault("QDRANT_COLLECTION", "regulatory_docs")
        os.environ.setdefault("EMBEDDING_DIMENSION", "1024")

        BEDROCK = boto3.client("bedrock-runtime", region_name="us-east-1")

        from src.llm import bedrock_client as bc
        def _embed(self, text):
            resp = BEDROCK.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=_json.dumps({"inputText": text[:8000]}),
                contentType="application/json", accept="application/json",
            )
            return _json.loads(resp["body"].read())["embedding"]
        def _init(self):
            self.client = bc.get_bedrock_client()
            self.model_id = "amazon.titan-embed-text-v2:0"
            self.dimensions = 1024
        bc.BedrockEmbedder.__init__ = _init
        bc.BedrockEmbedder.embed = _embed

        from src.retrieval.hybrid_search import HybridSearcher
        from src.retrieval.reranker import CrossEncoderReranker, format_chunks_for_llm

        searcher = HybridSearcher()
        reranker = CrossEncoderReranker()
        chunks   = searcher.search(question, top_k=5, regulation_families=[regulation])
        reranked = reranker.rerank(question, chunks, top_k=3)
        return format_chunks_for_llm(reranked)

    except Exception as e:
        return f"Context retrieval failed: {e}"


# ── Main evaluation loop ──────────────────────────────────────

def run_evaluation():
    print("\n" + "="*62)
    print("  FinServ Compliance Assistant — RAGAS Evaluation")
    print("="*62)
    print(f"  Questions: {len(TEST_DATASET)}")
    print(f"  Regulations: Basel III, RBI KYC, RBI PSL, MiFID II, FATF")
    print(f"  Metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*62)

    results     = []
    total_time  = 0

    for i, test in enumerate(TEST_DATASET, 1):
        print(f"\n[{i:02d}/{len(TEST_DATASET)}] {test['id']}: {test['question'][:60]}...")

        # Get answer from API
        t0     = time.time()
        result = query_api(test["question"], test["regulation"])
        elapsed = time.time() - t0
        total_time += elapsed

        answer     = result.get("answer", "")
        confidence = result.get("confidence", "LOW")
        error      = result.get("error", None)

        if error:
            print(f"  ERROR: {error}")
            results.append({**test, "answer": "", "confidence": "ERROR",
                           "faithfulness": 0, "answer_relevance": 0,
                           "context_precision": 0, "context_recall": 0,
                           "latency_ms": int(elapsed * 1000), "error": error})
            continue

        # Get retrieved context for scoring
        context = get_retrieved_context(test["question"], test["regulation"])

        # Compute metrics
        faithfulness      = compute_faithfulness(answer, context, test["key_terms"])
        answer_relevance  = compute_answer_relevance(answer, test["question"], test["key_terms"])
        context_precision = compute_context_precision(answer, context, test["key_terms"])
        context_recall    = compute_context_recall(test["ground_truth"], context, test["key_terms"])

        result_entry = {
            "id":                test["id"],
            "regulation":        test["regulation"],
            "question":          test["question"],
            "ground_truth":      test["ground_truth"],
            "answer":            answer[:500],
            "confidence":        confidence,
            "faithfulness":      faithfulness,
            "answer_relevance":  answer_relevance,
            "context_precision": context_precision,
            "context_recall":    context_recall,
            "latency_ms":        int(elapsed * 1000),
        }
        results.append(result_entry)

        print(f"  Confidence: {confidence}")
        print(f"  Faithfulness: {faithfulness:.2f} | Relevance: {answer_relevance:.2f} | Precision: {context_precision:.2f} | Recall: {context_recall:.2f}")
        print(f"  Latency: {int(elapsed*1000)}ms")

    # ── Aggregate scores ──────────────────────────────────────
    valid = [r for r in results if "error" not in r]

    avg_faithfulness      = sum(r["faithfulness"]      for r in valid) / max(len(valid), 1)
    avg_answer_relevance  = sum(r["answer_relevance"]  for r in valid) / max(len(valid), 1)
    avg_context_precision = sum(r["context_precision"] for r in valid) / max(len(valid), 1)
    avg_context_recall    = sum(r["context_recall"]    for r in valid) / max(len(valid), 1)
    avg_latency           = sum(r["latency_ms"]        for r in valid) / max(len(valid), 1)

    high_conf  = sum(1 for r in valid if r["confidence"] == "HIGH")
    med_conf   = sum(1 for r in valid if r["confidence"] == "MEDIUM")
    low_conf   = sum(1 for r in valid if r["confidence"] == "LOW")

    summary = {
        "evaluation_date":      datetime.now().isoformat(),
        "model":                "mistral.mistral-7b-instruct-v0:2",
        "embedding_model":      "amazon.titan-embed-text-v2:0",
        "total_questions":      len(TEST_DATASET),
        "successful":           len(valid),
        "avg_faithfulness":     round(avg_faithfulness, 3),
        "avg_answer_relevance": round(avg_answer_relevance, 3),
        "avg_context_precision":round(avg_context_precision, 3),
        "avg_context_recall":   round(avg_context_recall, 3),
        "avg_latency_ms":       round(avg_latency, 1),
        "confidence_distribution": {"HIGH": high_conf, "MEDIUM": med_conf, "LOW": low_conf},
        "results":              results,
    }

    # Print summary
    print("\n" + "="*62)
    print("  EVALUATION SUMMARY")
    print("="*62)
    print(f"  Questions evaluated:   {len(valid)}/{len(TEST_DATASET)}")
    print(f"  Faithfulness:          {avg_faithfulness:.3f}  (target ≥ 0.80)")
    print(f"  Answer Relevance:      {avg_answer_relevance:.3f}  (target ≥ 0.75)")
    print(f"  Context Precision:     {avg_context_precision:.3f}  (target ≥ 0.75)")
    print(f"  Context Recall:        {avg_context_recall:.3f}  (target ≥ 0.70)")
    print(f"  Avg Latency:           {avg_latency:.0f}ms")
    print(f"  High confidence:       {high_conf}/{len(valid)}")
    print(f"  Total eval time:       {total_time:.0f}s")

    # Save JSON
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: outputs/eval_report.json")

    # Generate Markdown report
    md = generate_markdown_report(summary)
    with open("outputs/eval_report.md", "w") as f:
        f.write(md)
    print(f"  Saved: outputs/eval_report.md")

    return summary


def generate_markdown_report(summary: dict) -> str:
    r = summary
    lines = [
        "# FinServ Compliance Assistant — Evaluation Report",
        "",
        f"**Date:** {r['evaluation_date'][:10]}",
        f"**Model:** {r['model']}",
        f"**Embedding:** {r['embedding_model']}",
        f"**Questions:** {r['total_questions']} across Basel III, RBI KYC, RBI PSL, MiFID II, FATF AML",
        "",
        "## Summary Scores",
        "",
        "| Metric | Score | Target | Status |",
        "|---|---|---|---|",
        f"| Faithfulness | {r['avg_faithfulness']:.3f} | ≥ 0.80 | {'✅ PASS' if r['avg_faithfulness'] >= 0.80 else '⚠️ BELOW TARGET'} |",
        f"| Answer Relevance | {r['avg_answer_relevance']:.3f} | ≥ 0.75 | {'✅ PASS' if r['avg_answer_relevance'] >= 0.75 else '⚠️ BELOW TARGET'} |",
        f"| Context Precision | {r['avg_context_precision']:.3f} | ≥ 0.75 | {'✅ PASS' if r['avg_context_precision'] >= 0.75 else '⚠️ BELOW TARGET'} |",
        f"| Context Recall | {r['avg_context_recall']:.3f} | ≥ 0.70 | {'✅ PASS' if r['avg_context_recall'] >= 0.70 else '⚠️ BELOW TARGET'} |",
        f"| Avg Latency | {r['avg_latency_ms']:.0f}ms | ≤ 10000ms | {'✅ PASS' if r['avg_latency_ms'] <= 10000 else '⚠️ ABOVE TARGET'} |",
        "",
        "## Confidence Distribution",
        "",
        f"- HIGH confidence: {r['confidence_distribution']['HIGH']}/{r['total_questions']}",
        f"- MEDIUM confidence: {r['confidence_distribution']['MEDIUM']}/{r['total_questions']}",
        f"- LOW confidence: {r['confidence_distribution']['LOW']}/{r['total_questions']}",
        "",
        "## Per-Question Results",
        "",
        "| ID | Regulation | Confidence | Faith. | Relevance | Precision | Recall | Latency |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for res in r["results"]:
        if "error" in res:
            lines.append(f"| {res['id']} | {res['regulation']} | ERROR | - | - | - | - | - |")
        else:
            conf_icon = {"HIGH": "✅", "MEDIUM": "⚠️", "LOW": "❌"}.get(res["confidence"], "❓")
            lines.append(
                f"| {res['id']} | {res['regulation']} | {conf_icon} {res['confidence']} "
                f"| {res['faithfulness']:.2f} | {res['answer_relevance']:.2f} "
                f"| {res['context_precision']:.2f} | {res['context_recall']:.2f} "
                f"| {res['latency_ms']}ms |"
            )

    lines += [
        "",
        "## Failure Analysis",
        "",
        "### Known Limitations",
        "1. **Table extraction** — RBI PSL and KYC PDFs contain structured tables where pdfminer loses column associations. Mitigated with camelot-py table extraction and clean reference text documents.",
        "2. **Cross-regulation vocabulary** — BM25 keyword search can match Basel III chunks for RBI queries due to shared terms (risk, years, requirements). Mitigated with explicit regulation filter in retrieval.",
        "3. **Hindi text encoding** — RBI PSL 2025 PDF contains Devanagari characters that pdfminer cannot extract. Supplemented with clean English reference document.",
        "",
        "### Production Improvements",
        "1. Use RAGAS library with LLM-as-judge for more precise faithfulness scoring",
        "2. Increase test dataset to 50+ questions with human-verified ground truth",
        "3. Add adversarial test cases (trick questions, out-of-scope queries)",
        "4. Implement automated drift detection — re-run eval monthly after new document ingestion",
        "",
        "## Metric Definitions",
        "",
        "- **Faithfulness**: Fraction of answer claims supported by retrieved context. Detects hallucination.",
        "- **Answer Relevance**: Fraction of ground truth key terms present in the answer. Measures directness.",
        "- **Context Precision**: Fraction of key terms in retrieved context. Measures retrieval noise.",
        "- **Context Recall**: Fraction of ground truth key terms found in retrieved context. Measures completeness.",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    run_evaluation()
