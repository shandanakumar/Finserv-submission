"""
src/evaluation/evaluator.py

RAGAS-based evaluation framework for the Regulatory Compliance RAG pipeline.

Metrics evaluated:
- Faithfulness: Are all claims in the answer supported by the retrieved context?
- Answer Relevance: Does the answer address the actual question asked?
- Context Precision: Are the retrieved chunks relevant to the question?
- Context Recall: Does the retrieved context cover all information needed to answer?

Evaluation approach:
- Uses RAGAS library (open-source) for standard RAG evaluation metrics
- Falls back to a lightweight custom implementation if RAGAS not available
- Runs against the 20-question ground truth dataset
- Outputs a structured JSON report with per-question scores and failure analysis

Note on judge model:
RAGAS uses an LLM as a judge. We use Mistral 7B (local Ollama) as the judge
to maintain data residency compliance. Scores may be lower than GPT-4-judged
evaluations but are consistent and comparable across runs.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import statistics

logger = logging.getLogger(__name__)


class CustomRAGEvaluator:
    """
    Lightweight custom evaluator when RAGAS is not available.
    Uses heuristic-based scoring rather than LLM-as-judge.
    Results are approximate but useful for comparative evaluation.
    """

    def compute_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        Heuristic faithfulness: ratio of answer sentences that share
        significant n-gram overlap with the retrieved contexts.
        """
        if not answer or not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]

        if not answer_sentences:
            return 0.0

        faithful_count = 0
        for sentence in answer_sentences:
            words = set(sentence.lower().split())
            # Remove stopwords
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                        "have", "has", "had", "do", "does", "did", "will", "would", "could",
                        "should", "may", "might", "shall", "of", "to", "in", "for", "on",
                        "with", "at", "by", "from", "as", "or", "and", "but", "if", "this", "that"}
            content_words = words - stopwords

            if len(content_words) < 3:
                faithful_count += 1  # Short sentences get benefit of doubt
                continue

            # Check what fraction of content words appear in context
            found_words = sum(1 for w in content_words if w in combined_context)
            overlap = found_words / len(content_words)
            if overlap >= 0.5:
                faithful_count += 1

        return faithful_count / len(answer_sentences)

    def compute_answer_relevance(self, question: str, answer: str) -> float:
        """
        Heuristic answer relevance: keyword overlap between question and answer.
        """
        if not question or not answer:
            return 0.0

        # Check for "I don't know" or insufficient context responses
        insufficient_markers = ["insufficient_context", "cannot answer", "not found", "please consult"]
        if any(m in answer.lower() for m in insufficient_markers):
            return 0.3  # Partial credit for appropriate refusal

        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())

        stopwords = {"what", "when", "where", "who", "why", "how", "is", "are", "the", "a", "an", "for", "in", "of", "to"}
        q_content = q_words - stopwords
        a_content = a_words - stopwords

        if not q_content:
            return 0.5

        overlap = len(q_content & a_content) / len(q_content)
        # Scale to 0.3-1.0 range (even good answers don't repeat every question word)
        return min(1.0, 0.3 + overlap * 0.7)

    def compute_context_precision(self, question: str, contexts: list[str]) -> float:
        """
        Heuristic context precision: what fraction of retrieved contexts are relevant?
        """
        if not contexts:
            return 0.0

        q_words = set(question.lower().split())
        relevant_count = 0

        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(q_words & ctx_words) / len(q_words) if q_words else 0
            if overlap > 0.2:
                relevant_count += 1

        return relevant_count / len(contexts)

    def compute_context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """
        Heuristic context recall: how much of the ground truth info appears in retrieved context?
        """
        if not ground_truth or not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()
        gt_sentences = [s.strip() for s in ground_truth.split('.') if s.strip()]

        if not gt_sentences:
            return 0.0

        covered_count = 0
        for sentence in gt_sentences:
            words = set(sentence.lower().split())
            content_words = {w for w in words if len(w) > 3}  # Skip short words
            if not content_words:
                covered_count += 1
                continue
            found = sum(1 for w in content_words if w in combined_context)
            if found / len(content_words) >= 0.4:
                covered_count += 1

        return covered_count / len(gt_sentences)


class RAGASEvaluator:
    """
    RAGAS-based evaluation (preferred if RAGAS is installed).
    Uses Mistral 7B via Ollama as the LLM judge for data residency compliance.
    """

    def __init__(self, llm_model: str = "mistral", ollama_url: str = "http://localhost:11434"):
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self._ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        try:
            import ragas
            return True
        except ImportError:
            logger.info("RAGAS not installed — using custom evaluator")
            return False

    def evaluate_dataset(self, eval_data: list[dict]) -> dict:
        """
        Evaluate a list of {question, answer, contexts, ground_truth} dicts.
        Returns aggregated metrics.
        """
        if self._ragas_available:
            return self._evaluate_ragas(eval_data)
        return self._evaluate_custom(eval_data)

    def _evaluate_ragas(self, eval_data: list[dict]) -> dict:
        """Use RAGAS with local Ollama LLM."""
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from ragas.llms import LangchainLLMWrapper
            from langchain_community.llms import Ollama
            from datasets import Dataset

            llm = Ollama(model=self.llm_model, base_url=self.ollama_url)
            wrapped_llm = LangchainLLMWrapper(llm)

            dataset = Dataset.from_list(eval_data)

            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=wrapped_llm,
            )

            return {
                "method": "ragas",
                "faithfulness": float(result["faithfulness"]),
                "answer_relevance": float(result["answer_relevancy"]),
                "context_precision": float(result["context_precision"]),
                "context_recall": float(result["context_recall"]),
            }
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e} — falling back to custom")
            return self._evaluate_custom(eval_data)

    def _evaluate_custom(self, eval_data: list[dict]) -> dict:
        """Custom heuristic evaluation."""
        evaluator = CustomRAGEvaluator()
        scores = {"faithfulness": [], "answer_relevance": [], "context_precision": [], "context_recall": []}

        for item in eval_data:
            question = item.get("question", "")
            answer = item.get("answer", "")
            contexts = item.get("contexts", [])
            ground_truth = item.get("ground_truth", "")

            scores["faithfulness"].append(evaluator.compute_faithfulness(answer, contexts))
            scores["answer_relevance"].append(evaluator.compute_answer_relevance(question, answer))
            scores["context_precision"].append(evaluator.compute_context_precision(question, contexts))
            scores["context_recall"].append(evaluator.compute_context_recall(ground_truth, contexts))

        return {
            "method": "custom_heuristic",
            "faithfulness": statistics.mean(scores["faithfulness"]),
            "answer_relevance": statistics.mean(scores["answer_relevance"]),
            "context_precision": statistics.mean(scores["context_precision"]),
            "context_recall": statistics.mean(scores["context_recall"]),
        }


class ComplianceEvaluationRunner:
    """
    Orchestrates the full evaluation run:
    1. Load test dataset
    2. Run each question through the RAG pipeline
    3. Collect answers and contexts
    4. Compute RAGAS metrics
    5. Generate report
    """

    def __init__(
        self,
        dataset_path: str = None,
        output_path: str = None,
    ):
        self.dataset_path = dataset_path or "src/evaluation/test_dataset.json"
        self.output_path = output_path or "outputs/eval_report.json"
        self.evaluator = RAGASEvaluator()

    def load_dataset(self) -> list[dict]:
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return data["questions"]

    def run_pipeline_for_question(self, question: dict) -> dict:
        """
        Run a single question through the RAG pipeline and collect results.
        Returns dict with question, answer, contexts for evaluation.
        """
        try:
            from src.retrieval.hybrid_search import HybridSearcher
            from src.retrieval.reranker import CrossEncoderReranker, format_chunks_for_llm
            from src.agent.compliance_agent import get_agent, ComplianceAgentState

            searcher = HybridSearcher()
            reranker = CrossEncoderReranker()

            q_text = question["question"]

            # Retrieve context
            chunks = searcher.search(q_text, top_k=10)
            reranked = reranker.rerank(q_text, chunks, top_k=5)
            contexts = [c.text for c in reranked]
            context_str = format_chunks_for_llm(reranked)

            # Generate answer
            agent = get_agent()
            initial_state: ComplianceAgentState = {
                "input": q_text,
                "intent": "qa",
                "transaction": None,
                "search_query": q_text,
                "retrieved_chunks": [],
                "jurisdictions_triggered": [],
                "regulations_triggered": [],
                "context": context_str,
                "draft_response": "",
                "citations": [],
                "confidence": "MEDIUM",
                "reflection_count": 0,
                "requires_human_review": False,
                "broader_search_needed": False,
                "broader_search_terms": [],
                "error": None,
                "final_response": {},
                "audit_log": [],
            }

            result = agent.invoke(initial_state)
            final = result.get("final_response", {})
            answer = final.get("response", "")
            confidence = final.get("confidence", "MEDIUM")

        except Exception as e:
            logger.error(f"Pipeline failed for Q{question.get('id')}: {e}")
            answer = f"Error: {str(e)}"
            contexts = []
            confidence = "LOW"

        return {
            "question": question["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": question["ground_truth"],
            "question_id": question["id"],
            "category": question["category"],
            "difficulty": question["difficulty"],
            "confidence": confidence,
        }

    def analyze_failures(self, per_question_results: list[dict]) -> dict:
        """Identify patterns in low-scoring questions."""
        failures = []
        low_threshold = 0.6

        for result in per_question_results:
            scores = result.get("scores", {})
            avg_score = sum(scores.values()) / len(scores) if scores else 0

            if avg_score < low_threshold:
                failures.append({
                    "question_id": result["question_id"],
                    "question": result["question"][:100],
                    "category": result.get("category"),
                    "difficulty": result.get("difficulty"),
                    "average_score": round(avg_score, 3),
                    "scores": {k: round(v, 3) for k, v in scores.items()},
                    "answer_preview": result.get("answer", "")[:200],
                    "failure_reason": self._diagnose_failure(scores),
                })

        # Identify failure patterns
        failure_categories = {}
        for f in failures:
            cat = f.get("category", "unknown")
            failure_categories[cat] = failure_categories.get(cat, 0) + 1

        return {
            "total_failures": len(failures),
            "failure_rate": len(failures) / len(per_question_results) if per_question_results else 0,
            "failures": failures,
            "failure_by_category": failure_categories,
        }

    def _diagnose_failure(self, scores: dict) -> str:
        """Diagnose the likely cause of low scores."""
        issues = []
        if scores.get("context_recall", 1.0) < 0.5:
            issues.append("Relevant documents not retrieved (recall issue — may need broader search or document ingestion)")
        if scores.get("context_precision", 1.0) < 0.5:
            issues.append("Irrelevant documents retrieved (precision issue — consider stricter filtering)")
        if scores.get("faithfulness", 1.0) < 0.5:
            issues.append("Answer contains claims not grounded in retrieved context (hallucination risk)")
        if scores.get("answer_relevance", 1.0) < 0.5:
            issues.append("Answer doesn't address the question (prompt engineering issue)")
        return "; ".join(issues) if issues else "Unexplained low score — manual review recommended"

    def run(self, max_questions: Optional[int] = None) -> dict:
        """Run full evaluation and generate report."""
        logger.info("Starting evaluation run...")
        start_time = time.time()

        questions = self.load_dataset()
        if max_questions:
            questions = questions[:max_questions]

        logger.info(f"Evaluating {len(questions)} questions...")

        # Collect pipeline results
        per_question_results = []
        eval_data_for_ragas = []

        for i, q in enumerate(questions):
            logger.info(f"Question {i+1}/{len(questions)}: {q['id']}")
            result = self.run_pipeline_for_question(q)
            per_question_results.append(result)
            eval_data_for_ragas.append({
                "question": result["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": result["ground_truth"],
            })

        # Compute aggregate metrics
        logger.info("Computing RAGAS metrics...")
        aggregate_metrics = self.evaluator.evaluate_dataset(eval_data_for_ragas)

        # Add per-question scores (using custom evaluator for individual scores)
        custom_eval = CustomRAGEvaluator()
        for result, eval_item in zip(per_question_results, eval_data_for_ragas):
            result["scores"] = {
                "faithfulness": round(custom_eval.compute_faithfulness(eval_item["answer"], eval_item["contexts"]), 3),
                "answer_relevance": round(custom_eval.compute_answer_relevance(eval_item["question"], eval_item["answer"]), 3),
                "context_precision": round(custom_eval.compute_context_precision(eval_item["question"], eval_item["contexts"]), 3),
                "context_recall": round(custom_eval.compute_context_recall(eval_item["ground_truth"], eval_item["contexts"]), 3),
            }

        # Failure analysis
        failure_analysis = self.analyze_failures(per_question_results)

        total_time = time.time() - start_time

        report = {
            "evaluation_metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "total_questions": len(questions),
                "total_time_seconds": round(total_time, 1),
                "avg_time_per_question_seconds": round(total_time / len(questions), 1),
                "evaluation_method": aggregate_metrics.get("method", "unknown"),
                "dataset_version": "1.0",
            },
            "aggregate_scores": {
                "faithfulness": round(aggregate_metrics["faithfulness"], 3),
                "answer_relevance": round(aggregate_metrics["answer_relevance"], 3),
                "context_precision": round(aggregate_metrics["context_precision"], 3),
                "context_recall": round(aggregate_metrics["context_recall"], 3),
                "overall_score": round(
                    sum([
                        aggregate_metrics["faithfulness"],
                        aggregate_metrics["answer_relevance"],
                        aggregate_metrics["context_precision"],
                        aggregate_metrics["context_recall"],
                    ]) / 4, 3
                ),
            },
            "scores_by_difficulty": self._scores_by_dimension(per_question_results, "difficulty"),
            "scores_by_category": self._scores_by_dimension(per_question_results, "category"),
            "failure_analysis": failure_analysis,
            "per_question_results": [
                {
                    "id": r["question_id"],
                    "category": r["category"],
                    "difficulty": r["difficulty"],
                    "confidence": r.get("confidence", "UNKNOWN"),
                    "scores": r["scores"],
                    "answer_preview": r["answer"][:300] + "..." if len(r.get("answer", "")) > 300 else r.get("answer", ""),
                }
                for r in per_question_results
            ],
        }

        # Write report
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Faithfulness:      {report['aggregate_scores']['faithfulness']:.3f}")
        logger.info(f"Answer Relevance:  {report['aggregate_scores']['answer_relevance']:.3f}")
        logger.info(f"Context Precision: {report['aggregate_scores']['context_precision']:.3f}")
        logger.info(f"Context Recall:    {report['aggregate_scores']['context_recall']:.3f}")
        logger.info(f"Overall Score:     {report['aggregate_scores']['overall_score']:.3f}")
        logger.info(f"Failures (<0.6):   {failure_analysis['total_failures']}/{len(questions)}")
        logger.info(f"Report saved to:   {self.output_path}")

        return report

    def _scores_by_dimension(self, results: list[dict], dimension: str) -> dict:
        """Aggregate scores by a categorical dimension (difficulty, category)."""
        groups: dict[str, list[dict]] = {}
        for r in results:
            key = r.get(dimension, "unknown")
            groups.setdefault(key, []).append(r["scores"])

        aggregated = {}
        for key, scores_list in groups.items():
            if scores_list:
                aggregated[key] = {
                    "count": len(scores_list),
                    "faithfulness": round(statistics.mean(s["faithfulness"] for s in scores_list), 3),
                    "answer_relevance": round(statistics.mean(s["answer_relevance"] for s in scores_list), 3),
                    "context_precision": round(statistics.mean(s["context_precision"] for s in scores_list), 3),
                    "context_recall": round(statistics.mean(s["context_recall"] for s in scores_list), 3),
                }
        return aggregated


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--output", default="outputs/eval_report.json", help="Output report path")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of questions (for quick testing)")
    parser.add_argument("--dataset", default="src/evaluation/test_dataset.json", help="Dataset path")
    args = parser.parse_args()

    runner = ComplianceEvaluationRunner(
        dataset_path=args.dataset,
        output_path=args.output,
    )
    report = runner.run(max_questions=args.max_questions)
    sys.exit(0)
