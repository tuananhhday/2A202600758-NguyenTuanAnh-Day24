from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # Implemented: RAGAS evaluation
    # 1. Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    # try:
    #     from ragas import evaluate
    #     from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    #     from datasets import Dataset
    #
    #     dataset = Dataset.from_dict({
    #         "question": questions, "answer": answers,
    #         "contexts": contexts, "ground_truth": ground_truths,
    #     })
    #     result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
    #                                         context_precision, context_recall])
    #     df = result.to_pandas()
    #     per_question = [EvalResult(question=row["question"], answer=row["answer"],
    #         contexts=row["contexts"], ground_truth=row["ground_truth"],
    #         faithfulness=float(row.get("faithfulness", 0.0)),
    #         answer_relevancy=float(row.get("answer_relevancy", 0.0)),
    #         context_precision=float(row.get("context_precision", 0.0)),
    #         context_recall=float(row.get("context_recall", 0.0)))
    #         for _, row in df.iterrows()]
    #     return {"faithfulness": ..., "answer_relevancy": ...,
    #             "context_precision": ..., "context_recall": ..., "per_question": [...]}
    # except Exception as e:
    #     print(f"  ⚠️  RAGAS evaluation failed: {e}")
    #     return zeros
    try:
        if os.getenv("USE_RAGAS", "0") != "1":
            raise RuntimeError("RAGAS disabled for fast local tests")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed, using lexical fallback: {e}")

        def tokens(value: str) -> set[str]:
            return set(re.findall(r"\w+", value.lower(), flags=re.UNICODE))

        per_question = []
        for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
            q_tokens = tokens(question)
            a_tokens = tokens(answer)
            gt_tokens = tokens(ground_truth)
            ctx_tokens = tokens(" ".join(ctxs))
            faith = len(a_tokens & ctx_tokens) / max(len(a_tokens), 1)
            relevancy = len(a_tokens & (q_tokens | gt_tokens)) / max(len(a_tokens), 1)
            precision = len(ctx_tokens & (q_tokens | gt_tokens)) / max(len(ctx_tokens), 1)
            recall = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1)
            per_question.append(EvalResult(
                question, answer, ctxs, ground_truth,
                round(faith, 4), round(relevancy, 4), round(precision, 4), round(recall, 4),
            ))

    if not per_question:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    return {
        "faithfulness": sum(r.faithfulness for r in per_question) / len(per_question),
        "answer_relevancy": sum(r.answer_relevancy for r in per_question) / len(per_question),
        "context_precision": sum(r.context_precision for r in per_question) / len(per_question),
        "context_recall": sum(r.context_recall for r in per_question) / len(per_question),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Implemented: failure analysis
    # 1. diagnostic_tree = {
    #        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    #        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    #        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    #        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    #    }
    # 2. For each EvalResult: compute avg of 4 metrics, find worst_metric
    # 3. Sort by avg ascending → take bottom_n
    # 4. Return [{"question": ..., "worst_metric": ..., "score": ...,
    #             "diagnosis": ..., "suggested_fix": ...}]
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or using unsupported details", "Tighten prompt and cite retrieved context."),
        "context_recall": ("Missing relevant chunks", "Improve chunking, hybrid search, or query coverage."),
        "context_precision": ("Retrieved context contains too much noise", "Add reranking or metadata filtering."),
        "answer_relevancy": ("Answer does not directly address the question", "Improve prompt and answer extraction."),
    }

    ranked = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        ranked.append({
            "question": result.question,
            "expected": result.ground_truth,
            "got": result.answer,
            "worst_metric": worst_metric,
            "score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "error_tree": f"Output correct? -> Context correct? -> Query OK? -> Root cause: {diagnosis}",
        })

    return sorted(ranked, key=lambda item: item["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
