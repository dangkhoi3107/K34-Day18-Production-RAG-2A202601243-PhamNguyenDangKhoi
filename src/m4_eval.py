from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
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
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        import pandas as pd

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            f_val = row.get("faithfulness", 0.0)
            ar_val = row.get("answer_relevancy", 0.0)
            cp_val = row.get("context_precision", 0.0)
            cr_val = row.get("context_recall", 0.0)
            per_question.append(
                EvalResult(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    contexts=list(row.get("contexts", [])),
                    ground_truth=str(row.get("ground_truth", "")),
                    faithfulness=float(f_val if pd.notna(f_val) else 0.0),
                    answer_relevancy=float(ar_val if pd.notna(ar_val) else 0.0),
                    context_precision=float(cp_val if pd.notna(cp_val) else 0.0),
                    context_recall=float(cr_val if pd.notna(cr_val) else 0.0),
                )
            )

        f_agg = result.get("faithfulness", 0.0)
        ar_agg = result.get("answer_relevancy", 0.0)
        cp_agg = result.get("context_precision", 0.0)
        cr_agg = result.get("context_recall", 0.0)

        return {
            "faithfulness": float(f_agg if pd.notna(f_agg) else 0.0),
            "answer_relevancy": float(ar_agg if pd.notna(ar_agg) else 0.0),
            "context_precision": float(cp_agg if pd.notna(cp_agg) else 0.0),
            "context_recall": float(cr_agg if pd.notna(cr_agg) else 0.0),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature, or provide more direct context"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25 keyword search"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template and instructions"),
    }
    scored_items = []
    for r in eval_results:
        metric_dict = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metric_dict, key=metric_dict.get)
        avg_score = sum(metric_dict.values()) / 4.0
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        scored_items.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "contexts": r.contexts,
            "avg_score": avg_score,
            "worst_metric": worst_metric,
            "score": metric_dict[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    scored_items.sort(key=lambda x: x["avg_score"])
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json",
                extra: dict | None = None):
    """Save evaluation report to JSON. (Đã implement sẵn)

    `extra` là chỗ để gắn thêm section tuỳ ý (vd. latency breakdown của
    pipeline) mà không phá cấu trúc aggregate/num_questions mà check_lab.py
    kiểm tra.
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    if extra:
        report.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
