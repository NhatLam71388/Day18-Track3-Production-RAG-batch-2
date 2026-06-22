from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, types, warnings
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL


def _stub_vertexai():
    """ragas 0.4.x imports ChatVertexAI from a langchain_community path removed in
    langchain_community 0.4.x. We use an OpenAI-compatible gateway (no VertexAI),
    so inject a harmless stub to let ragas import cleanly."""
    mod = "langchain_community.chat_models.vertexai"
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[mod] = stub


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
    """Run RAGAS evaluation (4 metrics). Dùng gateway LLM + local HF embeddings.

    Bọc try/except: nếu thiếu API key / RAGAS lỗi → trả về zeros để pipeline vẫn chạy.
    Tương thích ragas 0.4.x (EvaluationDataset + LangchainLLMWrapper).
    """
    import math
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
    if not OPENAI_API_KEY:
        print("  ⚠️  RAGAS skipped: no OPENAI_API_KEY")
        return zeros

    try:
        _stub_vertexai()
        warnings.filterwarnings("ignore")
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision, context_recall)
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_core.embeddings import Embeddings
        from sentence_transformers import SentenceTransformer

        llm = LangchainLLMWrapper(ChatOpenAI(
            model=LLM_MODEL, api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL or None, temperature=0.0, timeout=60))

        class _LocalEmbeddings(Embeddings):
            def __init__(self):
                self._m = SentenceTransformer("all-MiniLM-L6-v2")
            def embed_documents(self, texts):
                return self._m.encode(texts).tolist()
            def embed_query(self, text):
                return self._m.encode(text).tolist()

        embeddings = LangchainEmbeddingsWrapper(_LocalEmbeddings())

        samples = [{"user_input": q, "response": a,
                    "retrieved_contexts": list(c) if c else [""], "reference": gt}
                   for q, a, c, gt in zip(questions, answers, contexts, ground_truths)]
        dataset = EvaluationDataset.from_list(samples)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm, embeddings=embeddings, show_progress=True, raise_exceptions=False)
        df = result.to_pandas()

        # Map các cột metric (tên có thể khác nhau giữa version) → 4 key chuẩn.
        std_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        non_metric = {"user_input", "response", "retrieved_contexts", "reference",
                      "question", "answer", "contexts", "ground_truth"}
        colmap: dict[str, str] = {}
        for col in df.columns:
            if col in non_metric:
                continue
            lc = col.lower()
            if "faith" in lc:
                colmap[col] = "faithfulness"
            elif "relevan" in lc:
                colmap[col] = "answer_relevancy"
            elif "precision" in lc:
                colmap[col] = "context_precision"
            elif "recall" in lc:
                colmap[col] = "context_recall"

        def _val(row, key):
            for col, std in colmap.items():
                if std == key:
                    v = row.get(col, 0.0)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        return 0.0
                    return 0.0 if math.isnan(v) else v
            return 0.0

        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=row.get("user_input", row.get("question", "")),
                answer=row.get("response", row.get("answer", "")),
                contexts=list(row.get("retrieved_contexts", row.get("contexts", []))),
                ground_truth=row.get("reference", row.get("ground_truth", "")),
                faithfulness=_val(row, "faithfulness"),
                answer_relevancy=_val(row, "answer_relevancy"),
                context_precision=_val(row, "context_precision"),
                context_recall=_val(row, "context_recall"),
            ))

        def _avg(key):
            vals = [getattr(r, key) for r in per_question]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return {k: _avg(k) for k in std_keys} | {"per_question": per_question}

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — câu trả lời không bám context",
                         "Siết prompt 'chỉ dùng context', giảm temperature"),
        "context_recall": ("Thiếu chunk liên quan trong context",
                           "Cải thiện chunking, thêm BM25/hybrid, tăng top_k"),
        "context_precision": ("Quá nhiều chunk nhiễu trong context",
                              "Thêm reranking hoặc metadata filter"),
        "answer_relevancy": ("Câu trả lời lệch khỏi câu hỏi",
                             "Cải thiện prompt template, làm rõ instruction"),
    }
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    scored = []
    for r in eval_results:
        vals = {m: getattr(r, m) for m in metrics}
        avg = sum(vals.values()) / len(vals)
        worst_metric = min(vals, key=vals.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "worst_score": round(vals[worst_metric], 4),
            "avg_score": round(avg, 4),
            "score": round(avg, 4),
            "metrics": {m: round(v, 4) for m, v in vals.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:bottom_n]


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
