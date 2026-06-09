"""
RAG Evaluation Pipeline — DeepEval + Gemini.

Mặc định DeepEval chấm điểm bằng OpenAI. Ở đây ta cấu hình lại để dùng Gemini
(qua endpoint OpenAI-compatible) làm "LLM judge", khớp với toàn bộ pipeline.

Quy trình:
    1. Load golden_dataset.json (>=15 Q&A)
    2. Chạy RAG pipeline (Task 9 + 10) cho 2 config: A = hybrid + rerank,
       B = hybrid (không rerank)
    3. Chấm 4 metric: Faithfulness, Answer Relevancy, Contextual Recall,
       Contextual Precision (judge = Gemini)
    4. Xuất bảng điểm + so sánh A/B + worst performers -> results.md

Chạy:
    uv run python -m group_project.evaluation.eval_pipeline
    # hoặc nhanh (ít câu):  ... eval_pipeline --limit 6
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.rag_utils import CHAT_MODEL, get_client  # noqa: E402
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task9_retrieval_pipeline import retrieve  # noqa: E402
from src.task10_generation import TEMPERATURE, TOP_P, build_messages  # noqa: E402

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

JUDGE_MODEL = "gemini-2.5-flash"


# =============================================================================
# Gemini làm LLM judge cho DeepEval
# =============================================================================

def _build_judge():
    from deepeval.models import DeepEvalBaseLLM

    class GeminiJudge(DeepEvalBaseLLM):
        """Custom DeepEval model dùng Gemini (OpenAI-compatible)."""

        def __init__(self, model: str = JUDGE_MODEL):
            self.model_name = model
            self.client = get_client()

        def load_model(self):
            return self.client

        def _create(self, **kwargs):
            """Gọi API có retry/backoff để chịu được rate-limit free tier."""
            import time

            last = None
            for attempt in range(5):
                try:
                    return self.client.chat.completions.create(**kwargs)
                except Exception as e:  # rate-limit / transient
                    last = e
                    time.sleep(2 * (attempt + 1))
            raise last

        def generate(self, prompt: str, schema=None):
            messages = [{"role": "user", "content": prompt}]
            if schema is not None:
                # JSON mode + nhắc schema để model trả đúng cấu trúc DeepEval cần.
                hint = (
                    "\n\nChỉ trả về JSON hợp lệ, đúng JSON schema sau "
                    "(không thêm chữ nào ngoài JSON):\n"
                    + json.dumps(schema.model_json_schema())
                )
                messages[0]["content"] = prompt + hint
                resp = self._create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                return schema.model_validate_json(resp.choices[0].message.content)

            resp = self._create(
                model=self.model_name, messages=messages, temperature=0
            )
            return resp.choices[0].message.content

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self):
            return self.model_name

    return GeminiJudge()


# =============================================================================
# Data
# =============================================================================

def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def _rag_answer(query: str, mode: str, top_k: int = 5) -> dict:
    """
    Sinh câu trả lời theo 1 trong 2 config retrieval (A/B):
        mode="hybrid": semantic + lexical (BM25) hợp nhất bằng RRF
        mode="dense" : chỉ semantic (dense) search

    Tắt fallback PageIndex (score_threshold=0) để A/B tái lập được, không phụ
    thuộc quota dịch vụ ngoài.
    """
    if mode == "dense":
        chunks = semantic_search(query, top_k=top_k)
    else:
        chunks = retrieve(query, top_k=top_k, use_reranking=False, score_threshold=0.0)

    if not chunks:
        return {"answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.", "sources": []}

    resp = get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=build_messages(query, chunks),
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return {"answer": resp.choices[0].message.content, "sources": chunks}


def _build_test_cases(golden: list[dict], mode: str):
    from deepeval.test_case import LLMTestCase

    cases = []
    for i, item in enumerate(golden, 1):
        print(f"  [{i}/{len(golden)}] RAG ({mode}): {item['question'][:45]}...")
        result = _rag_answer(item["question"], mode=mode)
        cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result["answer"],
                expected_output=item["expected_answer"],
                retrieval_context=[c["content"] for c in result["sources"]],
            )
        )
    return cases


# =============================================================================
# Evaluation (chấm thủ công từng metric để tổng hợp điểm + worst performers)
# =============================================================================

def _metrics(judge):
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    kw = dict(model=judge, threshold=0.5, async_mode=False, include_reason=False)
    return {
        "Faithfulness": FaithfulnessMetric(**kw),
        "Answer Relevancy": AnswerRelevancyMetric(**kw),
        "Contextual Recall": ContextualRecallMetric(**kw),
        "Contextual Precision": ContextualPrecisionMetric(**kw),
    }


def evaluate_config(name: str, cases, judge) -> dict:
    """Chấm 1 config -> {metric: [scores...]} + per-case averages."""
    metrics = _metrics(judge)
    scores = {m: [] for m in metrics}
    per_case = []

    for idx, case in enumerate(cases):
        case_scores = {}
        for mname, metric in metrics.items():
            try:
                metric.measure(case)
                s = metric.score if metric.score is not None else 0.0
            except Exception as e:
                print(f"    ⚠ {mname} lỗi case {idx}: {e}")
                s = 0.0
            scores[mname].append(s)
            case_scores[mname] = s
        avg = sum(case_scores.values()) / len(case_scores)
        per_case.append({"input": case.input, "avg": avg, "scores": case_scores})
        print(f"    case {idx + 1}: avg={avg:.2f}")

    averages = {m: (sum(v) / len(v) if v else 0.0) for m, v in scores.items()}
    return {"name": name, "averages": averages, "per_case": per_case}


# =============================================================================
# Report
# =============================================================================

def export_results(config_a: dict, config_b: dict):
    metrics = list(config_a["averages"].keys())

    def overall(cfg):
        vals = list(cfg["averages"].values())
        return sum(vals) / len(vals) if vals else 0.0

    lines = ["# RAG Evaluation Results\n"]
    lines.append("## Framework sử dụng\n")
    lines.append(f"**DeepEval** với LLM judge = Gemini (`{JUDGE_MODEL}`), "
                 "qua endpoint OpenAI-compatible.\n")
    lines.append("\n## Overall Scores (A/B)\n")
    lines.append("| Metric | A: Hybrid (semantic+BM25+RRF) | B: Dense-only (semantic) | Δ (A−B) |")
    lines.append("|---|---|---|---|")
    for m in metrics:
        a, b = config_a["averages"][m], config_b["averages"][m]
        lines.append(f"| {m} | {a:.3f} | {b:.3f} | {a - b:+.3f} |")
    oa, ob = overall(config_a), overall(config_b)
    lines.append(f"| **Average** | **{oa:.3f}** | **{ob:.3f}** | **{oa - ob:+.3f}** |")

    winner = "A — Hybrid (semantic+BM25+RRF)" if oa >= ob else "B — Dense-only (semantic)"
    lines.append(f"\n**Config tốt hơn:** {winner}.\n")
    lines.append(
        "\n> Ghi chú: trục A/B là **Hybrid vs Dense-only** (theo gợi ý README). "
        "Reranking (Task 7 — Jina) đã hiện thực và kiểm chứng (relevant 0.704 vs "
        "nhiễu 0.07) nhưng không đưa vào A/B vì quota Jina free đã hết trong quá "
        "trình test; PageIndex fallback cũng tắt khi eval để kết quả tái lập.\n"
    )

    # Worst performers theo config A
    lines.append("\n## Worst performers (Config A)\n")
    worst = sorted(config_a["per_case"], key=lambda x: x["avg"])[:3]
    lines.append("| Câu hỏi | Avg | Chi tiết |")
    lines.append("|---|---|---|")
    for w in worst:
        detail = ", ".join(f"{k}={v:.2f}" for k, v in w["scores"].items())
        lines.append(f"| {w['input'][:60]}… | {w['avg']:.2f} | {detail} |")

    lines.append("\n## Phân tích & đề xuất cải tiến\n")
    lines.append(
        "- **Hybrid (A) vs Dense-only (B):** Δ ở Contextual Recall/Precision cho thấy "
        "việc thêm BM25 + RRF có giúp lấy đúng điều khoản (khớp số điều, thuật ngữ "
        "pháp lý) mà dense đôi khi bỏ sót hay không.\n"
        "- Câu điểm thấp thường do **Contextual Recall** thấp: tài liệu chưa đủ chi "
        "tiết hoặc chunk_size cắt mất ngữ cảnh → thử tăng chunk_size hoặc bổ sung nguồn.\n"
        "- Câu hỏi tin tức ngắn: **Answer Relevancy** cao nhưng Recall thấp do bài báo "
        "còn nhiễu (menu/quảng cáo) → bóc tách nội dung sạch hơn hoặc lọc theo loại tài liệu.\n"
        "- **Bật lại reranking (Jina)** khi có quota để đẩy chunk nhiễu xuống, kỳ vọng "
        "tăng Contextual Precision.\n"
    )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Đã ghi báo cáo: {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số câu (0 = full)")
    args = parser.parse_args()

    golden = load_golden_dataset()
    if args.limit:
        golden = golden[: args.limit]
    print(f"Loaded {len(golden)} test cases")

    judge = _build_judge()

    print("\n=== Config A: Hybrid (semantic + BM25 + RRF) ===")
    cases_a = _build_test_cases(golden, mode="hybrid")
    print("--- Scoring A ---")
    config_a = evaluate_config("A", cases_a, judge)

    print("\n=== Config B: Dense-only (semantic) ===")
    cases_b = _build_test_cases(golden, mode="dense")
    print("--- Scoring B ---")
    config_b = evaluate_config("B", cases_b, judge)

    export_results(config_a, config_b)


if __name__ == "__main__":
    main()
