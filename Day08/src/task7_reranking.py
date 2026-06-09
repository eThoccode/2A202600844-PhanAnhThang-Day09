"""
Task 7 — Reranking Module.

Mặc định: cross-encoder qua Jina Reranker API (multilingual, tốt cho tiếng Việt).
Nếu không có JINA_API_KEY hoặc API lỗi -> tự động fallback sang reranker
lexical-overlap (không cần model/API) để pipeline không bao giờ gãy.

Ngoài ra cung cấp 2 thuật toán tự implement:
    - rerank_rrf : Reciprocal Rank Fusion (gộp nhiều ranked list)
    - rerank_mmr : Maximal Marginal Relevance (tăng diversity, giảm trùng lặp)
"""

import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_MODEL = "jina-reranker-v2-base-multilingual"


def _tokens(text: str) -> set:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _rerank_lexical(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Fallback không cần API: chấm điểm theo độ trùng token query↔content."""
    q = _tokens(query)
    scored = []
    for c in candidates:
        overlap = len(q & _tokens(c["content"]))
        score = overlap / (len(q) + 1e-9)
        scored.append({**c, "score": float(score)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng cross-encoder (Jina API). Tự fallback lexical nếu không khả dụng.
    """
    if not candidates:
        return []
    if not JINA_API_KEY:
        return _rerank_lexical(query, candidates, top_k)

    try:
        resp = requests.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        return [
            {**candidates[r["index"]], "score": float(r["relevance_score"])}
            for r in results
        ]
    except Exception as e:
        print(f"  ⚠ Jina rerank lỗi ({e}); fallback lexical.")
        return _rerank_lexical(query, candidates, top_k)


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp nhiều ranked list.
        RRF(d) = Σ 1 / (k + rank_r(d))
    k=60 theo Cormack et al. 2009.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    ordered = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in ordered[:top_k]:
        item = dict(content_map[content])
        item["score"] = float(score)
        results.append(item)
    return results


def rerank_mmr(
    query_embedding,
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — cân bằng relevance và diversity.
        MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))
    Yêu cầu mỗi candidate có key 'embedding'.
    """
    import numpy as np

    def cos(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            relevance = cos(query_embedding, candidates[idx]["embedding"])
            max_sim = max(
                (cos(candidates[idx]["embedding"], candidates[s]["embedding"]) for s in selected),
                default=0.0,
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Interface thống nhất. Mặc định cross-encoder (Jina) + fallback lexical."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "lexical":
        return _rerank_lexical(query, candidates, top_k)
    elif method == "mmr":
        return rerank_mmr(query, candidates, top_k)  # query phải là embedding
    elif method == "rrf":
        raise ValueError("rerank_rrf cần nhiều ranked_lists — gọi rerank_rrf trực tiếp.")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    for r in rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2):
        print(f"[{r['score']:.3f}] {r['content']}")
