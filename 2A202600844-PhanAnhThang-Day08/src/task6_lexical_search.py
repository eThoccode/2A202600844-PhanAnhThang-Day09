"""
Task 6 — Lexical Search Module (BM25).

Dùng rank-bm25 (BM25Okapi) trên cùng corpus chunks đã index ở Task 4.

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document -> điểm cao.
    - Inverse Document Frequency (IDF): từ hiếm -> quan trọng hơn.
    - Length normalization: document dài không bị ưu tiên quá mức.
    - score(q,d) = Σ IDF(qi) * (tf*(k1+1)) / (tf + k1*(1-b+b*|d|/avgdl))
      với k1=1.5 (term saturation), b=0.75 (length normalization).

Tokenize tiếng Việt: lowercase + tách theo khoảng trắng/ký tự (đủ tốt cho BM25;
có thể nâng cấp bằng underthesea nếu muốn — sẽ là bonus trong demo).
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .rag_utils import load_chunks

_bm25 = None
_corpus: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """Lowercase + tách token chữ/số (giữ được từ có dấu tiếng Việt)."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def build_bm25_index(corpus: list[dict] | None = None):
    """Xây BM25 index từ corpus chunks (mặc định: load từ local store)."""
    global _bm25, _corpus
    _corpus = corpus if corpus is not None else load_chunks()
    if not _corpus:
        _bm25 = None
        return None
    tokenized = [_tokenize(doc["content"]) for doc in _corpus]
    _bm25 = BM25Okapi(tokenized)
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa bằng BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending (chỉ trả về chunk có score > 0).
    """
    if _bm25 is None or not _corpus:
        build_bm25_index()
    if _bm25 is None:
        return []

    scores = _bm25.get_scores(_tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_idx:
        if scores[i] <= 0:
            continue
        results.append(
            {
                "content": _corpus[i]["content"],
                "score": float(scores[i]),
                "metadata": _corpus[i].get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    for r in lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
