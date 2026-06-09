"""
Task 5 — Semantic Search Module (dense retrieval).

Embed query bằng cùng model ở Task 4 (Gemini text-embedding-004), rồi tính
cosine similarity với toàn bộ chunk embeddings trong local store.
"""

import numpy as np

from .rag_utils import cosine_similarity, embed_query, load_chunks, load_embeddings


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng vector similarity.

    Returns:
        List of {'content': str, 'score': float (cosine), 'metadata': dict}
        sorted by score descending. Rỗng nếu chưa index (Task 4).
    """
    chunks = load_chunks()
    embeddings = load_embeddings()
    if not chunks or embeddings.shape[0] == 0:
        return []

    q_vec = embed_query(query)
    scores = cosine_similarity(q_vec, embeddings)

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "content": chunks[i]["content"],
            "score": float(scores[i]),
            "metadata": chunks[i].get("metadata", {}),
        }
        for i in top_idx
    ]


if __name__ == "__main__":
    for r in semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
