"""
Shared utilities cho RAG pipeline (dùng chung cho Task 4–10).

Thiết kế "API-first, không tải model":
    - Embeddings + generation  -> Google Gemini qua endpoint OpenAI-compatible,
      nên ta tái sử dụng luôn `openai` SDK, không cần thư viện riêng.
    - Vector store             -> lưu local bằng numpy (data/index/), không cần
      Weaviate/Docker. Embedding vẫn lấy qua API.

Cấu hình model tập trung tại đây để các task khác import.
"""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# MODEL / API CONFIG
# =============================================================================

# Gemini expose 1 endpoint tương thích OpenAI -> dùng openai SDK trỏ vào đây.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Embedding: gemini-embedding-001 — multilingual (tốt cho tiếng Việt).
# Model hỗ trợ Matryoshka -> ép output 768 chiều cho nhẹ.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# Generation: gemini flash — nhanh, rẻ, đủ tốt cho RAG factual.
CHAT_MODEL = "gemini-3.5-flash"

# =============================================================================
# LOCAL VECTOR STORE PATHS
# =============================================================================

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"


# =============================================================================
# LLM / EMBEDDING CLIENT (Gemini via OpenAI-compatible API)
# =============================================================================

_client = None


def get_client():
    """Trả về OpenAI client đã cấu hình trỏ tới Gemini. Lazy + cached."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "Thiếu GEMINI_API_KEY trong .env. "
                "Lấy key tại https://aistudio.google.com/apikey"
            )
        from openai import OpenAI

        _client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    return _client


def embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed một list văn bản qua Gemini embeddings API.

    Returns:
        np.ndarray shape (len(texts), EMBEDDING_DIM), dtype float32.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    client = get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIM
        )
        vectors.extend(item.embedding for item in resp.data)
    return np.array(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed 1 query -> vector 1 chiều (EMBEDDING_DIM,)."""
    return embed_texts([query])[0]


# =============================================================================
# LOCAL VECTOR STORE (save / load)
# =============================================================================

def save_index(chunks: list[dict], embeddings: np.ndarray):
    """Lưu chunks (json) + embeddings (npy) xuống data/index/."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(EMBEDDINGS_PATH, embeddings.astype(np.float32))


def load_chunks() -> list[dict]:
    """Đọc danh sách chunks đã index (rỗng nếu chưa index)."""
    if not CHUNKS_PATH.exists():
        return []
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def load_embeddings() -> np.ndarray:
    """Đọc ma trận embeddings đã index (rỗng nếu chưa index)."""
    if not EMBEDDINGS_PATH.exists():
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return np.load(EMBEDDINGS_PATH)


def index_exists() -> bool:
    return CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists()


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity giữa 1 vector và từng hàng của matrix -> (N,)."""
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m @ q
