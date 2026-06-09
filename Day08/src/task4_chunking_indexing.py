"""
Task 4 — Chunking & Indexing.

Lựa chọn (API-first, không tải model):
    - Chunking : RecursiveCharacterTextSplitter
        Vì sao? Văn bản pháp luật + báo chí có cấu trúc đoạn/câu rõ; recursive
        splitter cắt ưu tiên theo \n\n -> \n -> ". " nên giữ được ngữ nghĩa,
        an toàn và phổ biến. chunk_size=800 ký tự (~150-200 từ) đủ chứa 1 điều
        khoản/ý mà không quá dài; overlap=120 (~15%) để không mất ngữ cảnh ở
        ranh giới chunk.
    - Embedding : Gemini text-embedding-004 (768 dim) qua API
        Vì sao? Multilingual (tốt cho tiếng Việt), gọi qua API nên không phải
        tải model nặng về máy.
    - Vector store : local numpy (data/index/)
        Vì sao? Nhẹ, không cần Docker/Weaviate; embedding đã tính qua API rồi
        nên chỉ cần lưu ma trận vector + metadata.

Chạy:  python -m src.task4_chunking_indexing
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .rag_utils import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    embed_texts,
    save_index,
)

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 800        # ~150-200 từ: đủ 1 điều khoản/ý, không gây lost-in-middle
CHUNK_OVERLAP = 120     # ~15% overlap: giữ ngữ cảnh ở ranh giới chunk
CHUNKING_METHOD = "recursive"

VECTOR_STORE = "local"  # numpy on disk (data/index/)


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type},
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc["content"])):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "content": piece,
                    "metadata": {**doc["metadata"], "chunk_index": i},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]):
    """Embed nội dung chunks qua Gemini API. Trả về np.ndarray (N, dim)."""
    texts = [c["content"] for c in chunks]
    return embed_texts(texts)


def index_to_vectorstore(chunks: list[dict], embeddings):
    """Lưu chunks + embeddings xuống local store (data/index/)."""
    save_index(chunks, embeddings)


def run_pipeline():
    """Chạy toàn bộ: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        print("⚠ Chưa có markdown trong data/standardized/. Chạy Task 3 trước.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks -> shape {embeddings.shape}")

    index_to_vectorstore(chunks, embeddings)
    print("✓ Indexed to local vector store (data/index/)")


if __name__ == "__main__":
    run_pipeline()
