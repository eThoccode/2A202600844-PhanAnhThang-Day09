"""In-process multi-agent layer for the RAG chatbot."""

from collections.abc import Iterator

from .rag_utils import CHAT_MODEL, get_client
from .task9_retrieval_pipeline import retrieve
from .task10_generation import TEMPERATURE, TOP_P, build_messages


class RetrievalAgent:
    """Agent chuyên truy hồi evidence từ RAG pipeline hiện có."""

    def run(self, query: str, top_k: int = 5, use_reranking: bool = True) -> list[dict]:
        return retrieve(query, top_k=top_k, use_reranking=use_reranking)


class GenerationAgent:
    """Agent chuyên sinh câu trả lời có citation từ chunks đã truy hồi."""

    def run(self, query: str, chunks: list[dict], history: list[dict] | None = None) -> str:
        resp = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=build_messages(query, chunks, history),
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return resp.choices[0].message.content

    def stream(self, query: str, chunks: list[dict], history: list[dict] | None = None) -> Iterator[str]:
        gen = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=build_messages(query, chunks, history),
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=True,
        )
        for ev in gen:
            delta = ev.choices[0].delta.content if ev.choices else None
            if delta:
                yield delta


class OrchestratorAgent:
    """Agent điều phối RetrievalAgent và GenerationAgent trong cùng process."""

    def __init__(self, retrieval: RetrievalAgent | None = None, generation: GenerationAgent | None = None):
        self.retrieval = retrieval or RetrievalAgent()
        self.generation = generation or GenerationAgent()

    def run(
        self,
        query: str,
        history: list[dict] | None = None,
        top_k: int = 5,
        use_reranking: bool = True,
    ) -> dict:
        chunks = self.retrieval.run(query, top_k=top_k, use_reranking=use_reranking)
        if not chunks:
            return {
                "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                "sources": [],
                "retrieval_source": "none",
            }

        answer = self.generation.run(query, chunks, history)
        return {
            "answer": answer,
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid"),
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True,
    ) -> dict:
        chunks = self.retrieval.run(query, top_k=top_k, use_reranking=use_reranking)
        return {
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        }

    def stream_answer(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        return self.generation.stream(query, chunks, history)
