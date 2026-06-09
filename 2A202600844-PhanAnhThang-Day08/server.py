"""
Demo backend (FastAPI) cho RAG Chatbot — phục vụ frontend tĩnh trong web/.

    GET  /            -> web/index.html (giao diện demo)
    POST /api/chat    -> stream NDJSON: {sources} rồi {token...} rồi {done}

Chạy:
    uv run uvicorn server:app --reload --port 8000
    # mở http://localhost:8000
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents import OrchestratorAgent
from src.source_labels import friendly_label

WEB_DIR = Path(__file__).parent / "web"
STD_LEGAL = Path(__file__).parent / "data" / "standardized" / "legal"
LANDING_NEWS = Path(__file__).parent / "data" / "landing" / "news"

app = FastAPI(title="RAG Pháp luật Ma túy")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    top_k: int = 5
    use_reranking: bool = True


def _source_card(chunk: dict) -> dict:
    meta = chunk.get("metadata", {})
    source = meta.get("source", "?")
    doc_type = meta.get("type", "unknown")
    return {
        "source": source,
        "label": friendly_label(source, doc_type),
        "type": doc_type,
        "score": round(float(chunk.get("score", 0)), 3),
        "via": chunk.get("source", "hybrid"),
        "snippet": chunk["content"][:280].strip(),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    def stream():
        orchestrator = OrchestratorAgent()
        retrieval = orchestrator.retrieve(
            req.message,
            top_k=req.top_k,
            use_reranking=req.use_reranking,
        )
        chunks = retrieval["sources"]
        via = retrieval["retrieval_source"]
        yield json.dumps(
            {"type": "sources", "via": via, "sources": [_source_card(c) for c in chunks]},
            ensure_ascii=False,
        ) + "\n"

        if not chunks:
            yield json.dumps(
                {"type": "token", "text": "Tôi không thể xác minh thông tin này từ nguồn hiện có."},
                ensure_ascii=False,
            ) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        for token in orchestrator.stream_answer(req.message, chunks, req.history):
            yield json.dumps({"type": "token", "text": token}, ensure_ascii=False) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/corpus")
def corpus():
    """Danh sách tài liệu & bài báo hệ thống đang dùng (hiện trên UI)."""
    legal = [
        {"label": friendly_label(f.name, "legal"), "file": f.name}
        for f in sorted(STD_LEGAL.glob("*.md"))
    ]
    news = []
    if LANDING_NEWS.exists():
        for f in sorted(LANDING_NEWS.glob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            news.append(
                {
                    "title": d.get("title", f.stem),
                    "url": d.get("url", ""),
                    "date": d.get("date", ""),
                    "author": d.get("author", ""),
                }
            )
    return {"legal": legal, "news": news}


# Frontend tĩnh (đặt cuối để không che /api/*).
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
