"""
Task 8 — PageIndex Vectorless RAG.

PageIndex (https://pageindex.ai) là RAG "vectorless": thay vì embedding, nó dựng
cây cấu trúc (tree) của tài liệu rồi LLM duyệt cây để truy hồi. Rất hợp với văn
bản pháp luật có mục/điều/khoản rõ ràng.

Lưu ý API:
    - PageIndex chỉ nhận file PDF. Văn bản gốc của ta là .doc -> ta convert sang
      PDF bằng Microsoft Word (COM) trước khi upload.
    - Xử lý bất đồng bộ: upload -> chờ build tree -> submit query -> chờ kết quả.

Cách dùng:
    1. Đặt PAGEINDEX_API_KEY trong .env
    2. python -m src.task8_pageindex_vectorless     # upload + test query
"""

import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
DOC_MAP_PATH = INDEX_DIR / "pageindex_docs.json"

# Word FileFormat: wdFormatPDF
_WD_FORMAT_PDF = 17


def _get_client():
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env (đăng ký tại pageindex.ai)")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _doc_to_pdf(doc_path: Path) -> Path:
    """Convert .doc/.docx -> PDF bằng Word COM. Trả về đường dẫn PDF tạm."""
    import pythoncom  # type: ignore
    import win32com.client  # type: ignore

    pythoncom.CoInitialize()
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    out_path = Path(tempfile.gettempdir()) / f"{doc_path.stem}.pdf"
    try:
        doc = word.Documents.Open(str(doc_path.resolve()), ReadOnly=True)
        doc.SaveAs2(str(out_path), FileFormat=_WD_FORMAT_PDF)
        doc.Close(SaveChanges=False)
    finally:
        word.Quit()
        pythoncom.CoUninitialize()
    return out_path


def upload_documents(wait: bool = True, timeout: int = 600) -> dict:
    """
    Upload toàn bộ tài liệu pháp luật lên PageIndex (convert .doc -> PDF nếu cần).
    Lưu mapping {doc_id: filename} vào data/index/pageindex_docs.json.
    """
    client = _get_client()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    doc_map: dict[str, str] = {}

    for filepath in sorted(LEGAL_DIR.iterdir()):
        suffix = filepath.suffix.lower()
        if suffix not in (".pdf", ".doc", ".docx"):
            continue

        tmp_pdf = None
        try:
            pdf_path = filepath
            if suffix != ".pdf":
                tmp_pdf = _doc_to_pdf(filepath)
                pdf_path = tmp_pdf

            print(f"Uploading: {filepath.name}")
            resp = client.submit_document(str(pdf_path))
            doc_id = resp["doc_id"]
            doc_map[doc_id] = filepath.name
            print(f"  ✓ doc_id={doc_id}")
        except Exception as e:
            print(f"  ✗ Lỗi upload {filepath.name}: {e}")
        finally:
            if tmp_pdf and tmp_pdf.exists():
                os.remove(tmp_pdf)

    DOC_MAP_PATH.write_text(json.dumps(doc_map, ensure_ascii=False, indent=2), encoding="utf-8")

    if wait:
        print("\nĐang chờ PageIndex build tree (có thể vài phút)...")
        deadline = time.time() + timeout
        pending = set(doc_map)
        while pending and time.time() < deadline:
            for doc_id in list(pending):
                if client.is_retrieval_ready(doc_id):
                    print(f"  ✓ Ready: {doc_map[doc_id]}")
                    pending.discard(doc_id)
            if pending:
                time.sleep(10)
        if pending:
            print(f"  ⚠ Còn {len(pending)} tài liệu chưa sẵn sàng (timeout).")

    return doc_map


def _load_doc_ids() -> list[str]:
    if not DOC_MAP_PATH.exists():
        return []
    return list(json.loads(DOC_MAP_PATH.read_text(encoding="utf-8")).keys())


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval qua PageIndex (Chat API + citations). Dùng làm fallback ở Task 9.

    Lưu ý: endpoint /retrieval cũ đã deprecated -> dùng chat-completions API: PageIndex
    tự duyệt cây tài liệu, sinh câu trả lời kèm citation <doc=...;page=N> và trả về
    danh sách citations (document, page).

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}
    """
    doc_ids = _load_doc_ids()
    if not doc_ids:
        raise RuntimeError("Chưa upload tài liệu lên PageIndex. Chạy upload_documents() trước.")

    client = _get_client()
    resp = client.chat_completions(
        messages=[{"role": "user", "content": query}],
        doc_id=doc_ids,
        enable_citations=True,
        stream=False,
    )
    answer = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not answer:
        return []

    citations = resp.get("citations", [])
    cited = ", ".join(f"{c.get('document')} p.{c.get('page')}" for c in citations) or "PageIndex"

    return [
        {
            "content": answer,
            "score": 1.0,
            "metadata": {"source": cited, "type": "legal", "citations": citations},
            "source": "pageindex",
        }
    ][:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong .env (đăng ký tại https://pageindex.ai/)")
    else:
        if not DOC_MAP_PATH.exists():
            upload_documents()
        print("\nTest query:")
        for r in pageindex_search("hình phạt sử dụng ma tuý", top_k=3):
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
