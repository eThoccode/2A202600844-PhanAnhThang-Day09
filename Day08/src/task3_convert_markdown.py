"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Văn bản pháp luật ở đây là định dạng .doc cũ (Word 97-2003, OLE2) — MarkItDown
chỉ đọc được .docx. Do máy có sẵn Microsoft Word, ta dùng Word COM tự động
convert .doc -> .docx trước, rồi mới đưa qua MarkItDown.

    .doc  --(Word COM)-->  .docx  --(MarkItDown)-->  .md
    .docx --(MarkItDown)-->  .md
    news .json (từ Task 2) --(đọc trực tiếp)--> .md

Cách dùng:
    python -m src.task3_convert_markdown
"""

import json
import os
import tempfile
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Word FileFormat: wdFormatXMLDocument (.docx)
_WD_FORMAT_DOCX = 16


def _convert_doc_to_docx(doc_path: Path) -> Path:
    """Dùng Microsoft Word (COM) convert .doc -> .docx. Trả về đường dẫn .docx tạm."""
    import pythoncom  # type: ignore
    import win32com.client  # type: ignore

    pythoncom.CoInitialize()
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    out_path = Path(tempfile.gettempdir()) / f"{doc_path.stem}.docx"
    try:
        doc = word.Documents.Open(str(doc_path.resolve()), ReadOnly=True)
        doc.SaveAs2(str(out_path), FileFormat=_WD_FORMAT_DOCX)
        doc.Close(SaveChanges=False)
    finally:
        word.Quit()
        pythoncom.CoUninitialize()
    return out_path


def convert_legal_docs():
    """Convert PDF/DOC/DOCX trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        suffix = filepath.suffix.lower()
        if suffix not in (".pdf", ".docx", ".doc"):
            continue
        print(f"Converting: {filepath.name}")

        tmp_docx = None
        try:
            source = filepath
            if suffix == ".doc":  # legacy -> convert qua Word trước
                tmp_docx = _convert_doc_to_docx(filepath)
                source = tmp_docx

            result = md.convert(str(source))
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(result.text_content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path.name} ({len(result.text_content)} chars)")
        except Exception as e:
            print(f"  ✗ Lỗi convert {filepath.name}: {e}")
        finally:
            if tmp_docx and tmp_docx.exists():
                os.remove(tmp_docx)


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  (chưa có data/landing/news/ — bỏ qua)")
        return

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            header = (
                f"# {data.get('title', 'Unknown')}\n\n"
                f"**Source:** {data.get('url', 'N/A')}\n"
                f"**Author:** {data.get('author', 'N/A')}\n"
                f"**Date:** {data.get('date', 'N/A')}\n"
                f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            )
            content = header + data.get("content_markdown", "")
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path.name} ({len(content)} chars)")
        except Exception as e:
            print(f"  ✗ Lỗi convert {filepath.name}: {e}")


def convert_all():
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
