"""
Sinh "nhãn nguồn" thân thiện để hiển thị citation đẹp trên UI.

    legal  -> tên văn bản luật (vd: "Luật Phòng, chống ma túy 2021")
    news   -> tiêu đề bài báo (đọc từ heading '# ' trong file markdown đã chuẩn hoá)

Chỉ phục vụ HIỂN THỊ — không ảnh hưởng retrieval/generation/eval.
"""

import re
from functools import lru_cache
from pathlib import Path

STD_NEWS = Path(__file__).parent.parent / "data" / "standardized" / "news"

# Tên đẹp cho các văn bản luật đã thu thập.
LEGAL_MAP = {
    "12_2017_QH14_354053": "Bộ luật Hình sự (sửa đổi 2017)",
    "73_2021_QH14_445185": "Luật Phòng, chống ma túy 2021",
    "105_2021_ND-CP_496664": "Nghị định 105/2021/NĐ-CP",
    "28_2026_ND-CP_690473": "Nghị định 28/2026/NĐ-CP",
}


@lru_cache(maxsize=128)
def _news_title(stem: str) -> str | None:
    f = STD_NEWS / f"{stem}.md"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    return None


def friendly_label(source: str, doc_type: str = "unknown") -> str:
    """Tên nguồn dễ đọc từ tên file (vd '02-...md' -> tiêu đề bài báo)."""
    stem = re.sub(r"\.(md|pdf|docx?|json)$", "", source, flags=re.I)

    if stem in LEGAL_MAP:
        return LEGAL_MAP[stem]

    m = re.match(r"^(\d+)_(\d+)_ND-CP", stem)
    if m:
        return f"Nghị định {m.group(1)}/{m.group(2)}/NĐ-CP"
    m = re.match(r"^(\d+)_(\d+)_QH(\d+)", stem)
    if m:
        return f"Luật {m.group(1)}/{m.group(2)}/QH{m.group(3)}"

    title = _news_title(stem)
    if title:
        return title

    # Fallback: bỏ tiền tố số "02-", thay '-' thành khoảng trắng.
    s = re.sub(r"^\d+-", "", stem).replace("-", " ").replace("_", " ").strip()
    return s[:80] or source
