"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Cách tiếp cận nhẹ (không tải headless browser / model):
    requests  -> tải HTML
    trafilatura -> bóc tách nội dung chính + metadata (title, date, author)

Mỗi bài lưu 1 file JSON trong data/landing/news/ với metadata:
    url, title, author, date, date_crawled, content_markdown

Cách dùng:
    1. Điền danh sách link vào ARTICLE_URLS bên dưới.
    2. Chạy:  python -m src.task2_crawl_news
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Giả lập trình duyệt để tránh bị một số trang chặn request thô.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# =============================================================================
# Điền link bài báo cần crawl vào đây (>= 5 bài).
# =============================================================================
ARTICLE_URLS: list[str] = [
    "https://baochinhphu.vn/khoi-to-bat-tam-giam-ca-si-long-nhat-son-ngoc-minh-vi-to-chuc-su-dung-ma-tuy-102260520125739676.htm",
    "https://znews.vn/toan-canh-vu-miu-le-bi-bat-qua-tang-dung-ma-tuy-post1650763.html",
    "https://vietnamnet.vn/de-nghi-truy-to-ca-si-chi-dan-cung-anh-trai-vi-to-chuc-su-dung-ma-tuy-2434484.html",
    "https://thanhnien.vn/dien-vien-hai-tran-huu-tin-lanh-7-nam-6-thang-tu-185230428134549434.htm",
    "https://vietnamnet.vn/su-kien/vu-an-ca-si-chau-viet-cuong-434282.html",
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str, fallback: str) -> str:
    """Tạo tên file an toàn từ tiêu đề bài báo."""
    text = text or fallback
    text = text.lower()
    # bỏ dấu tiếng Việt cơ bản
    repl = {
        "[àáạảãâầấậẩẫăằắặẳẵ]": "a", "[èéẹẻẽêềếệểễ]": "e",
        "[ìíịỉĩ]": "i", "[òóọỏõôồốộổỗơờớợởỡ]": "o",
        "[ùúụủũưừứựửữ]": "u", "[ỳýỵỷỹ]": "y", "đ": "d",
    }
    for pat, ch in repl.items():
        text = re.sub(pat, ch, text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:60] or fallback)


def crawl_article(url: str) -> dict:
    """
    Crawl 1 bài báo -> dict metadata + nội dung markdown.

    Returns:
        {url, title, author, date, date_crawled, content_markdown}
    """
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    # Bóc nội dung chính ở dạng markdown (loại bỏ menu, quảng cáo, comment...).
    content_md = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        url=url,
    )
    if not content_md:
        raise ValueError(f"Không bóc được nội dung từ {url}")

    # Lấy metadata (title, author, date).
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta else None) or url
    author = (meta.author if meta else None) or ""
    date = (meta.date if meta else None) or ""

    return {
        "url": url,
        "title": title,
        "author": author,
        "date": date,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content_md,
    }


def crawl_all(urls: list[str] | None = None):
    """Crawl toàn bộ bài báo và lưu mỗi bài thành 1 file JSON."""
    urls = urls if urls is not None else ARTICLE_URLS
    setup_directory()

    saved = 0
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Crawling: {url}")
        try:
            article = crawl_article(url)
        except Exception as e:  # 1 bài lỗi không làm hỏng cả mẻ
            print(f"  ✗ Lỗi: {e}")
            continue

        slug = _slugify(article["title"], f"article-{i:02d}")
        filepath = DATA_DIR / f"{i:02d}-{slug}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✓ Saved: {filepath.name}  ({len(article['content_markdown'])} chars)")
        saved += 1

    print(f"\n✓ Done. Lưu được {saved}/{len(urls)} bài vào {DATA_DIR}")
    return saved


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("  Gợi ý nguồn: VnExpress, Tuổi Trẻ, Thanh Niên, Dân Trí, ...")
    else:
        crawl_all()
