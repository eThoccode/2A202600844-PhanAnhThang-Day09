"""Day 8 — RAG Pipeline v2: Pháp luật ma tuý & tin tức nghệ sĩ."""

# Console Windows mặc định là cp1252 -> không in được tiếng Việt / emoji.
# Ép stdout/stderr về UTF-8 cho mọi script trong package.
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
