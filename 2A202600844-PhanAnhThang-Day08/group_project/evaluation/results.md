# RAG Evaluation Results

## Framework sử dụng

**DeepEval** với LLM judge = Gemini (`gemini-2.5-flash`), qua endpoint OpenAI-compatible.


## Overall Scores (A/B)

| Metric | A: Hybrid (semantic+BM25+RRF) | B: Dense-only (semantic) | Δ (A−B) |
|---|---|---|---|
| Faithfulness | 0.965 | 0.930 | +0.036 |
| Answer Relevancy | 0.981 | 0.944 | +0.037 |
| Contextual Recall | 0.833 | 0.833 | +0.000 |
| Contextual Precision | 0.815 | 0.916 | -0.101 |
| **Average** | **0.899** | **0.906** | **-0.007** |

**Config tốt hơn:** B — Dense-only (semantic).


> Ghi chú: trục A/B là **Hybrid vs Dense-only** (theo gợi ý README). Reranking (Task 7 — Jina) đã hiện thực và kiểm chứng (relevant 0.704 vs nhiễu 0.07) nhưng không đưa vào A/B vì quota Jina free đã hết trong quá trình test; PageIndex fallback cũng tắt khi eval để kết quả tái lập.


## Worst performers (Config A)

| Câu hỏi | Avg | Chi tiết |
|---|---|---|
| Tội tổ chức sử dụng trái phép chất ma tuý bị xử lý thế nào?… | 0.66 | Faithfulness=1.00, Answer Relevancy=1.00, Contextual Recall=0.00, Contextual Precision=0.64 |
| Cá nhân, gia đình có trách nhiệm gì trong phòng, chống ma tu… | 0.68 | Faithfulness=1.00, Answer Relevancy=1.00, Contextual Recall=0.00, Contextual Precision=0.70 |
| Hành vi trồng cây thuốc phiện, cây cần sa bị xử lý hình sự k… | 0.75 | Faithfulness=0.43, Answer Relevancy=0.89, Contextual Recall=1.00, Contextual Precision=0.68 |

## Phân tích & đề xuất cải tiến

- **Hybrid (A) vs Dense-only (B):** Δ ở Contextual Recall/Precision cho thấy việc thêm BM25 + RRF có giúp lấy đúng điều khoản (khớp số điều, thuật ngữ pháp lý) mà dense đôi khi bỏ sót hay không.
- Câu điểm thấp thường do **Contextual Recall** thấp: tài liệu chưa đủ chi tiết hoặc chunk_size cắt mất ngữ cảnh → thử tăng chunk_size hoặc bổ sung nguồn.
- Câu hỏi tin tức ngắn: **Answer Relevancy** cao nhưng Recall thấp do bài báo còn nhiễu (menu/quảng cáo) → bóc tách nội dung sạch hơn hoặc lọc theo loại tài liệu.
- **Bật lại reranking (Jina)** khi có quota để đẩy chunk nhiễu xuống, kỳ vọng tăng Contextual Precision.
