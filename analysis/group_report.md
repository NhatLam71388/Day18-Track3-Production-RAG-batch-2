# Group Report — Lab 18: Production RAG

**Hình thức:** Cá nhân (1 người thực hiện toàn bộ) · **Tên:** Lam
**Ngày:** 2026-06-22

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|:----------:|:----------:|
| Lam | M1: Chunking | ✅ | 13/13 |
| Lam | M2: Hybrid Search | ✅ | 5/5 |
| Lam | M3: Reranking | ✅ | 5/5 |
| Lam | M4: Evaluation | ✅ | 4/4 |
| Lam | M5: Enrichment | ✅ | 10/10 |

> **Tổng: 37/37 tests pass.** Stack thật: `gemini-3-flash` (AntcoAI Gateway) · `bge-m3` · `bge-reranker-v2-m3` · RAGAS 0.4.3 (judge embeddings = MiniLM local) · Qdrant in-memory.

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|------:|-----------:|---:|
| Faithfulness | 0.9333 | 0.9167 | −0.017 |
| Answer Relevancy | 0.6202 | 0.6270 | +0.007 |
| Context Precision | 0.9167 | 0.9042 | −0.013 |
| Context Recall | 0.8500 | 0.8167 | −0.033 |

- 3/4 metric ≥ 0.70 (Rubric #7 = 10/10) · Faithfulness ≥ 0.85 (Bonus +3) · Combined enrichment 1 call/chunk (Bonus +2).

## Key Findings

1. **Biggest improvement:** Trên corpus nhỏ/sạch này pipeline production **không vượt** baseline (baseline dense-only đã đạt precision 0.92, recall 0.85 → gần trần). "Win" lớn nhất về mặt năng lực là **hybrid BM25+Dense cho tiếng Việt** và **đo lường định lượng được bằng RAGAS** — thứ baseline không có.
2. **Biggest challenge:** Dependency hell — `transformers 4.41 ↔ tokenizers 0.23` và RAGAS 0.4.3 lỗi import `ChatVertexAI`. Giải quyết bằng nâng transformers→5.12 (không downgrade tokenizers vì project khác cần) + inject stub module VertexAI + viết theo API RAGAS 0.4.x mới.
3. **Surprise finding:** Lỗi lớn **không nằm ở retrieval mà ở generation**. Nhiều câu retrieval hoàn hảo (precision/recall = 1.0) nhưng model suy luận numeric sai (vd hoàn chi phí đào tạo: model nói 100% thay vì pro-rata). Ngoài ra `answer_relevancy` bị **artifact đo lường** = 0 dù câu trả lời đúng (do câu trả lời quá ngắn + RAGAS chỉ sinh 1/3 câu hỏi ngược + embedding MiniLM yếu).

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** xem bảng trên — 3/4 metric ≥ 0.70, faithfulness 0.92, answer_relevancy là metric yếu nhất (0.63) ở cả hai pipeline.
2. **Biggest win — module nào, tại sao:** M2 Hybrid Search + M4 RAGAS. Hybrid bắt được cả từ khoá số/định danh (BM25 underthesea) lẫn ngữ nghĩa (bge-m3); M4 cho phép đo và truy nguyên lỗi định lượng.
3. **Case study — 1 failure, Error Tree:** "Hoàn chi phí đào tạo khi nghỉ sớm" — Output sai (100%) → Context đúng & đủ (precision 1.0, recall 1.0) → Query OK → **lỗi ở bước GENERATION** (suy luận pro-rata), không phải chunking/search/rerank.
4. **Next optimization nếu có thêm 1 giờ:** (a) đổi embedding judge RAGAS sang bge-m3 để khử artifact answer_relevancy=0 (kỳ vọng 0.63→~0.8); (b) few-shot prompt cho 3 dạng numeric; (c) bật auto-metadata filter (M5) để khử chunk nhiễu (cải thiện precision).
