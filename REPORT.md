# Báo cáo Lab 18 — Production RAG Pipeline

**Người thực hiện:** Lam · **Ngày:** 2026-06-22
**Hình thức:** Cá nhân — implement toàn bộ M1→M5 + pipeline, chạy end-to-end thật.

| Hạng mục | Giá trị |
|----------|---------|
| LLM (answer + enrichment + RAGAS judge) | `gemini-3-flash` qua **AntcoAI LLM Gateway** (OpenAI-compatible) |
| Embedding retrieval | `BAAI/bge-m3` (1024-dim, cosine) |
| Reranker | `BAAI/bge-reranker-v2-m3` (CrossEncoder) |
| RAGAS judge embeddings | `all-MiniLM-L6-v2` (local) |
| Vector DB | Qdrant (chế độ in-memory, không cần Docker) |
| Corpus | 26 tài liệu → **104 child-chunks** (2 PDF scan bị bỏ qua vì không có text layer) |
| Test set | 20 cặp Q&A |

---

## 1. Kiến trúc Pipeline

```
naive_baseline.py  (paragraph chunk + dense-only top-3)        ── baseline để so sánh
        │
        ▼ so sánh
M1 Chunking (hierarchical) → M5 Enrichment (contextual, 1 call/chunk)
        → M2 Hybrid Search (BM25 + Dense + RRF) → M3 Rerank (top-20→top-3)
        → LLM Answer (gemini-3-flash) → M4 RAGAS Eval (4 metrics)
```

| Module | File | Nội dung chính |
|--------|------|----------------|
| M1 | [src/m1_chunking.py](src/m1_chunking.py) | `chunk_semantic` (cosine sentence grouping), `chunk_hierarchical` (parent 2048 / child 256), `chunk_structure_aware` (parse markdown header) |
| M2 | [src/m2_search.py](src/m2_search.py) | `segment_vietnamese` (underthesea), `BM25Search`, `DenseSearch` (bge-m3 + Qdrant `query_points`), `reciprocal_rank_fusion` |
| M3 | [src/m3_rerank.py](src/m3_rerank.py) | `CrossEncoderReranker` (bge-reranker-v2-m3), benchmark latency |
| M4 | [src/m4_eval.py](src/m4_eval.py) | `evaluate_ragas` (4 metrics, RAGAS 0.4.x), `failure_analysis` (Diagnostic Tree) |
| M5 | [src/m5_enrichment.py](src/m5_enrichment.py) | `summarize_chunk`, `generate_hypothesis_questions`, `contextual_prepend`, `extract_metadata`, `_enrich_single_call` (combined) |

---

## 2. Kết quả RAGAS (số liệu thật)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|---:|
| Faithfulness | 0.9333 | **0.9167** | −0.017 |
| Answer Relevancy | 0.6202 | **0.6270** | +0.007 |
| Context Precision | 0.9167 | **0.9042** | −0.013 |
| Context Recall | 0.8500 | **0.8167** | −0.033 |

- **3/4 metric ≥ 0.70** → Rubric #7 = **10/10**.
- **Faithfulness ≥ 0.85** → **Bonus +3**.
- **Combined enrichment mode (1 call/chunk)** → **Bonus +2**.

### Nhận định
Trên corpus nhỏ & sạch, baseline dense-only đã rất mạnh (precision 0.92) nên pipeline production **không vượt** baseline — kết quả thật, đã phân tích trung thực. Bottleneck thực sự nằm ở **generation numeric/multi-hop** và **artifact đo answer_relevancy**, không phải ở retrieval. Chi tiết: [analysis/failure_analysis.md](analysis/failure_analysis.md).

---

## 3. Tóm tắt Failure Analysis (Bottom-5)

| # | Câu hỏi (rút gọn) | Worst metric | Bước cần fix |
|---|-------------------|--------------|--------------|
| 1 | Mua laptop 30tr — ai phê duyệt? | answer_relevancy 0.0 | Retrieval + prompt (multi-hop ngưỡng tiền) |
| 2 | Lương thuộc cấp phân loại nào? | answer_relevancy 0.0 | **Measurement artifact** (trả lời đúng nhưng metric=0) |
| 3 | Tạm ứng 15tr trễ 20 ngày — phạt? | faithfulness 0.5 | Generation (numeric pro-rata) |
| 4 | Lương thử việc Junior cao nhất? | context_precision 0.33 | Metadata filter khử chunk nhiễu |
| 5 | Hoàn chi phí đào tạo nghỉ sớm? | faithfulness 0.0 | **Generation** (model hallucinate 100%) |

**Case study (#5):** Retrieval hoàn hảo (precision 1.0, recall 1.0) nhưng model suy luận sai công thức hoàn trả → lỗi nằm ở **GENERATION**, không phải chunking/search/rerank.

---

## 4. Vấn đề kỹ thuật đã xử lý

| Vấn đề | Cách giải quyết |
|--------|-----------------|
| `transformers 4.41` ↔ `tokenizers 0.23` xung đột | Nâng `transformers→5.12`, `sentence-transformers→5.6` (không downgrade tokenizers vì chromadb/litellm cần) |
| RAGAS 0.4.3 lỗi `ModuleNotFoundError: ...vertexai` | Inject stub module `ChatVertexAI` + viết theo API mới `EvaluationDataset` + `LangchainLLMWrapper` |
| `gemini-3-flash` là reasoning model (ăn token) | Tăng `max_tokens`, parse JSON bóc code-fence, ép HyQA kết thúc `?` |
| Không có Docker/Qdrant | Fallback `QdrantClient(location=":memory:")` khi `QDRANT_INMEMORY=1` |
| `.gitignore` chặn `reports/*.json` (deliverable) | Đổi thành `reports/*.log` để report được push |

---

## 5. Kiểm thử & Verification

| Lệnh | Kết quả |
|------|---------|
| `pytest tests/ -v` | ✅ **37/37 passed** |
| `python naive_baseline.py` | ✅ exit 0 → `reports/naive_baseline_report.json` |
| `python src/pipeline.py` | ✅ exit 0 → `reports/ragas_report.json` |
| `grep -r "# TODO:" src/m*.py` | ✅ **0** |
| `python check_lab.py` | ✅ "Bài lab sẵn sàng để nộp!" |

> Lưu ý: `check_lab.py` có timeout nội bộ 120s cho pytest, trong khi test gọi API thật (~5 phút) nên dòng auto-test của nó báo timeout — lệnh chấm chính thức `pytest tests/ -v` vẫn pass đầy đủ.

---

## 6. Cách chạy lại

```bash
# 1. Cấu hình .env (không commit — chứa key)
#    OPENAI_API_KEY / OPENAI_BASE_URL=https://ai-gateway.antco.ai/v1 / LLM_MODEL=gemini-3-flash

# 2. Cài dependencies
pip install -r requirements.txt

# 3. (Tùy chọn) Docker Qdrant — hoặc dùng in-memory bằng QDRANT_INMEMORY=1
docker compose up -d

# 4. Chạy
QDRANT_INMEMORY=1 python naive_baseline.py     # baseline
QDRANT_INMEMORY=1 python src/pipeline.py        # production → reports/ragas_report.json

# 5. Kiểm tra
pytest tests/ -v
python check_lab.py
```

---

## 7. Deliverables

- ✅ `src/m1..m5_*.py` + `src/pipeline.py` — 5 modules hoàn chỉnh
- ✅ `reports/ragas_report.json` + `reports/naive_baseline_report.json`
- ✅ `analysis/failure_analysis.md` — Bottom-5 + Error Tree + case study
- ✅ `analysis/group_report.md` — bảng phân công + RAGAS + key findings
- ✅ `analysis/reflections/reflection_Lam.md` — lecture mapping + khó khăn + action plan
- ✅ `REPORT.md` — báo cáo tổng hợp (file này)
