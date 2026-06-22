# Individual Reflection — Lab 18: Production RAG

**Tên:** Lam
**Ngày:** 2026-06-22
**Phạm vi:** Implement toàn bộ M1→M5 + pipeline, chạy end-to-end (Naive + Production), failure analysis.
**Stack thực tế:** `gemini-3-flash` (AntcoAI Gateway, OpenAI-compatible) · retrieval `bge-m3` · rerank `bge-reranker-v2-m3` · RAGAS 0.4.3 (judge embeddings = MiniLM local) · Qdrant in-memory (không có Docker).

---

## Phần 1 — Mapping bài giảng → Code

| Lecture Concept | Module | Hàm cụ thể | Observation (số liệu thật) |
|----------------|--------|------------|----------------------------|
| Semantic chunking | M1 | `chunk_semantic()` ([src/m1_chunking.py](../../src/m1_chunking.py)) | Threshold 0.85 + embedding MiniLM tạo **208 chunk** (avg 99 ký tự) vs basic **51 chunk** (avg 410). Threshold cao ⇒ tách rất nhỏ; trong production tôi chọn **hierarchical** cho cân bằng. |
| Hierarchical (parent/child) | M1 | `chunk_hierarchical()` | Parent ≤2048, child ≤256 → **99 child** (avg 210), mỗi child có `parent_id` hợp lệ. Đây là chiến lược dùng cho pipeline (104 child sau khi gộp toàn corpus). |
| Structure-aware | M1 | `chunk_structure_aware()` | Parse `^#{1,3}` → **106 chunk** giữ nguyên header/section; max 789 ký tự (giữ trọn bảng → tốt cho dữ liệu numeric). |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()`, `segment_vietnamese()` | RRF `1/(k+rank+1)` gộp 2 danh sách → `method="hybrid"`. **Bài học VN:** underthesea nối từ ghép bằng `_` (`nghỉ_phép`) → phải `replace("_"," ")` nếu không BM25 không khớp query 2 token. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder` (KHÔNG FlagEmbedding — crash với transformers 5.x). top-20→top-3, sort theo `rerank_score`. Trên corpus nhỏ rerank gần như không đổi top-3 (hybrid đã đúng). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: faith **0.92**, relevancy **0.63**, precision **0.90**, recall **0.82**. Metric **thấp nhất = answer_relevancy** vì gemini trả lời quá súc tích + RAGAS sinh 1 thay vì 3 câu hỏi ngược. |
| Contextual embeddings | M5 | `contextual_prepend()`, `_enrich_single_call()` | Combined mode 1 call/chunk (bonus). Trên corpus sạch, contextual prepend **giảm nhẹ recall (−0.033)** vì làm loãng tín hiệu chunk — kỹ thuật này lợi cho corpus dài/nhiễu (benchmark Anthropic −49% failure). |

**Kết luận mapping:** "Production ≠ luôn tốt hơn". Pipeline phức tạp chỉ thắng khi corpus đủ lớn/nhiễu để baseline thất bại. Ở đây baseline dense-only đã đạt precision 0.92 nên không còn nhiều dư địa.

---

## Phần 2 — Khó khăn & Cách giải quyết

1. **Xung đột `transformers 4.41.2` ↔ `tokenizers 0.23.1`.**
   - *Exact error:* `ImportError: tokenizers>=0.19,<0.20 is required ... but found tokenizers==0.23.1`.
   - *Debug:* `pip show tokenizers` → thấy `Required-by: chromadb, litellm, langchain-mistralai` ⇒ **không được downgrade** tokenizers (vỡ project khác). Giải pháp: **nâng** `transformers→5.12.1` + `sentence-transformers→5.6.0` cho khớp tokenizers mới.

2. **RAGAS 0.4.3 import lỗi VertexAI.**
   - *Exact error:* `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` (ragas import `ChatVertexAI` từ path đã bị gỡ ở langchain_community 0.4.x).
   - *Debug/Fix:* Vì dùng gateway OpenAI-compatible (không dùng VertexAI), tôi **inject 1 module stub** `ChatVertexAI` rỗng trước khi import ragas (`_stub_vertexai()` trong [src/m4_eval.py](../../src/m4_eval.py)). Đồng thời ragas 0.4.x đổi API → viết theo `EvaluationDataset` + `LangchainLLMWrapper` thay vì `datasets.Dataset` như scaffold cũ.

3. **`gemini-3-flash` là reasoning model — tốn token suy luận.**
   - *Triệu chứng:* `max_tokens` nhỏ → content rỗng (reasoning ăn hết budget); test HyQA/summarize fail.
   - *Fix:* nâng `max_tokens` (summary 600, combined 1200), parse JSON bóc ```code fence```, và ép câu hỏi HyQA kết thúc bằng `?`.

4. **Không có Docker/Qdrant.**
   - *Fix:* thêm fallback `QdrantClient(location=":memory:")` khi `QDRANT_INMEMORY=1` trong `DenseSearch.__init__` → chạy pipeline đầy đủ không cần Docker.

5. **Kiến thức thiếu:** API RAGAS 0.4 (metrics dạng class, EvaluationDataset) và cách cắm custom LLM/embeddings. Bổ sung bằng cách `dir(ragas.metrics)` + `inspect.signature(evaluate)` để dò API thực tế thay vì đoán theo doc cũ.

**Thời gian debug:** dependency hell (~40 phút) là phần tốn nhất, không phải logic module.

---

## Phần 3 — Action Plan cho project cá nhân

### Project: RAG trợ lý chính sách nội bộ (HR/IT/Finance tiếng Việt)

#### Hiện tại
- Pipeline: dense-only embedding + top-k, chưa có rerank/eval định lượng.
- Known issues: câu hỏi numeric/multi-hop sai; không đo được chất lượng; không phân biệt version tài liệu (v2023 vs v2024).

#### Plan áp dụng (rút ra từ lab này)
1. [ ] **Chunking:** `hierarchical` (child 256 / parent 2048) làm mặc định; `structure-aware` riêng cho tài liệu có **bảng số** (lương, ngưỡng phê duyệt) để không cắt bảng.
2. [ ] **Search:** **Hybrid BM25(underthesea)+Dense(bge-m3)+RRF** — bắt buộc cho tiếng Việt (BM25 bắt từ khoá số/định danh, dense bắt ngữ nghĩa).
3. [ ] **Reranking:** `bge-reranker-v2-m3`, top-20→top-5 (không phải top-3) cho câu multi-hop cần nhiều mảnh context.
4. [ ] **Evaluation:** RAGAS 4 metric nhưng **dùng bge-m3 làm judge embedding** (tránh artifact answer_relevancy=0); thêm test set numeric/version/negation; chạy CI mỗi lần đổi prompt.
5. [ ] **Enrichment:** **Auto-metadata (category/version/date)** là ưu tiên #1 để **metadata-filter** lọc nhiễu (giải quyết câu hỏi #4) và xử lý version. Contextual-prepend chỉ bật cho tài liệu dài.
6. [ ] **Generation:** few-shot cho 3 dạng numeric (phạt quá hạn, hoàn đào tạo pro-rata, bậc phê duyệt theo ngưỡng) — vì lab cho thấy **lỗi lớn nằm ở generation, không phải retrieval**.

#### Timeline
- **Tuần 1 (23–29/06):** Hybrid search + hierarchical chunking + test set 30 câu có RAGAS CI.
- **Tuần 2 (30/06–06/07):** Auto-metadata + metadata filter + version routing (chọn tài liệu hiện hành).
- **Tuần 3 (07–13/07):** Reranker + few-shot numeric + tối ưu prompt theo failure analysis.

---

## Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|:-------------:|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Problem solving (dependency hell, gateway, RAGAS API) | 5 |
| Phân tích kết quả (dám chỉ ra production không thắng baseline) | 5 |
