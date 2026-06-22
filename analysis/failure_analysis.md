# Failure Analysis — Lab 18: Production RAG

**Hình thức:** Cá nhân — **Lam** (implement toàn bộ M1→M5 + pipeline)
**LLM:** `gemini-3-flash` qua AntcoAI Gateway · **Embedding retrieval:** `BAAI/bge-m3` · **Reranker:** `BAAI/bge-reranker-v2-m3` · **RAGAS judge embeddings:** `all-MiniLM-L6-v2` (local)
**Test set:** 20 câu · **Corpus:** 26 tài liệu → 104 child-chunks (2 PDF scan bị bỏ qua vì không có text layer)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|---:|
| Faithfulness | 0.9333 | 0.9167 | −0.017 |
| Answer Relevancy | 0.6202 | 0.6270 | +0.007 |
| Context Precision | 0.9167 | 0.9042 | −0.013 |
| Context Recall | 0.8500 | 0.8167 | −0.033 |

> **Naive** = paragraph chunking + dense-only (bge-m3), top-3, không rerank, không enrichment.
> **Production** = hierarchical chunking → M5 enrichment (contextual prepend, 1 call/chunk) → hybrid BM25+Dense (RRF) → cross-encoder rerank top-3.

### Quan sát quan trọng (đọc kỹ)

Pipeline production **không cải thiện** so với baseline trên corpus này — thậm chí nhỉnh thấp hơn ở 3/4 metric. Đây là kết quả **thật** và đáng phân tích, không phải lỗi:

1. **Corpus nhỏ & sạch → baseline đã rất mạnh.** 104 chunk, mỗi chính sách gần như nằm gọn trong 1 tài liệu. Dense bge-m3 top-3 đã đủ để recall đúng đoạn → "trần" cải thiện cho hybrid/rerank rất thấp (context_precision/recall của baseline đã ~0.85–0.92).
2. **Contextual prepend (M5) thêm 1 câu mô tả vào đầu mỗi chunk** → làm loãng nhẹ tín hiệu embedding của chunk gốc, kéo context_recall xuống −0.033. Trên corpus lớn/nhiễu kỹ thuật này mới có lợi (benchmark Anthropic −49% retrieval failure áp dụng cho corpus dài).
3. **Reranker top-20→top-3** không đổi nhiều vì hybrid đã trả đúng đoạn trong top-3 sẵn.
4. **Bottleneck thật nằm ở 2 chỗ, không phải retrieval:**
   - **Answer Relevancy thấp (0.63) ở cả 2 pipeline** — đây là metric kém nhất. Nguyên nhân kép: (a) `gemini-3-flash` trả lời **quá súc tích** (ví dụ "17.000.000 VNĐ.") khiến RAGAS sinh câu hỏi ngược khó khớp; (b) RAGAS answer_relevancy đo bằng **embedding MiniLM** + cần LLM sinh 3 câu hỏi nhưng gateway thường **trả về 1** (log: *"LLM returned 1 generations instead of requested 3"*) → tín hiệu yếu, điểm tụt về 0 ở vài câu dù câu trả lời **đúng**.
   - **Câu hỏi numeric/multi-hop** (tính phạt tạm ứng, hoàn chi phí đào tạo theo cam kết) cần suy luận nhiều bước → faithfulness/recall giảm.

---

## Bottom-5 Failures

### #1 — avg 0.542 · worst = answer_relevancy (0.00)
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30tr ∈ [5–50tr] → **Giám đốc phòng ban (Director)** phê duyệt; cần xác nhận cấu hình của CNTT.
- **Got:** Liệt kê "yêu cầu CNTT" nhưng phần người phê duyệt bị cắt/không rõ ("Không...").
- **Metrics:** faith 0.83 · **relevancy 0.0** · precision 1.0 · recall 0.33
- **Error Tree:** Output sai một phần → Context đúng (precision 1.0) nhưng **thiếu mảnh "ngưỡng 5–50tr → Director"** (recall 0.33) → câu hỏi multi-hop (map số tiền → bậc phê duyệt) → **Fix ở retrieval + prompt**.
- **Root cause:** Multi-hop numeric: model không nối "30tr" với bảng ngưỡng phê duyệt; chunk chứa bảng ngưỡng không vào top-3.
- **Suggested fix:** Tăng `RERANK_TOP_K`=5, thêm few-shot hướng dẫn "đối chiếu số tiền với bảng ngưỡng"; chunk bảng ngưỡng ở dạng structure-aware để giữ nguyên bảng.

### #2 — avg 0.625 · worst = answer_relevancy (0.00)
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Dữ liệu **Bí mật** (cấm chia sẻ với đồng nghiệp).
- **Got:** "Thông tin lương thuộc cấp độ 3 (**Bí mật**)." → **ĐÚNG**.
- **Metrics:** faith 1.0 · **relevancy 0.0** · precision 1.0 · recall 0.5
- **Error Tree:** Output **đúng** → nhưng metric=0 → **lỗi đo lường (measurement artifact)**, không phải lỗi pipeline.
- **Root cause:** Câu trả lời quá ngắn + RAGAS chỉ sinh 1 câu hỏi ngược (thay vì 3) + embedding MiniLM cho điểm cosine thấp → answer_relevancy bị về 0 sai lệch.
- **Suggested fix:** Đổi embedding RAGAS sang bge-m3 (đa ngữ, mạnh tiếng Việt); ép model trả lời đủ câu ("Thông tin lương được phân loại là Bí mật."); dùng strict_mode/nhiều generations.

### #3 — avg 0.645 · worst = faithfulness (0.50)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày → 2%/tháng × 15tr = **300.000đ/tháng** (pro-rata).
- **Got:** "2%/tháng… **300.000 VNĐ**" → kết quả đúng nhưng thiếu giải thích pro-rata → 1 claim không truy được về context.
- **Metrics:** **faith 0.5** · relevancy 0.75 · precision 0.83 · recall 0.5
- **Error Tree:** Output gần đúng → context có công thức nhưng **thiếu mốc "thời hạn 15 ngày"** (recall 0.5) → model tự suy phần quá hạn → faithfulness giảm.
- **Root cause:** Numeric reasoning: model tính đúng nhưng "bịa" bước trung gian không có nguyên văn trong context.
- **Suggested fix:** Prompt "chỉ nêu con số có trong context, trình bày từng bước"; giảm temperature (đã 0); index thêm chunk chứa "thời hạn thanh toán 15 ngày".

### #4 — avg 0.661 · worst = context_precision (0.33)
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất 20tr → thử việc 85% × 20tr = **17tr**.
- **Got:** "17.000.000 VNĐ." → **ĐÚNG** (faith 1.0, recall 1.0).
- **Metrics:** faith 1.0 · relevancy 0.31 · **precision 0.33** · recall 1.0
- **Error Tree:** Output đúng → recall đủ → nhưng **top-3 lẫn 2 chunk nhiễu** (precision 0.33) + trả lời cụt → relevancy thấp.
- **Root cause:** Hybrid kéo vào chunk bảng lương khác (Senior/Mid) gần nghĩa; reranker chưa đẩy đúng chunk Junior lên #1.
- **Suggested fix:** Metadata filter theo `category`/`topic` (M5 auto-metadata) để lọc đúng bảng lương; tăng trọng số rerank.

### #5 — avg 0.701 · worst = faithfulness (0.00)
- **Question:** Được tài trợ khóa học 25tr, nghỉ sau 8 tháng hoàn thành khóa học, hoàn trả bao nhiêu?
- **Expected:** Cam kết tối thiểu 1 năm; nghỉ trước hạn → hoàn theo tỷ lệ thời gian còn lại (không phải 100%).
- **Got:** "Hoàn trả **25.000.000 VNĐ (100%)**" → **SAI** (model bỏ qua quy tắc pro-rata theo tháng còn lại).
- **Metrics:** **faith 0.0** · relevancy 0.80 · precision 1.0 · recall 1.0
- **Error Tree:** Output **sai** → Context **đúng & đủ** (precision 1.0, recall 1.0) → query OK → **lỗi ở bước generation/suy luận**, không phải retrieval.
- **Root cause:** Model đọc đúng đoạn nhưng suy luận sai công thức hoàn trả theo tỷ lệ (hallucinate "100%").
- **Suggested fix:** Few-shot ví dụ tính pro-rata; chain-of-thought có kiểm soát; hoặc tách bước "trích công thức → áp số".

---

## Case Study (cho trình bày)

**Question chọn phân tích: #5 — Hoàn trả chi phí đào tạo.**

**Error Tree walkthrough:**
1. **Output đúng?** → KHÔNG (model nói 100% = 25tr, đúng phải pro-rata theo thời gian còn lại của cam kết 1 năm).
2. **Context đúng?** → CÓ — precision 1.0, recall 1.0, đoạn quy định cam kết & cách hoàn trả đều có trong top-3.
3. **Query rewrite OK?** → CÓ — câu hỏi rõ, retrieval đúng tài liệu `hoan_chi_dao_tao.md`.
4. **Fix ở bước:** **GENERATION** (không phải chunking/search/rerank). Retrieval đã hoàn hảo; lỗi nằm ở khả năng suy luận numeric của LLM.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Đổi embedding judge của RAGAS sang **bge-m3** để loại bỏ artifact answer_relevancy=0 (ước tính kéo answer_relevancy từ 0.63 → ~0.8).
- Few-shot prompt cho 3 dạng numeric (phạt quá hạn, hoàn đào tạo pro-rata, bậc phê duyệt theo ngưỡng tiền).
- Bật **auto-metadata filter** (M5) theo `category` để khử chunk nhiễu (cải thiện precision ở #4).
