# Failure Analysis — Lab 18: Production RAG

**Người thực hiện:** Phạm Nguyễn Đăng Khôi (2A202601243)
**Bài cá nhân** — tự implement và tối ưu toàn bộ M1 → M5 + pipeline.

---

## RAGAS Scores

Cả hai report chạy trên cùng `test_set.json` (20 câu), cùng generation model `gpt-4o-mini`.
Naive baseline dùng `temperature` mặc định (1.0, không set); production pipeline set `temperature=0`
để kết quả lặp lại được — đây là một phần lý do Δ ở dưới không chỉ phản ánh riêng cải tiến retrieval.

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7738 | 0.8178 | **+0.0440** |
| Answer Relevancy | 0.7694 | 0.8086 | **+0.0391** |
| Context Precision | 0.9250 | 0.9458 | **+0.0208** |
| Context Recall | 0.9250 | 0.9083 | **-0.0167** |

Naive = `chunk_basic` (paragraph, ~500 ký tự) + dense-only search, top-3, không rerank, không enrichment.
Production = `chunk_structure_aware` (cắt theo heading, giữ bảng/list nguyên vẹn, có source header trong
text) + M5 enrichment (1 call/chunk) + Hybrid (BM25 + dense + RRF, top-20) + CrossEncoder rerank (top-3).

### Ghi chú quan trọng: lần chạy đầu tiên (v1) đã THUA baseline ở cả 4 metric

Trước khi ra được bảng trên, lần chạy production đầu tiên — dùng `chunk_hierarchical` (child 256 ký tự)
và prompt generation tối giản — cho kết quả **thấp hơn naive baseline ở cả 4/4 metric** (lưu lại làm
bằng chứng ở `reports/ragas_report_v1_hierarchical.json`: faithfulness 0.742, answer_relevancy 0.728,
context_precision 0.904, context_recall 0.892). Bài học chính của lab này không phải "ghép đủ 5 module
là xong", mà là: thêm component (hybrid search, rerank, enrichment) **không tự động** cải thiện chất
lượng — phải đọc `failures` trong report để tìm đúng module đang làm hỏng pipeline. Ba nguyên nhân tìm
được từ bottom-5 của v1, và cách sửa trong `src/pipeline.py`:

1. **Chunking cắt đôi bảng ngưỡng phê duyệt.** `chunk_hierarchical` với `child_size=256` cắt bảng
   "Thẩm quyền phê duyệt" trong `mua_sam.md` giữa chừng → câu hỏi "laptop 30 triệu ai phê duyệt" bị trả
   lời "Không tìm thấy." dù thông tin có trong tài liệu. **Fix:** đổi sang `chunk_structure_aware` — cắt
   theo heading, bảng/list nằm nguyên trong 1 chunk (kiểm chứng: 107 sections, avg 217 ký tự, max 565,
   0 section nào bị cắt giữa bảng).
2. **Chunk con mất dấu vết phiên bản.** `format_source_header()` đã viết sẵn ở M1 nhưng chưa được gọi
   trong pipeline → chỉ chunk đầu tiên của mỗi document giữ dòng "Phiên bản / Ngày hiệu lực", các chunk
   sau (đa số) mất hoàn toàn. LLM không phân biệt được `nghi_phep_nam_v2023` vs `v2024`. **Fix:** gọi
   `format_source_header()` và ghép vào đầu text của MỌI chunk trước khi index.
3. **Prompt quá chặt gây từ chối sai (false refusal).** "Trả lời CHỈ dựa trên context" bị model hiểu là
   cấm luôn cả suy luận đơn giản trên context (tra bảng ngưỡng, cộng số ngày thâm niên) → trả
   "Không tìm thấy." dù context đã đủ bằng chứng. RAGAS chấm `faithfulness = 0` cho câu không có claim
   nào để verify, kéo tụt aggregate mạnh hơn một câu trả lời sai một phần. **Fix:** viết lại
   `SYSTEM_PROMPT` trong `src/pipeline.py` phân biệt rõ "bịa thêm dữ kiện" (cấm) với "suy luận trên
   dữ kiện có trong context" (cho phép), và thêm quy tắc chọn phiên bản có ngày hiệu lực mới nhất khi
   context có mâu thuẫn.

Sau ba fix trên, production đảo ngược thành thắng 3/4 metric (bảng ở trên). `context_recall` giảm nhẹ
-0.0167 — nguyên nhân cụ thể ở case #2 dưới đây: structure-aware chunking tách sạch section theo
heading, nhưng hybrid search + rerank không có cơ chế ưu tiên "section liền kề trong cùng document",
nên với câu hỏi chỉ hỏi 1 phần của chính sách, phần phụ (VD: ảnh hưởng bảo hiểm) nằm ở section khác
trong cùng file dễ bị rớt khỏi top-3.

## Bottom-5 Failures

(Từ `reports/ragas_report.json`, sắp theo `avg_score` tăng dần — tệ nhất trước.)

### #1 — avg_score 0.625, worst_metric = answer_relevancy (0.00)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** "...nhân viên có 9 năm thâm niên sẽ được 18 ngày phép (15 + 3). Về lương, không có thông tin cụ thể trong context về mức lương cho nhân viên Senior, vì vậy không thể xác định được lương trong khoảng nào."
- **Error Tree:**
  1. Output sai? → Nửa đầu (18 ngày) **đúng chính xác**. Nửa sau (lương) → "không tìm thấy" — không hallucinate, nhưng không trả lời được.
  2. Context có bằng chứng? → Cả 3 context đều là `nghi_phep_nam_v2023`/`v2024` (thâm niên + số ngày phép). **Không có chunk nào từ `bang_luong_2024.md`** (bảng lương Senior P3-P4).
  3. Query OK? → Câu hỏi là **multi-hop 2 chủ đề** (nghỉ phép + lương) nhưng hybrid search chỉ chạy **1 lần embed/1 lần BM25** cho toàn câu → vector câu hỏi bị chủ đề "nghỉ phép" (xuất hiện nhiều từ khóa hơn) áp đảo, "lương Senior" gần như biến mất trong query vector.
- **Root cause:** Retrieval — single-vector search không xử lý được câu hỏi ghép 2 chủ đề độc lập (M2).
- **Suggested fix:** Query decomposition — tách câu hỏi ghép thành 2 sub-query ("nghỉ phép thâm niên 9 năm", "lương Senior") trước khi search, retrieve + rerank riêng từng sub-query rồi gộp context. Có thể kiểm tra lại bằng cách thêm assertion trong test set: multi-hop question phải có context chứa cả 2 nguồn tài liệu liên quan.

### #2 — avg_score 0.704, worst_metric = faithfulness (0.50)
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** "Nghỉ phép không lương 20 ngày cần phê duyệt của Giám đốc điều hành (CEO)." — **đúng** phần được hỏi, chỉ thiếu phần bảo hiểm (không được hỏi trực tiếp nhưng ground truth có).
- **Error Tree:**
  1. Output sai? → Phần trả lời cho câu hỏi thực tế là đúng 100%.
  2. Context có bằng chứng? → Đã kiểm tra nguyên văn cả 3 context: **không context nào chứa câu "tự đóng phần bảo hiểm"**. Câu đó nằm ở section `## Ảnh hưởng đến phúc lợi` — section thứ 3, riêng biệt, trong CÙNG file `nghi_phep_khong_luong.md` — nhưng không được retrieval/rerank chọn vào top-3.
  3. Query OK? → Query chỉ hỏi "ai phê duyệt" nên đúng ngữ nghĩa retrieval nhắm vào section "Quy trình phê duyệt" là hợp lý; retrieval không "sai", nhưng ground_truth của test set kỳ vọng thêm thông tin phụ mà câu hỏi không trực tiếp gợi ý.
- **Root cause:** Giới hạn của `RERANK_TOP_K=3` kết hợp việc hybrid search không có cơ chế ưu tiên "section liền kề cùng document" khi 1 section đã match mạnh.
- **Suggested fix:** Sau khi rerank chọn được top-k, thêm 1 bước lấy thêm các section cùng `source` (cùng file markdown) đứng ngay sau/trước section đã chọn, tăng nhẹ `RERANK_TOP_K` (3→4) cho câu hỏi chính sách có nhiều mệnh đề phụ. Có thể verify bằng cách so `context_recall` trước/sau khi bật cơ chế "sibling-section retrieval".

### #3 — avg_score 0.747, worst_metric = faithfulness (0.33)
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Đơn hàng trị giá 55 triệu VNĐ thuộc khoảng 'Trên 50.000.000 VNĐ', do đó cần phê duyệt của Tổng Giám đốc (CEO)." — **khớp ground truth**, không sai dữ kiện.
- **Error Tree:**
  1. Output sai? → Không — đúng cả số tiền, đúng người phê duyệt.
  2. Context có bằng chứng? → Có, chunk chứa bảng ngưỡng phê duyệt đầy đủ (`mua_sam.md`).
  3. Query OK? → Có.
- **Root cause:** Đây là **hạn chế của bản thân metric `faithfulness`**, không phải lỗi pipeline: RAGAS kiểm tra từng claim trong answer có được context "nói thẳng" (NLI-style) hay không. Câu "55 triệu thuộc khoảng trên 50 triệu" là một bước **suy luận số học** (55 > 50) — kết luận đúng nhưng không phải câu chữ trích nguyên văn từ context, nên bị model giám khảo của RAGAS chấm claim đó là "không được context hỗ trợ trực tiếp".
- **Suggested fix:** Không sửa pipeline cho case này — sửa sẽ làm answer dài dòng hơn không cần thiết. Ghi nhận đây là điểm cần đọc RAGAS `faithfulness` một cách có phê phán: điểm thấp không luôn đồng nghĩa answer sai, đặc biệt với câu hỏi cần so ngưỡng/số học. Nếu muốn tăng điểm giả tạo có thể ép model trích dẫn nguyên văn dòng ngưỡng trước khi kết luận, nhưng đánh đổi là answer dài và kém tự nhiên hơn.

### #4 — avg_score 0.763, worst_metric = faithfulness (0.22)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** "...Thời gian quá hạn là 5 ngày (20 - 15). Phí phạt = 15.000.000 × 2% = 300.000 VNĐ..." — khớp số liệu chính (5 ngày quá hạn, 300.000 VNĐ) với ground truth.
- **Error Tree:** Cùng dạng lỗi với #3: answer đúng về mặt số học và khớp ground truth, nhưng mỗi bước tính (trừ ngày, nhân %) là suy luận trên context chứ không phải trích dẫn nguyên văn → `faithfulness` giám khảo chấm thấp cho các claim tính toán.
- **Root cause:** Metric artifact (RAGAS faithfulness khó chấm điểm cao cho câu trả lời cần nhiều bước tính toán số học), không phải lỗi retrieval/generation.
- **Suggested fix:** Theo dõi thêm case tương tự qua nhiều lần eval; nếu tỷ lệ này lớn và ảnh hưởng đến điểm tổng thể có hệ thống, có thể bổ sung một custom metric rule-based (so khớp số trong answer vs ground_truth) làm thước đo phụ, thay vì chỉ dựa vào RAGAS cho câu hỏi tính toán.

### #5 — avg_score 0.775, worst_metric = faithfulness (0.33)
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** "Nhân viên nghỉ việc trước thời hạn cam kết 1 năm, do đó phải hoàn trả 100% chi phí đào tạo... 25.000.000 VNĐ." — khớp hoàn toàn ground truth.
- **Error Tree / Root cause:** Cùng nhóm lỗi metric-artifact như #3, #4 — answer đúng, suy luận đúng (8 tháng < 1 năm cam kết → hoàn trả 100%), nhưng bước so sánh "8 tháng < 1 năm" không phải câu trích nguyên văn từ context.
- **Suggested fix:** Không cần sửa pipeline; đây là bằng chứng cho thấy RAGAS `faithfulness` hệ thống hoá phạt câu hỏi cần một bước so sánh/số học, cần ghi rõ trong báo cáo để không đọc sai là "model đang hallucinate" khi thực ra không phải.

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"

**Error Tree walkthrough:**
1. **Output đúng?** → Đúng một nửa: 18 ngày phép đúng 100%; phần lương bị trả "không tìm thấy" — không hallucinate, nhưng không trả lời được ý thứ hai.
2. **Context đúng?** → Sai ở đây: cả 3 context đều thuộc `nghi_phep_nam_v2023`/`v2024`, không context nào từ `bang_luong_2024.md`.
3. **Query rewrite OK?** → Đây là gốc vấn đề: hệ thống chưa có bước rewrite/decompose. Query gốc là câu ghép 2 ý (nghỉ phép + lương) được nhúng thành **1 vector duy nhất**; tín hiệu "nghỉ phép" áp đảo tín hiệu "lương Senior" nên retrieval chỉ mang về tài liệu nghỉ phép.
4. **Fix ở bước:** M2 (retrieval) — cần query decomposition cho câu hỏi multi-hop trước khi search, không phải ở M1 (chunk đã đúng và đủ chi tiết) hay M3 (rerank không cứu được vì candidate set đầu vào đã thiếu tài liệu lương).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm bước phát hiện câu hỏi multi-hop (heuristic: có liên từ "và" nối 2 cụm danh từ chủ đề khác nhau, hoặc dùng LLM nhỏ để tách) → chạy hybrid search + rerank riêng cho từng sub-query, gộp context trước khi generate. Đây là fix trực tiếp cho case #1 và một phần case #2.
- Thêm "sibling-section retrieval": sau rerank, nếu 1 chunk được chọn có `source` giống chunk khác top-K bị loại, ưu tiên giữ lại section liền kề cùng file — giảm rủi ro mất context_recall như case #2.
