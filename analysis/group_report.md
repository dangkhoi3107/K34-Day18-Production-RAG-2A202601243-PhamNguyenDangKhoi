# Group Report — Lab 18: Production RAG

**Nhóm:** Bài cá nhân — không có nhóm.
**Thành viên:** Phạm Nguyễn Đăng Khôi (2A202601243)
**Ngày:** 2026-08-19

## Thành viên & Phân công

Bài làm cá nhân: một mình phụ trách toàn bộ M1 → M5 và pipeline (`src/pipeline.py`).

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Phạm Nguyễn Đăng Khôi | M1: Chunking | ☑ | 13/13 |
| Phạm Nguyễn Đăng Khôi | M2: Hybrid Search | ☑ | 5/5 |
| Phạm Nguyễn Đăng Khôi | M3: Reranking | ☑ | 5/5 |
| Phạm Nguyễn Đăng Khôi | M4: Evaluation | ☑ | 4/4 |
| Phạm Nguyễn Đăng Khôi | M5: Enrichment | ☑ | 10/10 |

`pytest tests/ -v` → 37/37 pass.

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.7738 | 0.8178 | +0.0440 |
| Answer Relevancy | 0.7694 | 0.8086 | +0.0391 |
| Context Precision | 0.9250 | 0.9458 | +0.0208 |
| Context Recall | 0.9250 | 0.9083 | -0.0167 |

Chi tiết root cause của từng con số, bao gồm lần chạy v1 (thua baseline cả 4/4 metric) trước khi fix, ở
[`analysis/failure_analysis.md`](failure_analysis.md).

## Key Findings

1. **Biggest improvement:** Chuyển từ `chunk_hierarchical` (child 256 ký tự, cắt cơ học) sang
   `chunk_structure_aware` (cắt theo heading Markdown, giữ nguyên bảng/list) + gắn source header
   (nguồn + phiên bản + ngày hiệu lực) vào text của MỌI chunk. Riêng hai thay đổi này đảo pipeline từ
   thua baseline cả 4 metric sang thắng 3/4 metric — không phải do thêm component gì mới, mà do chunk
   không còn cắt đứt bảng ngưỡng phê duyệt và không còn làm mất dấu vết version giữa các chính sách
   2023/2024.

2. **Biggest challenge:** Test hẹp (`pytest`) xanh hết ở cả 5 module không đảm bảo pipeline production
   tốt hơn baseline — v1 pass toàn bộ test nhưng vẫn thua baseline cả 4/4 metric. Phải đọc `failures`
   trong `reports/ragas_report.json` (không phải đọc code) để tìm ra 3 nguyên nhân thật: chunk cắt bảng,
   metadata version bị mất, và prompt quá chặt gây từ chối sai ("Không tìm thấy." dù context đủ bằng
   chứng). RAGAS chấm `faithfulness = 0` cho câu từ chối, nên false refusal đắt hơn tưởng.

3. **Surprise finding:** Ba trong 5 câu ở bottom-5 sau khi fix (case #3, #4, #5 trong
   `failure_analysis.md`) có answer **khớp hoàn toàn ground truth** nhưng vẫn bị `faithfulness` chấm
   thấp (0.22–0.33), vì câu trả lời cần một bước suy luận số học/so ngưỡng (VD: "55 triệu ∈ khoảng trên
   50 triệu") mà RAGAS chỉ chấm điểm cao cho claim được context "nói thẳng" nguyên văn. Đây là hạn chế
   của metric, không phải lỗi pipeline — bài học: điểm RAGAS thấp không luôn đồng nghĩa answer sai.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** Naive faithfulness 0.774 / answer_relevancy 0.769 /
   context_precision 0.925 / context_recall 0.925 → Production 0.818 / 0.809 / 0.946 / 0.908. Thắng
   3/4 metric, context_recall giảm nhẹ 0.017 (nguyên nhân: xem case #2 trong failure_analysis.md).

2. **Biggest win — module nào, tại sao:** M1 (chunking). `chunk_structure_aware` giữ nguyên bảng ngưỡng
   phê duyệt và breadcrumb heading (mang theo tên chính sách + version) trong mỗi chunk — hai điều
   `chunk_hierarchical` (cắt theo số ký tự cố định) không đảm bảo được. Chunking quyết định trực tiếp
   context_recall và context_precision; rerank/hybrid search chỉ có thể chọn tốt trong số ứng viên đã
   có, không tạo ra thông tin mà chunk đã làm mất.

3. **Case study — 1 failure, Error Tree walkthrough:** Câu "Senior 9 năm thâm niên nghỉ bao nhiêu ngày
   và lương khoảng nào?" (avg_score 0.625, tệ nhất). Output đúng nửa đầu (18 ngày), sai/thiếu nửa sau
   (lương) → do context không có chunk `bang_luong_2024.md` nào cả → do query là câu ghép 2 chủ đề được
   nhúng thành 1 vector duy nhất, tín hiệu "nghỉ phép" áp đảo tín hiệu "lương Senior". Fix đúng ở bước
   M2 (retrieval): cần query decomposition cho câu hỏi multi-hop, không phải M1 hay M3.

4. **Next optimization nếu có thêm 1 giờ:** (a) Query decomposition cho câu hỏi multi-hop trước khi
   hybrid search — sửa trực tiếp case #1. (b) "Sibling-section retrieval": khi 1 section được rerank
   chọn, ưu tiên giữ thêm section liền kề cùng file nguồn — giảm rủi ro mất context_recall như case #2.
