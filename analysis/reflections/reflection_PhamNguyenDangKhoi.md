# Individual Reflection — Lab 18

**Tên:** Phạm Nguyễn Đăng Khôi
**MSSV:** 2A202601243
**Module phụ trách:** M1 → M5 (làm bài cá nhân, implement toàn bộ 5 module + pipeline)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M1 Chunking, M2 Hybrid Search, M3 Reranking, M4 RAGAS Eval, M5 Enrichment, cộng với việc tối ưu `src/pipeline.py`.

- **Các hàm/class chính đã viết:**

| Module | Hàm/class | Điểm cốt lõi |
|--------|-----------|--------------|
| M1 | `chunk_semantic()` | Tách câu bằng regex `(?<=[.!?])\s+|\n\n`, encode bằng `all-MiniLM-L6-v2`, cắt chunk mới khi cosine similarity giữa hai câu kề < `SEMANTIC_THRESHOLD`. |
| M1 | `chunk_hierarchical()` | Gộp paragraph → parent (2048 ký tự), cắt parent → child (256 ký tự); mỗi parent có `parent_id` ổn định (`parent_0`, `parent_1`…) và mọi child thuộc parent đó mang đúng id. |
| M1 | `chunk_structure_aware()` | Quét theo dòng (không dùng `re.split`) để bỏ qua `#` nằm trong fenced code block; giữ stack heading H1–H3 làm breadcrumb; bảng/list nằm nguyên trong body section. |
| M1 | `extract_doc_metadata()` + `format_source_header()` | Parse dòng `> Phiên bản: … \| Ngày hiệu lực: …` thành metadata, rồi render lại thành 1 dòng header nhét vào chính `text` của chunk. |
| M2 | `segment_vietnamese()` | `underthesea.word_tokenize(format="text")` rồi đổi `_` → space, có `try/except` fallback trả text gốc. |
| M2 | `BM25Search.index()/search()` | Tokenize corpus, `BM25Okapi`, lọc bỏ score = 0 trước khi cắt top-k. |
| M2 | `DenseSearch.index()/search()` | Tạo collection Qdrant (dim 1024 từ `config.py`), encode bằng `BAAI/bge-m3`, upsert payload `{**metadata, "text": …}`, search bằng `query_points()`. |
| M2 | `reciprocal_rank_fusion()` | `score(d) = Σ 1/(k + rank)` với `k=60`, gộp theo `text` làm key. |
| M3 | `CrossEncoderReranker` | Load `BAAI/bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder`, cache trong instance, `predict()` trên list `(query, doc_text)`, sort giảm dần, cắt `RERANK_TOP_K=3`, giữ `original_score` + metadata + rank. |
| M4 | `evaluate_ragas()` | `Dataset.from_dict()` + 4 metric, bọc `try/except` để thiếu key/lỗi API không làm crash pipeline, fallback trả đủ 4 key số + `per_question` rỗng. |
| M4 | `failure_analysis()` | Sắp theo trung bình 4 metric, lấy bottom-N, xác định metric tệ nhất và map sang Diagnostic Tree → `diagnosis` + `suggested_fix`. |
| M5 | `_enrich_single_call()` | 1 API call trả JSON gồm `summary` + `questions` + `context` + `metadata` thay vì 4 call riêng; không có key thì rơi về fallback rule-based. |
| M5 | `enrich_chunks()` | Trả `list[EnrichedChunk]`, giữ nguyên `original_text`, merge metadata theo thứ tự `{**auto_meta, **metadata_gốc}` để LLM không ghi đè mất `source`/`version`. |

- **Số tests pass:** 37/37 (`pytest tests/ -v` — M1 13, M2 5, M3 5, M4 4, M5 10).

## 2. Kiến thức học được

- **Khái niệm mới nhất — RRF cộng theo *rank*, không cộng theo *score*.** Trước lab này tôi sẽ normalize BM25 score và cosine score rồi cộng có trọng số. Nhưng hai thang điểm đó không cùng ý nghĩa: BM25 không chặn trên và phụ thuộc độ dài corpus, cosine nằm trong [-1, 1]. RRF bỏ hẳn giá trị score, chỉ dùng thứ hạng, nên gộp được hai retriever mà không cần tune trọng số.

- **Điều bất ngờ nhất — pipeline "xịn" hơn không tự động cho điểm cao hơn.** Lần chạy đầu tiên, production pipeline (hierarchical chunking + hybrid + rerank + enrichment) **thua naive baseline ở cả 4 metric** (faithfulness 0.742 vs 0.858 baseline, tương tự ở 3 metric còn lại). Sau khi đổi sang structure-aware chunking + source header trong text + prompt cho phép suy luận có kiểm soát, production đảo lại thành thắng 3/4 metric (faithfulness 0.818, answer_relevancy 0.809, context_precision 0.946 so với baseline 0.774/0.769/0.925; chỉ context_recall giảm nhẹ 0.908 vs 0.925). Đây là bài học lớn nhất của lab: thêm component không phải là tối ưu; phải đo rồi mới biết component nào đang làm hỏng chất lượng. Chi tiết ở [`analysis/failure_analysis.md`](../failure_analysis.md).

- **Metadata phải nằm trong `text` thì LLM mới thấy.** Tôi viết `extract_doc_metadata()` rất kỹ để giữ `version` + `effective_date`, nhưng ở v1 chúng chỉ nằm trong payload Qdrant. Retriever match theo text, LLM đọc context là text — nên toàn bộ thông tin phiên bản coi như vô hình. Đó là lý do câu "Bao lâu phải đổi mật khẩu?" bị trả lời nước đôi "có hai chu kỳ 120 ngày và 90 ngày".

- **RAGAS `faithfulness` phạt cả câu trả lời "Không tìm thấy."** Khi model từ chối trả lời, không có claim nào để verify → điểm 0, kéo tụt aggregate mạnh hơn cả một câu trả lời sai một phần. Từ chối sai (false refusal) đắt hơn tôi tưởng.

- **`faithfulness` cũng phạt cả câu trả lời ĐÚNG nếu answer cần suy luận số học.** Ở bottom-5 sau khi fix, 3/5 case (mua thiết bị 55 triệu → CEO; tạm ứng phạt 300k; hoàn trả chi phí đào tạo 25 triệu) có answer khớp 100% ground truth nhưng vẫn bị `faithfulness` chấm 0.22–0.33, vì bước "55 triệu thuộc khoảng trên 50 triệu" là suy luận, không phải câu trích nguyên văn từ context. Bài học: điểm RAGAS thấp không luôn đồng nghĩa pipeline sai — phải đọc answer thật trước khi kết luận cần sửa retrieval hay prompt.

- **Kết nối bài giảng:** phần Advanced Chunking (semantic vs hierarchical vs structure-aware), Hybrid Search + RRF, Cross-encoder reranking (bi-encoder nhanh để lấy nhiều ứng viên → cross-encoder chậm để lọc ít ứng viên tốt), và RAGAS Error Tree (Answer sai → Context có bằng chứng? → lỗi ở retrieval hay ở generation?).

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất: production pipeline thua baseline.** Test đều xanh, không có exception, nhưng cả 4 metric đều thấp hơn baseline. Không có traceback nào để lần theo.
  - **Cách giải quyết:** đọc `failures` trong `reports/ragas_report.json` thay vì đọc code. Bottom-5 chỉ ra ba nguyên nhân tách bạch: (1) child chunk 256 ký tự cắt đôi bảng ngưỡng phê duyệt trong `mua_sam.md`; (2) chunk con mất dòng phiên bản; (3) prompt "CHỈ dựa trên context" làm model từ chối cả khi context đã có bằng chứng nhưng cần một bước suy luận. Fix từng cái một, ghi lại số đo trước/sau.

- **`recreate_collection()` đã deprecated.** Scaffold gợi ý dùng `recreate_collection()` nhưng repo chạy qdrant-client 1.19 (deprecated từ 1.11).
  - **Cách giải quyết:** đổi sang `collection_exists()` → `delete_collection()` → `create_collection()`.

- **Windows console `cp1252` làm crash mọi lần print.** Mọi `print` có emoji hoặc dấu tiếng Việt đều raise `UnicodeEncodeError` trước khi pipeline chạy được dòng nào.
  - **Cách giải quyết:** `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` đặt trong `config.py` — file được mọi module import nên chỉ cần fix một chỗ.

- **`main.py` crash ở lần chạy thứ hai.** `os.rename()` raise `FileExistsError` trên Windows khi `reports/ragas_report.json` đã tồn tại → đổi sang `os.replace()`.

- **`check_lab.py` báo "không chạy được tests".** `subprocess.run(timeout=120)` quá ngắn: riêng việc load `bge-m3` + `bge-reranker-v2-m3` đã mất hơn 2 phút → nâng timeout lên 1800 và ép `encoding="utf-8"` cho subprocess.

- **`.gitignore` nuốt mất deliverable.** Dòng `reports/*.json` khiến hai file report bắt buộc nộp không được commit → thêm hai dòng `!reports/ragas_report.json` và `!reports/naive_baseline_report.json`.

- **Thời gian debug:** phần lớn thời gian không nằm ở việc viết thuật toán mà ở môi trường (encoding, API deprecated, timeout) và ở vòng lặp đo → sửa → đo lại của pipeline.

## 4. Nếu làm lại

- **Sẽ làm khác:** chạy đủ vòng baseline → eval → đọc bottom-5 **ngay sau M1**, thay vì implement hết M1–M5 rồi mới eval một lần. Chiến lược chunking là quyết định ảnh hưởng lớn nhất tới cả context_recall lẫn context_precision, mà tôi lại chọn nó sớm nhất và kiểm chứng muộn nhất.
- **Sẽ tách biến khi A/B.** Lần đầu tôi đổi cùng lúc chunking + prompt nên không biết phần cải thiện đến từ đâu; đúng ra nên đổi từng biến một và lưu report riêng cho mỗi lần chạy (giờ đã lưu `reports/ragas_report_v1_hierarchical.json` để so sánh).
- **Module muốn thử tiếp:** M2 — thêm query rewriting/expansion trước khi search (câu hỏi multi-hop như "laptop 30 triệu cần ai duyệt và cần gì từ CNTT?" nên được tách thành hai truy vấn con), và thử metadata filter theo `effective_date` để loại thẳng document đã bị thay thế thay vì để LLM tự xử lý xung đột.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |

**Action plan áp dụng vào project của tôi:**

1. **Chunk theo cấu trúc tài liệu, không theo số ký tự cố định.** Với tài liệu có bảng/biểu mẫu, cắt theo heading và giữ bảng nguyên vẹn; chỉ fallback về cắt theo kích thước khi section quá lớn.
2. **Đưa nguồn + phiên bản vào chính text của chunk.** Metadata chỉ nằm trong payload là metadata mà retriever và LLM không bao giờ nhìn thấy.
3. **Prompt phải phân biệt "bịa" với "suy luận trên context".** Cấm cả hai sẽ tạo ra false refusal, và RAGAS phạt false refusal rất nặng.
4. **Có sẵn baseline và luôn so với nó.** Không có baseline thì không biết component mới là cải thiện hay là hồi quy.
5. **`temperature=0` cho mọi lần eval.** Nếu không, chênh lệch giữa hai lần chạy có thể là nhiễu sampling chứ không phải hiệu quả của thay đổi.
