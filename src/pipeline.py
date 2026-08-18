from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4.

Lịch sử tối ưu (số liệu đầy đủ ở analysis/failure_analysis.md):

- v1: chunk_hierarchical(child=256) + prompt "chỉ dựa trên context" tối giản.
      → thua naive baseline ở CẢ 4 metric. Ba nguyên nhân đọc được từ bottom-5:
      (a) child 256 ký tự cắt đôi bảng ngưỡng phê duyệt trong mua_sam.md và
          bang_luong_2024.md → context thiếu đúng dòng cần thiết;
      (b) chỉ chunk đầu của mỗi document giữ dòng "> Phiên bản: ... | Ngày hiệu
          lực: ..." → các chunk sau mất dấu vết version, LLM không phân biệt
          được nghi_phep_nam_v2023 vs v2024 và mat_khau_v1 vs v2;
      (c) prompt quá chặt → model trả "Không tìm thấy." ngay cả khi context đã
          có bằng chứng nhưng cần một bước suy luận (30 triệu ∈ 5-50 triệu).
- v2 (hiện tại): structure-aware chunking + source header nằm trong text +
      prompt cho phép suy luận có kiểm soát + quy tắc chọn version mới nhất.
"""

import os, sys, time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_structure_aware, format_source_header
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


# "CHỈ dựa trên context" vẫn là ràng buộc chính, nhưng phải nói rõ: suy luận
# TRÊN context (tra bảng, so ngưỡng, cộng trừ) không phải là bịa. Nếu không,
# mọi câu multi-hop đều bị trả "Không tìm thấy." dù context đã đủ bằng chứng.
SYSTEM_PROMPT = """Bạn là trợ lý tra cứu chính sách nội bộ. Quy tắc trả lời:

1. CHỈ dùng dữ kiện có trong context. Tuyệt đối không thêm con số hay điều kiện không xuất hiện trong context.
2. ĐƯỢC PHÉP suy luận từ context: tra bảng, xác định một giá trị thuộc khoảng nào, cộng/trừ/nhân theo công thức đã cho trong context.
   Ví dụ: context ghi "từ 5.000.000 - 50.000.000 VNĐ: Giám đốc phòng ban" thì đơn hàng 30 triệu thuộc khoảng đó.
3. Nếu context chứa nhiều phiên bản mâu thuẫn, chọn phiên bản có "Ngày hiệu lực" MỚI NHẤT, trả lời theo phiên bản đó và ghi rõ phiên bản cũ đã bị thay thế.
4. Chỉ trả lời "Không tìm thấy." khi context thực sự không có bằng chứng liên quan. Nếu chỉ trả lời được một phần, hãy trả lời phần đó rồi nói rõ phần còn lại không có trong context.
5. Trả lời ngắn gọn bằng tiếng Việt, nêu rõ con số và điều kiện cụ thể."""


# Latency breakdown đo ở lần chạy eval thật, không phải số ước lượng.
_LATENCY: dict[str, list[float]] = {"retrieval_ms": [], "rerank_ms": [], "generation_ms": []}


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    # Structure-aware thay cho hierarchical: data/ là markdown chính sách có
    # bảng ngưỡng phê duyệt + list quy trình. Cắt theo heading giữ nguyên bảng
    # và list, đồng thời mang theo breadcrumb "# Chính sách ... (Phiên bản 2024)".
    t0 = time.time()
    print("\n[1/4] Chunking documents (structure-aware + source header)...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        for sec in chunk_structure_aware(doc["text"], metadata=doc["metadata"]):
            # Header nguồn/phiên bản phải nằm trong TEXT: retriever và LLM chỉ
            # nhìn thấy text, không nhìn thấy payload metadata.
            header = format_source_header(sec.metadata)
            text = f"{header}\n{sec.text}" if header else sec.text
            all_chunks.append({"text": text, "metadata": sec.metadata})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t0 = time.perf_counter()
    results = search.search(query)
    _LATENCY["retrieval_ms"].append((time.perf_counter() - t0) * 1000)

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    _LATENCY["rerank_ms"].append((time.perf_counter() - t0) * 1000)

    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    from config import OPENAI_API_KEY
    t0 = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                # Eval phải lặp lại được: temperature mặc định 1.0 làm điểm nhảy
                # giữa hai lần chạy trên cùng một pipeline.
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    _LATENCY["generation_ms"].append((time.perf_counter() - t0) * 1000)
    return answer, contexts


def _latency_summary() -> dict:
    """Tổng hợp latency breakdown từ lần chạy eval vừa rồi."""
    out = {}
    for stage, samples in _LATENCY.items():
        if samples:
            out[stage] = {
                "avg": round(sum(samples) / len(samples), 1),
                "min": round(min(samples), 1),
                "max": round(max(samples), 1),
                "n": len(samples),
            }
    if out:
        out["total_avg_ms"] = round(sum(v["avg"] for v in out.values() if isinstance(v, dict)), 1)
    return out


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    latency = _latency_summary()
    if latency:
        print("\nLATENCY BREAKDOWN (per query, ms)")
        for stage, v in latency.items():
            if isinstance(v, dict):
                print(f"  {stage:<16} avg={v['avg']:>8.1f}  min={v['min']:>8.1f}  max={v['max']:>8.1f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, extra={"latency": latency} if latency else None)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
