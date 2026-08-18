from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)

from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


# Model dùng cho semantic chunking. Cache ở module level: SentenceTransformer
# mất ~5s để load, gọi lại mỗi lần chunk_semantic() sẽ rất chậm.
_SEMANTIC_MODEL = None


def _get_semantic_model(name: str = "all-MiniLM-L6-v2"):
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        _SEMANTIC_MODEL = SentenceTransformer(name)
    return _SEMANTIC_MODEL


def extract_doc_metadata(text: str, source: str = "") -> dict:
    """Trích metadata cấp document từ header markdown.

    Mỗi policy trong data/ mở đầu bằng:
        # Chính sách nghỉ phép năm (Phiên bản 2024)
        > Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự

    Version + ngày hiệu lực là bằng chứng để xử lý xung đột v2023/v2024 và
    mat_khau v1/v2 — nếu không giữ lại thì chunk con mất hết dấu vết phiên bản.
    """
    meta: dict = {}
    if source:
        meta["source"] = source

    m = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if m:
        meta["doc_title"] = m.group(1).strip()

    m = re.search(r"^>\s*(.+)$", text, flags=re.MULTILINE)
    if m:
        for field_str in m.group(1).split("|"):
            key, sep, value = field_str.partition(":")
            if not sep:
                continue
            key, value = key.strip().lower(), value.strip()
            if not value:
                continue
            if key.startswith("phiên bản"):
                meta["version"] = value
            elif key.startswith("ngày hiệu lực"):
                meta["effective_date"] = value
            elif key.startswith("phòng ban"):
                meta["department"] = value
            elif key.startswith("trạng thái"):
                meta["status"] = value
    return meta


def format_source_header(metadata: dict) -> str:
    """Render 1 dòng nguồn để gắn vào đầu chunk text.

    Retriever/LLM chỉ nhìn thấy `text`, không nhìn thấy payload metadata →
    phải đưa nguồn + phiên bản vào chính text thì mới dùng được lúc generate.
    """
    parts = []
    if metadata.get("source"):
        parts.append(f"Nguồn: {metadata['source']}")
    if metadata.get("doc_title"):
        parts.append(f"Tài liệu: {metadata['doc_title']}")
    if metadata.get("version"):
        parts.append(f"Phiên bản: {metadata['version']}")
    if metadata.get("effective_date"):
        parts.append(f"Hiệu lực: {metadata['effective_date']}")
    if metadata.get("status"):
        parts.append(f"Trạng thái: {metadata['status']}")
    return f"[{' | '.join(parts)}]" if parts else ""


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        docs.append({"text": text, "metadata": extract_doc_metadata(text, os.path.basename(fp))})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": extract_doc_metadata(text, os.path.basename(fp))})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """

    # 1. Nếu metadata là None thì dùng dictionary rỗng
    metadata = metadata or {}

    # 2. Tách văn bản thành các câu, theo dấu 
    sentences = re.split(r'(?<=[.!?])\s+|\n\n', text)

    # 3. Xóa khoảng trắng và câu rỗng
    sentences = [s.strip() for s in sentences if s.strip()]

    # Không có câu nào thì không có chunk
    if not sentences:
        return []

    # 4. Load embedding model (cache ở module level, xem _get_semantic_model)
    model = _get_semantic_model()

    # 5. Chuyển tất cả câu câu thành vector
    embeddings = model.encode(sentences)

    # 6. Chunk đầu tiên bắt đầu bằng sentence đầu tiên
    groups = [[sentences[0]]]

    # 7. So sánh từng câu với câu đứng trước nó
    for i in range(1, len(sentences)):

        a = embeddings[i - 1]
        b = embeddings[i]

        # Cosine similarity
        similarity = dot(a, b) / (norm(a) * norm(b) + 1e-9)

        # Nếu hai câu khác chủ đề nhiều
        if similarity < threshold:
            # Tạo chunk mới
            groups.append([sentences[i]])
        else:
            # Nếu cùng chủ đề → thêm vào chunk hiện tại
            groups[-1].append(sentences[i])

    # 8. Chuyển từng group thành Chunk object
    chunks = []

    for i, group in enumerate(groups):
        chunks.append(
            Chunk(
                text=" ".join(group),
                metadata={
                    **metadata,
                    "strategy": "semantic",
                    "chunk_index": i,
                },
            )
        )

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    metadata = metadata or {}
    if not text or not text.strip():
        return ([], [])

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    # 1. Tạo Parent Chunks
    parents: list[Chunk] = []
    current_p = ""
    for para in paragraphs:
        if len(current_p) + len(para) > parent_size and current_p:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=current_p.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))
            current_p = ""
        current_p += para + "\n\n"

    if current_p.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=current_p.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))

    # 2. Tạo Child Chunks từ mỗi Parent
    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata.get("parent_id")
        # Tách câu trong parent text
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', parent.text) if s.strip()]
        if not sentences:
            sentences = [parent.text]

        current_c = ""
        for sent in sentences:
            if len(current_c) + len(sent) > child_size and current_c:
                children.append(Chunk(
                    text=current_c.strip(),
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid
                ))
                current_c = ""
            current_c += sent + " "

        if current_c.strip():
            children.append(Chunk(
                text=current_c.strip(),
                metadata={**metadata, "chunk_type": "child"},
                parent_id=pid
            ))

    return (parents, children)



# ─── Strategy 3: Structure-Aware Chunking ────────────────


_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Quét theo dòng thay vì re.split() để:
    - Bỏ qua "#" nằm trong fenced code block (```), tránh cắt đôi code block.
    - Giữ breadcrumb heading (H1 > H2 > H3): section con vẫn mang tiêu đề tài
      liệu, nhờ đó chunk "12 ngày phép" biết mình thuộc bản 2023 hay 2024.
    - Table / list nằm trong body của section nên không bao giờ bị cắt giữa.
    """
    metadata = metadata or {}
    if not text or not text.strip():
        return []

    sections: list[tuple[list[tuple[int, str, str]], str]] = []
    stack: list[tuple[int, str, str]] = []   # [(level, raw_line, clean_title)]
    buffer: list[str] = []
    in_fence = False

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append((list(stack), body))

    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue

        heading = None if in_fence else _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            # H2 mới đóng mọi H2/H3 trước đó nhưng vẫn nằm dưới H1 hiện tại
            stack = [h for h in stack if h[0] < level]
            stack.append((level, line.strip(), heading.group(2).strip()))
            buffer = []
        else:
            buffer.append(line)
    flush()

    if not sections:   # tài liệu không có heading → giữ nguyên 1 chunk
        return [Chunk(text=text.strip(),
                      metadata={**metadata, "section": "Introduction",
                                "section_path": "", "strategy": "structure",
                                "chunk_index": 0})]

    chunks = []
    for heads, body in sections:
        breadcrumb = "\n".join(h[1] for h in heads)
        titles = [h[2] for h in heads]
        chunks.append(Chunk(
            text=f"{breadcrumb}\n\n{body}" if breadcrumb else body,
            metadata={
                **metadata,
                "section": titles[-1] if titles else "Introduction",
                "section_path": " > ".join(titles),
                "heading_level": heads[-1][0] if heads else 0,
                "strategy": "structure",
                "chunk_index": len(chunks),
            },
        ))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
