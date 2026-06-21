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


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


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
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
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

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    # TODO: Implement semantic chunking
    # 1. Split text into sentences:
    #    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    #
    # 2. Encode sentences:
    #    from sentence_transformers import SentenceTransformer
    #    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast
    #    embeddings = model.encode(sentences)
    #
    # 3. Compare consecutive sentences:
    #    from numpy import dot
    #    from numpy.linalg import norm
    #    def cosine_sim(a, b): return dot(a, b) / (norm(a) * norm(b))
    #
    # 4. Group sentences:
    #    current_group = [sentences[0]]
    #    for i in range(1, len(sentences)):
    #        sim = cosine_sim(embeddings[i-1], embeddings[i])
    #        if sim < threshold:
    #            chunks.append(Chunk(text=" ".join(current_group), metadata=...))
    #            current_group = []
    #        current_group.append(sentences[i])
    #    # Don't forget last group
    #
    # 5. Return chunks with metadata: {"chunk_index": i, "strategy": "semantic"}
    return []


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    # TODO: Implement hierarchical chunking
    # 1. Split text into parents:
    #    paragraphs = text.split("\n\n")
    #    Gom paragraphs cho đến khi đạt parent_size → 1 parent chunk
    #    pid = f"parent_{p_index}"
    #    parent = Chunk(text=parent_text, metadata={**metadata, "chunk_type": "parent", "parent_id": pid})
    #
    # 2. Split each parent into children:
    #    Slide window child_size trên parent text
    #    child = Chunk(text=child_text, metadata={**metadata, "chunk_type": "child"}, parent_id=pid)
    #
    # 3. Return (parents_list, children_list)
    #
    # Production pattern:
    #   - Index CHILDREN vào vector DB (nhỏ → embedding chính xác)
    #   - Khi retrieve child → lookup parent_id → trả parent cho LLM (đủ context)
    return [], []


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    # TODO: Implement structure-aware chunking
    # 1. Split by markdown headers:
    #    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    #
    # 2. Pair headers with their content:
    #    chunks = []
    #    current_header = ""
    #    current_content = ""
    #    for part in sections:
    #        if re.match(r'^#{1,3}\s+', part):
    #            if current_content.strip():
    #                chunks.append(Chunk(
    #                    text=f"{current_header}\n{current_content}".strip(),
    #                    metadata={**metadata, "section": current_header, "strategy": "structure"}
    #                ))
    #            current_header = part.strip()
    #            current_content = ""
    #        else:
    #            current_content += part
    #    # Don't forget last section
    #
    # 3. Return chunks — mỗi chunk = 1 section hoàn chỉnh
    #
    # Ưu điểm: giữ nguyên tables, lists, code blocks
    # Dùng khi: corpus có structured documents (docs, API refs, manuals)
    return []


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    # TODO: Implement comparison
    # 1. For each doc, run: chunk_basic, chunk_semantic, chunk_hierarchical, chunk_structure_aware
    # 2. Collect stats: num_chunks, avg_length, min_length, max_length
    # 3. Print comparison table:
    #    Strategy      | Chunks | Avg Len | Min | Max
    #    basic         |   12   |   420   | 100 | 500
    #    semantic      |    8   |   580   | 200 | 900
    #    hierarchical  | 5p/15c |   256   | 100 | 2048
    #    structure     |   10   |   450   | 150 | 800
    # 4. Return results dict
    return {}


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
