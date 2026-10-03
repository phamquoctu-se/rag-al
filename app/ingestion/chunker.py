from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.config import settings
from app.core.embedder import embed_document_texts
from app.ingestion.loader import RawDocument


@dataclass
class TextChunk:
    source_file: str
    chunk_index: int
    content: str
    metadata: dict


def _recursive_chunks(doc: RawDocument, chunk_size: int, overlap: int) -> list[TextChunk]:
    """Keep the previous character splitter available as a configured fallback."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        length_function=len,
    )
    texts = splitter.split_text(doc.content)
    return [
        TextChunk(
            source_file=doc.source_file,
            chunk_index=index,
            content=text.strip(),
            metadata={
                **doc.metadata,
                "chunk_size_chars": len(text.strip()),
                "chunking_strategy": "recursive",
            },
        )
        for index, text in enumerate(texts)
        if text.strip()
    ]


def _split_sentences(text: str) -> list[str]:
    """Split Vietnamese prose while retaining punctuation in each sentence."""
    normalized = re.sub(r"[ \t\r\f\v]+", " ", text.strip())
    if not normalized:
        return []
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?…])\s+", normalized)
        if part.strip()
    ]


def _hard_split(text: str, limit: int) -> list[str]:
    """Split an oversized sentence on words, with a final character fallback."""
    parts: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.extend(word[start : start + limit] for start in range(0, len(word), limit))
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _semantic_units(text: str, unit_target: int) -> list[str]:
    """Create small paragraph-aware units whose embeddings can be compared."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    units: list[str] = []

    for paragraph in paragraphs:
        sentences = _split_sentences(paragraph)
        current = ""
        for sentence in sentences:
            sentence_parts = (
                _hard_split(sentence, unit_target) if len(sentence) > unit_target else [sentence]
            )
            for part in sentence_parts:
                candidate = f"{current} {part}".strip()
                if current and len(candidate) > unit_target:
                    units.append(current)
                    current = part
                else:
                    current = candidate
        if current:
            units.append(current)

    return units


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without adding NumPy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _overlap_tail(text: str, max_chars: int) -> str:
    """Reuse complete trailing sentences when they fit the overlap budget."""
    if max_chars <= 0:
        return ""
    sentences = _split_sentences(text)
    selected: list[str] = []
    current_length = 0
    for sentence in reversed(sentences):
        added = len(sentence) + (1 if selected else 0)
        if current_length + added > max_chars:
            break
        selected.append(sentence)
        current_length += added
    return " ".join(reversed(selected))


def _assemble_chunks(
    units: list[str],
    break_before: set[int],
    min_size: int,
    max_size: int,
    overlap: int,
) -> list[tuple[str, int]]:
    """Build bounded chunks, honoring semantic boundaries when size permits."""
    assembled: list[tuple[str, int]] = []
    current_units: list[str] = []
    semantic_unit_count = 0

    for index, unit in enumerate(units):
        current_text = "\n\n".join(current_units)
        candidate = f"{current_text}\n\n{unit}".strip()
        semantic_break = index in break_before and len(current_text) >= min_size
        size_break = bool(current_units) and len(candidate) > max_size

        if semantic_break or size_break:
            assembled.append((current_text, semantic_unit_count))
            tail = _overlap_tail(current_text, overlap)
            current_units = [tail] if tail and len(f"{tail}\n\n{unit}") <= max_size else []
            semantic_unit_count = 0

        current_units.append(unit)
        semantic_unit_count += 1

    if current_units:
        final_text = "\n\n".join(current_units)
        # The trailing chunk may be smaller than min_size when a genuine topic
        # boundary precedes it. Merging it back would undo that semantic split.
        assembled.append((final_text, semantic_unit_count))

    return assembled


async def chunk_document(
    doc: RawDocument,
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_chunk_size: int | None = None,
    max_chunk_size: int | None = None,
    breakpoint_percentile: float | None = None,
) -> list[TextChunk]:
    """Split a document with semantic boundaries, or use the configured fallback."""
    strategy = settings.chunking_strategy.strip().lower()
    ovlp = overlap if overlap is not None else settings.chunk_overlap

    if strategy == "recursive":
        size = chunk_size if chunk_size is not None else settings.chunk_size
        return _recursive_chunks(doc, size, ovlp)
    if strategy != "semantic":
        raise ValueError(f"Unsupported chunking strategy: {settings.chunking_strategy}")

    min_size = min_chunk_size if min_chunk_size is not None else settings.min_chunk_size
    max_size = max_chunk_size if max_chunk_size is not None else settings.max_chunk_size
    percentile = (
        breakpoint_percentile
        if breakpoint_percentile is not None
        else settings.semantic_breakpoint_percentile
    )

    if min_size < 1 or max_size < min_size:
        raise ValueError("Chunk sizes must satisfy 1 <= min_chunk_size <= max_chunk_size")
    if ovlp < 0 or ovlp >= max_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < max_chunk_size")
    if not 0 <= percentile <= 100:
        raise ValueError("semantic_breakpoint_percentile must be between 0 and 100")
    if not doc.content.strip():
        return []

    unit_target = min(300, max(120, min_size // 2))
    units = _semantic_units(doc.content, unit_target)
    if not units:
        return []

    break_before: set[int] = set()
    if len(units) > 1:
        embeddings = await embed_document_texts(units)
        distances = [
            1.0 - _cosine_similarity(embeddings[index], embeddings[index + 1])
            for index in range(len(embeddings) - 1)
        ]
        # Equal distances contain no relative topic shift, so size limits alone
        # decide boundaries in that case.
        if max(distances) - min(distances) > 1e-9:
            threshold = _percentile(distances, percentile)
            break_before = {
                index + 1 for index, distance in enumerate(distances) if distance >= threshold
            }

    assembled = _assemble_chunks(units, break_before, min_size, max_size, ovlp)
    return [
        TextChunk(
            source_file=doc.source_file,
            chunk_index=index,
            content=content,
            metadata={
                **doc.metadata,
                "chunk_size_chars": len(content),
                "chunking_strategy": "semantic",
                "semantic_breakpoint_percentile": percentile,
                "semantic_unit_count": unit_count,
            },
        )
        for index, (content, unit_count) in enumerate(assembled)
    ]
