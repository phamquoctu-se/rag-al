import time
from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.core.embedder import embed_text
from app.core.retriever import retrieve
from app.core.generator import generate_answer
from app.db.vector_store import DocumentChunk
from app.models.schemas import ChatResponse, RetrievedChunk
from app.utils.logger import logger
from app.utils.errors import EmbeddingError, LLMError


@dataclass
class PipelineResult:
    answer: Optional[str]
    query_status: str  # answered | no_source | low_match | error
    chunks: list[DocumentChunk]
    top_similarity: Optional[float]
    model_version: Optional[str]
    processing_time_ms: int
    error_message: Optional[str] = None


async def run_rag_pipeline(
    question: str,
    top_k: Optional[int] = None,
) -> ChatResponse:
    """
    Full RAG pipeline:
      1. Embed question
      2. Retrieve top-K similar chunks
      3. Check similarity threshold
      4. Generate answer with LLM (if threshold met)
      5. Return structured ChatResponse

    Args:
        question: KTV's question in Vietnamese
        top_k: Override number of chunks to retrieve

    Returns:
        ChatResponse ready to be returned by the API endpoint
    """
    start_ms = time.monotonic()

    k = top_k if top_k is not None else settings.top_k_default
    threshold = settings.similarity_threshold

    try:
        # ── Step 1: Embed the query ──────────────────────────────────────────
        logger.info(f"RAG pipeline started | question_len={len(question)} top_k={k}")
        query_embedding = await embed_text(question)

        # ── Step 2: Retrieve similar chunks ─────────────────────────────────
        chunks = await retrieve(query_embedding, top_k=k)

        top_similarity = chunks[0].similarity if chunks else None

        # ── Step 3: Threshold check ──────────────────────────────────────────
        if not chunks:
            answer, model_version = await generate_answer(question, [])
            elapsed = int((time.monotonic() - start_ms) * 1000)
            logger.info(f"RAG result=no_source ({elapsed}ms)")
            return ChatResponse(
                answer=answer,
                query_status="no_source",
                retrieved_chunks=[],
                top_similarity=None,
                model_version=model_version,
                processing_time_ms=elapsed,
            )

        if top_similarity < threshold:
            answer, model_version = await generate_answer(question, chunks)
            elapsed = int((time.monotonic() - start_ms) * 1000)
            logger.info(f"RAG result=low_match top_sim={top_similarity:.4f} ({elapsed}ms)")
            return ChatResponse(
                answer=answer,
                query_status="low_match",
                retrieved_chunks=_to_schema_chunks(chunks),
                top_similarity=top_similarity,
                model_version=model_version,
                processing_time_ms=elapsed,
            )

        # ── Step 4: Generate answer ───────────────────────────────────────────
        answer, model_version = await generate_answer(question, chunks)

        elapsed = int((time.monotonic() - start_ms) * 1000)
        logger.info(f"RAG result=answered top_sim={top_similarity:.4f} ({elapsed}ms)")

        return ChatResponse(
            answer=answer,
            query_status="answered",
            retrieved_chunks=_to_schema_chunks(chunks),
            top_similarity=top_similarity,
            model_version=model_version,
            processing_time_ms=elapsed,
        )

    except (EmbeddingError, LLMError) as e:
        elapsed = int((time.monotonic() - start_ms) * 1000)
        logger.error(f"RAG pipeline error: {e.message} ({elapsed}ms)")
        return ChatResponse(
            answer=None,
            query_status="error",
            retrieved_chunks=[],
            top_similarity=None,
            model_version=None,
            processing_time_ms=elapsed,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - start_ms) * 1000)
        logger.exception(f"Unexpected RAG pipeline error ({elapsed}ms)")
        return ChatResponse(
            answer=None,
            query_status="error",
            retrieved_chunks=[],
            top_similarity=None,
            model_version=None,
            processing_time_ms=elapsed,
        )


def _to_schema_chunks(chunks: list[DocumentChunk]) -> list[RetrievedChunk]:
    """Convert internal DocumentChunk dataclasses to Pydantic schema objects."""
    return [
        RetrievedChunk(
            source_file=c.source_file,
            chunk_index=c.chunk_index,
            content=c.content,
            similarity=c.similarity,
            metadata=c.metadata,
        )
        for c in chunks
    ]
