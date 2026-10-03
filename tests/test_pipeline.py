"""Unit tests for RAG pipeline logic."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.pipeline import run_rag_pipeline
from app.db.vector_store import DocumentChunk


def make_chunk(similarity: float, content: str = "Nội dung chunk mẫu") -> DocumentChunk:
    return DocumentChunk(
        id="uuid-001",
        source_file="test.pdf",
        chunk_index=0,
        content=content,
        similarity=similarity,
        metadata={},
    )


# ── Normal Cases ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_answered_when_similarity_above_threshold():
    """N-01: câu hỏi hợp lệ, chunks đủ ngưỡng → status=answered"""
    with (
        patch("app.core.pipeline.embed_text", new_callable=AsyncMock) as mock_embed,
        patch("app.core.pipeline.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.core.pipeline.generate_answer", new_callable=AsyncMock) as mock_gen,
    ):
        mock_embed.return_value = [0.1] * 768
        mock_retrieve.return_value = [make_chunk(0.85)]
        mock_gen.return_value = ("Câu trả lời mẫu", "gemini-2.0-flash")

        response = await run_rag_pipeline("Cách xử lý nước ao?")

        assert response.query_status == "answered"
        assert response.answer == "Câu trả lời mẫu"
        assert response.top_similarity == 0.85
        assert response.model_version == "gemini-2.0-flash"
        assert len(response.retrieved_chunks) == 1
        assert response.processing_time_ms >= 0


# ── Abnormal Cases ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_low_match_when_similarity_below_threshold():
    """A-01: similarity dưới ngưỡng → status=low_match"""
    with (
        patch("app.core.pipeline.embed_text", new_callable=AsyncMock) as mock_embed,
        patch("app.core.pipeline.retrieve", new_callable=AsyncMock) as mock_retrieve,
    ):
        mock_embed.return_value = [0.1] * 768
        mock_retrieve.return_value = [make_chunk(0.3)]  # below threshold 0.5

        response = await run_rag_pipeline("Câu hỏi không liên quan lắm")

        assert response.query_status == "low_match"
        assert response.answer is None
        assert response.top_similarity == 0.3
        assert len(response.retrieved_chunks) == 1


@pytest.mark.asyncio
async def test_no_source_when_no_chunks_returned():
    """A-02: không có chunk nào retrieve được → status=no_source"""
    with (
        patch("app.core.pipeline.embed_text", new_callable=AsyncMock) as mock_embed,
        patch("app.core.pipeline.retrieve", new_callable=AsyncMock) as mock_retrieve,
    ):
        mock_embed.return_value = [0.1] * 768
        mock_retrieve.return_value = []

        response = await run_rag_pipeline("Câu hỏi bất kỳ")

        assert response.query_status == "no_source"
        assert response.answer is None
        assert response.top_similarity is None
        assert response.retrieved_chunks == []


@pytest.mark.asyncio
async def test_error_when_embedding_fails():
    """A-03: embedding API lỗi → status=error, không raise exception ra ngoài"""
    from app.utils.errors import EmbeddingError

    with patch("app.core.pipeline.embed_text", new_callable=AsyncMock) as mock_embed:
        mock_embed.side_effect = EmbeddingError("API timeout")

        response = await run_rag_pipeline("Câu hỏi bất kỳ")

        assert response.query_status == "error"
        assert response.answer is None


@pytest.mark.asyncio
async def test_error_when_llm_fails():
    """A-04: LLM call lỗi → status=error"""
    from app.utils.errors import LLMError

    with (
        patch("app.core.pipeline.embed_text", new_callable=AsyncMock) as mock_embed,
        patch("app.core.pipeline.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.core.pipeline.generate_answer", new_callable=AsyncMock) as mock_gen,
    ):
        mock_embed.return_value = [0.1] * 768
        mock_retrieve.return_value = [make_chunk(0.9)]
        mock_gen.side_effect = LLMError("LLM rate limit")

        response = await run_rag_pipeline("Câu hỏi bất kỳ")

        assert response.query_status == "error"
        assert response.answer is None


# ── Boundary Cases ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_similarity_exactly_at_threshold():
    """B-01: similarity = threshold chính xác → answered (>= threshold)"""
    with (
        patch("app.core.pipeline.embed_text", new_callable=AsyncMock) as mock_embed,
        patch("app.core.pipeline.retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch("app.core.pipeline.generate_answer", new_callable=AsyncMock) as mock_gen,
    ):
        mock_embed.return_value = [0.1] * 768
        mock_retrieve.return_value = [make_chunk(0.5)]  # exactly threshold
        mock_gen.return_value = ("Câu trả lời", "gemini-2.0-flash")

        response = await run_rag_pipeline("Câu hỏi biên")

        assert response.query_status == "answered"


@pytest.mark.asyncio
async def test_custom_top_k_is_passed_to_retriever():
    """B-02: top_k custom được truyền đúng vào retriever"""
    with (
        patch("app.core.pipeline.embed_text", new_callable=AsyncMock) as mock_embed,
        patch("app.core.pipeline.retrieve", new_callable=AsyncMock) as mock_retrieve,
    ):
        mock_embed.return_value = [0.1] * 768
        mock_retrieve.return_value = []

        await run_rag_pipeline("Câu hỏi", top_k=10)

        mock_retrieve.assert_called_once_with([0.1] * 768, top_k=10)
