"""Unit tests for retriever module."""
import pytest
from unittest.mock import AsyncMock, patch

from app.core.retriever import retrieve
from app.db.vector_store import DocumentChunk


def make_chunks(count: int) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            id=f"uuid-{i:03d}",
            source_file="test.pdf",
            chunk_index=i,
            content=f"Nội dung chunk {i}",
            similarity=0.9 - i * 0.05,
            metadata={},
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_retrieve_returns_chunks():
    """N-01: retrieve trả về danh sách chunks từ vector store"""
    with patch("app.core.retriever.search_similar_chunks", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = make_chunks(3)

        embedding = [0.1] * 768
        chunks = await retrieve(embedding, top_k=3)

        assert len(chunks) == 3
        mock_search.assert_called_once_with(query_embedding=embedding, top_k=3)


@pytest.mark.asyncio
async def test_retrieve_uses_default_top_k_from_config():
    """N-02: nếu top_k=None, dùng settings.top_k_default"""
    with (
        patch("app.core.retriever.search_similar_chunks", new_callable=AsyncMock) as mock_search,
        patch("app.core.retriever.settings") as mock_settings,
    ):
        mock_settings.top_k_default = 7
        mock_search.return_value = []

        await retrieve([0.1] * 768)

        mock_search.assert_called_once_with(query_embedding=[0.1] * 768, top_k=7)


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_no_chunks():
    """A-01: không có chunks → trả về list rỗng"""
    with patch("app.core.retriever.search_similar_chunks", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []

        chunks = await retrieve([0.1] * 768, top_k=5)

        assert chunks == []
