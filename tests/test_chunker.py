"""Unit tests for semantic document chunking."""

from unittest.mock import AsyncMock, patch

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.loader import RawDocument


def make_document(content: str) -> RawDocument:
    return RawDocument(
        source_file="guide.pdf",
        content=content,
        metadata={"format": "pdf", "pages": 1},
    )


@pytest.mark.asyncio
async def test_semantic_distance_creates_topic_boundary():
    content = "\n\n".join(
        [
            "Ao nuôi và nguồn nước. " * 3,
            "Độ mặn và pH ổn định. " * 3,
            "Dấu hiệu bệnh trên tôm. " * 3,
        ]
    )
    embeddings = [
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
    ]

    with patch(
        "app.ingestion.chunker.embed_document_texts",
        new=AsyncMock(return_value=embeddings),
    ) as mock_embed:
        chunks = await chunk_document(
            make_document(content),
            min_chunk_size=100,
            max_chunk_size=500,
            overlap=0,
            breakpoint_percentile=80,
        )

    assert len(chunks) == 2
    assert "nguồn nước" in chunks[0].content
    assert "Độ mặn" in chunks[0].content
    assert "Dấu hiệu bệnh" in chunks[1].content
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[0].metadata["chunking_strategy"] == "semantic"
    mock_embed.assert_awaited_once()


@pytest.mark.asyncio
async def test_max_size_forces_split_without_semantic_shift():
    content = "\n\n".join(
        [
            "Quản lý môi trường ao nuôi ổn định mỗi ngày và ghi lại đầy đủ.",
            "Theo dõi nhiệt độ và độ mặn thường xuyên để phát hiện biến động.",
            "Kiểm tra thức ăn dư và sức khỏe của tôm sau từng cữ ăn.",
            "Thay nước có kiểm soát khi các chỉ số môi trường vượt ngưỡng.",
        ]
    )

    with patch(
        "app.ingestion.chunker.embed_document_texts",
        new=AsyncMock(return_value=[[1.0, 0.0]] * 4),
    ):
        chunks = await chunk_document(
            make_document(content),
            min_chunk_size=50,
            max_chunk_size=150,
            overlap=0,
        )

    assert len(chunks) >= 2
    assert all(len(chunk.content) <= 150 for chunk in chunks)


@pytest.mark.asyncio
async def test_empty_document_does_not_request_embeddings():
    with patch(
        "app.ingestion.chunker.embed_document_texts",
        new=AsyncMock(),
    ) as mock_embed:
        chunks = await chunk_document(make_document("  \n\n "))

    assert chunks == []
    mock_embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_semantic_limits_are_rejected():
    with pytest.raises(ValueError, match="min_chunk_size"):
        await chunk_document(
            make_document("Nội dung hợp lệ."),
            min_chunk_size=500,
            max_chunk_size=100,
        )


@pytest.mark.asyncio
async def test_recursive_strategy_remains_available():
    pytest.importorskip("langchain_text_splitters")
    with (
        patch("app.ingestion.chunker.settings") as mock_settings,
        patch(
            "app.ingestion.chunker.embed_document_texts",
            new=AsyncMock(),
        ) as mock_embed,
    ):
        mock_settings.chunking_strategy = "recursive"
        mock_settings.chunk_size = 80
        mock_settings.chunk_overlap = 10
        chunks = await chunk_document(make_document("Nội dung hướng dẫn. " * 20))

    assert len(chunks) > 1
    assert all(chunk.metadata["chunking_strategy"] == "recursive" for chunk in chunks)
    mock_embed.assert_not_awaited()
