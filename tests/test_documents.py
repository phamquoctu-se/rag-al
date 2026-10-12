from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.vector_store import insert_new_document
from app.ingestion.ingest import ingest_upload
from app.main import app
from app.utils.errors import DocumentConflictError, EmbeddingError, InvalidDocumentError

HEADERS = {"X-Internal-Key": "test-key", "Content-Type": "application/octet-stream"}


@pytest.fixture
async def client():
    with patch("app.dependencies.settings") as settings:
        settings.internal_api_key = "test-key"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            yield value


@pytest.mark.parametrize(
    "filename",
    ["guide.pdf", "guide.docx", "guide.md", "guide.TXT", "kỹ thuật.txt", "a" * 496 + ".txt"],
)
async def test_upload_success_and_cleanup(client, filename):
    paths = []

    async def ingest(path, *, source_file):
        assert source_file == filename
        assert path.read_bytes() == b"content"
        paths.append(path)
        return 2

    with patch("app.api.v1.endpoints.documents.ingest_upload", side_effect=ingest):
        response = await client.post(
            "/v1/documents", params={"filename": filename}, content=b"content", headers=HEADERS
        )
    assert response.status_code == 201
    assert response.json() == {"source_file": filename, "chunks_count": 2}
    assert not paths[0].parent.exists()


@pytest.mark.parametrize(
    "filename",
    [
        "../guide.txt",
        "folder\\guide.txt",
        "guide\x00.txt",
        "guide.doc",
        "guide.exe",
        "a" * 497 + ".txt",
    ],
)
async def test_invalid_filename(client, filename):
    with patch("app.api.v1.endpoints.documents.ingest_upload", new_callable=AsyncMock) as ingest:
        response = await client.post(
            "/v1/documents", params={"filename": filename}, content=b"content", headers=HEADERS
        )
    assert response.status_code in (400, 422)
    ingest.assert_not_awaited()


@pytest.mark.parametrize("key, expected", [(None, 422), ("wrong", 401)])
async def test_auth_guard(client, key, expected):
    headers = {**HEADERS}
    if key is None:
        headers.pop("X-Internal-Key")
    else:
        headers["X-Internal-Key"] = key
    with patch("app.api.v1.endpoints.documents.ingest_upload", new_callable=AsyncMock) as ingest:
        response = await client.post(
            "/v1/documents?filename=guide.txt", content=b"content", headers=headers
        )
    assert response.status_code == expected
    ingest.assert_not_awaited()


async def test_upload_limits(client):
    with (
        patch("app.api.v1.endpoints.documents.MAX_FILE_BYTES", 8),
        patch(
            "app.api.v1.endpoints.documents.ingest_upload", new_callable=AsyncMock, return_value=1
        ) as ingest,
    ):
        for body, status in [(b"", 400), (b"12345678", 201), (b"123456789", 413)]:
            response = await client.post(
                "/v1/documents?filename=guide.txt", content=body, headers=HEADERS
            )
            assert response.status_code == status
        assert ingest.await_count == 1


async def test_streaming_limit_without_content_length(client):
    async def body():
        yield b"1234"
        yield b"56789"

    with (
        patch("app.api.v1.endpoints.documents.MAX_FILE_BYTES", 8),
        patch("app.api.v1.endpoints.documents.ingest_upload", new_callable=AsyncMock) as ingest,
    ):
        response = await client.post(
            "/v1/documents?filename=guide.txt", content=body(), headers=HEADERS
        )
    assert response.status_code == 413
    ingest.assert_not_awaited()


async def test_requires_binary_content_type(client):
    response = await client.post(
        "/v1/documents?filename=guide.txt", json={}, headers={"X-Internal-Key": "test-key"}
    )
    assert response.status_code == 415


@pytest.mark.parametrize(
    "error, expected",
    [
        (InvalidDocumentError("secret"), 400),
        (DocumentConflictError("secret"), 409),
        (EmbeddingError("secret"), 503),
        (RuntimeError("secret"), 503),
    ],
)
async def test_safe_errors_and_cleanup(client, error, expected):
    paths = []

    async def fail(path, **kwargs):
        paths.append(path)
        raise error

    with patch("app.api.v1.endpoints.documents.ingest_upload", side_effect=fail):
        response = await client.post(
            "/v1/documents?filename=guide.txt", content=b"content", headers=HEADERS
        )
    assert response.status_code == expected
    assert "secret" not in response.text
    assert not paths[0].parent.exists()


async def test_real_text_ingestion_uses_logical_source_and_embeddings(tmp_path):
    target = tmp_path / "document.txt"
    target.write_text("Shrimp water quality guidance.", encoding="utf-8")
    with (
        patch("app.ingestion.chunker.settings") as settings,
        patch(
            "app.ingestion.ingest.embed_document_chunk",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
        patch(
            "app.ingestion.ingest.insert_new_document", new_callable=AsyncMock, return_value=1
        ) as store,
    ):
        settings.chunking_strategy = "recursive"
        settings.chunk_size = 800
        settings.chunk_overlap = 150
        assert await ingest_upload(target, source_file="kỹ thuật.txt") == 1
    chunk = store.call_args.args[0][0]
    assert chunk["source_file"] == "kỹ thuật.txt"
    assert chunk["content"] == "Shrimp water quality guidance."
    assert len(chunk["embedding"]) == 768


@pytest.mark.parametrize(
    "suffix, content",
    [(".txt", b" \n "), (".txt", b"\xff\xfe"), (".pdf", b"invalid PDF"), (".docx", b"invalid ZIP")],
)
async def test_unreadable_document_never_stores_chunks(tmp_path, suffix, content):
    target = tmp_path / ("document" + suffix)
    target.write_bytes(content)
    with patch("app.ingestion.ingest.insert_new_document", new_callable=AsyncMock) as store:
        with pytest.raises(InvalidDocumentError):
            await ingest_upload(target, source_file="guide" + suffix)
    store.assert_not_awaited()


async def test_embedding_failure_never_stores_partial_document(tmp_path):
    target = tmp_path / "document.txt"
    target.write_text("content", encoding="utf-8")
    with (
        patch("app.ingestion.ingest.chunk_document", new_callable=AsyncMock) as chunk,
        patch(
            "app.ingestion.ingest.embed_document_chunk",
            new_callable=AsyncMock,
            side_effect=[[0.1], EmbeddingError()],
        ),
        patch("app.ingestion.ingest.insert_new_document", new_callable=AsyncMock) as store,
    ):
        chunk.return_value = [MagicMock(content="one"), MagicMock(content="two")]
        with pytest.raises(EmbeddingError):
            await ingest_upload(target, source_file="guide.txt")
    store.assert_not_awaited()


@pytest.mark.parametrize("exists, database_error", [(False, False), (True, False), (False, True)])
async def test_document_storage_transaction(exists, database_error):
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.fetchval = AsyncMock(return_value=exists)
    conn.executemany = AsyncMock(
        side_effect=RuntimeError("insert failed") if database_error else None
    )
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock()
    transaction.__aexit__ = AsyncMock(return_value=False)
    conn.transaction.return_value = transaction

    @asynccontextmanager
    async def connection():
        yield conn

    chunks = [
        {
            "source_file": "guide.txt",
            "chunk_index": i,
            "content": f"chunk {i}",
            "embedding": [0.1],
            "metadata": {},
        }
        for i in range(2)
    ]
    with patch("app.db.vector_store.get_connection", connection):
        if exists or database_error:
            with pytest.raises(DocumentConflictError if exists else RuntimeError):
                await insert_new_document(chunks)
            assert transaction.__aexit__.call_args.args[0] is (
                DocumentConflictError if exists else RuntimeError
            )
        else:
            assert await insert_new_document(chunks) == 2
            assert transaction.__aexit__.call_args.args == (None, None, None)
            assert len(conn.executemany.call_args.args[1]) == 2
        if exists:
            conn.executemany.assert_not_awaited()
    conn.execute.assert_awaited_once_with(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "guide.txt"
    )
