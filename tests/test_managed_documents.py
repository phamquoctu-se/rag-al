from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.db import vector_store
from app.ingestion import managed
from app.main import app
from app.utils.errors import DocumentConflictError

DOC_ID = "cdbb18a3-17b6-440c-8e89-a3f35c3a144b"


@pytest.fixture
def connection(monkeypatch):
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.executemany = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock()
    transaction.__aexit__ = AsyncMock(return_value=False)
    conn.transaction.return_value = transaction

    @asynccontextmanager
    async def get_connection():
        yield conn

    monkeypatch.setattr(managed, "get_connection", get_connection)
    monkeypatch.setattr(vector_store, "get_connection", get_connection)
    return conn


@pytest.mark.asyncio
async def test_repeated_attempt_skips_embedding(connection, monkeypatch, tmp_path):
    connection.fetchrow.return_value = {"deleted": False, "attempt": 2, "chunk_count": 5}
    prepare = AsyncMock()
    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    assert await managed.index_document(tmp_path / "unused.txt", DOC_ID, 2, "guide.txt", "Guide") == 5
    prepare.assert_not_awaited()
    connection.executemany.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [
    {"deleted": True, "attempt": 1, "chunk_count": 0},
    {"deleted": False, "attempt": 3, "chunk_count": 5},
])
async def test_deleted_and_old_requests_cannot_resurrect(connection, tmp_path, state):
    connection.fetchrow.return_value = state
    with pytest.raises(DocumentConflictError):
        await managed.index_document(tmp_path / "unused.txt", DOC_ID, 2, "guide.txt", "Guide")
    connection.executemany.assert_not_awaited()


@pytest.mark.asyncio
async def test_index_replaces_atomically_and_keeps_source_metadata(connection, monkeypatch, tmp_path):
    path = tmp_path / "guide.txt"
    path.write_text("Shrimp guide", encoding="utf-8")
    prepare = AsyncMock(return_value=[{"chunk_index": 0, "content": "Shrimp guide", "embedding": [0.1], "metadata": {"page": 1}}])
    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    assert await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide") == 1
    metadata = connection.executemany.call_args.args[1][0][4]
    assert '"document_id": "' + DOC_ID + '"' in metadata
    assert '"attempt": 2' in metadata and '"original_filename": "guide.txt"' in metadata
    assert '"page": 1' in metadata
    sqls = [call.args[0] for call in connection.execute.call_args_list]
    assert "pg_advisory_xact_lock" in sqls[0]
    assert "DELETE FROM document_chunks" in sqls[1]
    connection.transaction.return_value.__aexit__.assert_awaited_with(None, None, None)


@pytest.mark.asyncio
async def test_embedding_error_does_not_delete_previous_index(connection, monkeypatch, tmp_path):
    path = tmp_path / "guide.txt"
    path.write_text("Shrimp guide", encoding="utf-8")
    monkeypatch.setattr(managed, "prepare_document_chunks", AsyncMock(side_effect=RuntimeError("embedding failed")))
    with pytest.raises(RuntimeError):
        await managed.index_document(path, DOC_ID, 2, "guide.txt", "Guide")
    assert len(connection.execute.call_args_list) == 1  # only the lock
    connection.executemany.assert_not_awaited()
    assert connection.transaction.return_value.__aexit__.call_args.args[0] is RuntimeError


@pytest.mark.asyncio
async def test_delete_persists_fence_before_removing_chunks(connection):
    await managed.delete_vectors(DOC_ID)
    calls = connection.execute.call_args_list
    assert "pg_advisory_xact_lock" in calls[0].args[0]
    assert "deleted=TRUE" in calls[1].args[0]
    assert calls[2].args == ("DELETE FROM document_chunks WHERE metadata->>'document_id' = $1", DOC_ID)


@pytest.mark.asyncio
async def test_retrieval_enforces_id_and_current_attempt_in_sql(connection):
    docs = [{"document_id": DOC_ID, "attempt": 2}]
    await vector_store.search_similar_chunks([0.1], eligible_documents=docs)
    sql, *params = connection.fetch.call_args.args
    assert "metadata->>'document_id' = allowed->>'document_id'" in sql
    assert "metadata->>'attempt' = allowed->>'attempt'" in sql
    assert '"attempt": 2' in params[-1]
    await vector_store.search_similar_chunks([0.1], eligible_documents=[])
    assert connection.fetch.call_args.args[-1] == "[]"
    await vector_store.search_similar_chunks([0.1])
    assert "NOT (metadata ? 'document_id')" in connection.fetch.call_args.args[0]


@pytest.mark.asyncio
async def test_internal_upload_auth_contract_and_cleanup(monkeypatch):
    from app.api.v1.endpoints import documents
    paths = []

    async def index(path, doc_id, attempt, filename, title):
        paths.append(path)
        assert path.read_text() == "Shrimp guide"
        assert (doc_id, attempt, filename, title) == (DOC_ID, 2, "guide.txt", "Guide")
        return 3

    monkeypatch.setattr(documents, "index_document", index)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        url = f"/v1/documents/{DOC_ID}/ingest?filename=guide.txt&title=Guide&attempt=2"
        assert (await client.post(url, content=b"Shrimp guide")).status_code == 422
        assert (await client.post(url, content=b"Shrimp guide", headers={"X-Internal-Key": "invalid"})).status_code == 401
        response = await client.post(url, content=b"Shrimp guide", headers={"X-Internal-Key": settings.internal_api_key, "Content-Type": "application/octet-stream"})
        assert response.status_code == 200
        assert response.json() == {"document_id": DOC_ID, "attempt": 2, "chunks_count": 3}
    assert paths and not paths[0].exists()
