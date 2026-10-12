"""Opt-in PostgreSQL integration. Uses a unique UUID and rolls back all fixture data."""
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest
from dotenv import load_dotenv

from app.db import vector_store
from app.ingestion import managed
from app.utils.errors import DocumentConflictError

pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(os.getenv("RAG_DB_INTEGRATION") != "1", reason="requires opt-in dev DB")]


async def test_real_pgvector_index_retry_filter_and_delete(monkeypatch, tmp_path):
    load_dotenv()
    conn = await asyncpg.connect(os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://"))
    transaction = conn.transaction()
    await transaction.start()

    @asynccontextmanager
    async def get_connection():
        yield conn

    monkeypatch.setattr(managed, "get_connection", get_connection)
    monkeypatch.setattr(vector_store, "get_connection", get_connection)
    doc_id = str(uuid4())
    embedding = [1.0] + [0.0] * 767
    prepare = AsyncMock(return_value=[{"chunk_index": 0, "content": "Integration test shrimp source", "embedding": embedding, "metadata": {"page": 1}}])
    monkeypatch.setattr(managed, "prepare_document_chunks", prepare)
    path = tmp_path / "test.txt"
    path.write_text("Integration fixture", encoding="utf-8")
    try:
        assert await managed.index_document(path, doc_id, 1, "fixture.txt", "Fixture") == 1
        assert await managed.index_document(path, doc_id, 1, "fixture.txt", "Fixture") == 1
        assert prepare.await_count == 1
        assert await conn.fetchval("SELECT COUNT(*) FROM document_chunks WHERE metadata->>'document_id'=$1", doc_id) == 1
        chunks = await vector_store.search_similar_chunks(embedding, eligible_documents=[{"document_id": doc_id, "attempt": 1}])
        assert len(chunks) == 1 and chunks[0].source_file == "fixture.txt"
        assert chunks[0].metadata["document_id"] == doc_id
        assert await vector_store.search_similar_chunks(embedding, eligible_documents=[]) == []
        assert await vector_store.search_similar_chunks(embedding, eligible_documents=[{"document_id": doc_id, "attempt": 2}]) == []
        assert await managed.index_document(path, doc_id, 2, "fixture.txt", "Fixture") == 1
        assert await conn.fetchval("SELECT COUNT(*) FROM document_chunks WHERE metadata->>'document_id'=$1", doc_id) == 1
        await managed.delete_vectors(doc_id)
        await managed.delete_vectors(doc_id)
        assert await conn.fetchval("SELECT COUNT(*) FROM document_chunks WHERE metadata->>'document_id'=$1", doc_id) == 0
        with pytest.raises(DocumentConflictError):
            await managed.index_document(path, doc_id, 3, "fixture.txt", "Fixture")
    finally:
        await transaction.rollback()
        await conn.close()
