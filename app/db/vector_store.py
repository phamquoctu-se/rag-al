import json
from dataclasses import dataclass
from typing import Optional

from app.db.session import get_connection
from app.utils.logger import logger


@dataclass
class DocumentChunk:
    id: str
    source_file: str
    chunk_index: int
    content: str
    similarity: float
    metadata: dict


async def search_similar_chunks(
    query_embedding: list[float],
    top_k: int = 5,
    source_file_filter: Optional[str] = None,
) -> list[DocumentChunk]:
    """
    Query pgvector for the top-K most similar chunks to query_embedding.
    Uses cosine similarity (1 - cosine_distance).

    Args:
        query_embedding: Float vector from embedding model (768 dims)
        top_k: Number of results to return
        source_file_filter: Optional -- restrict search to a specific file

    Returns:
        List of DocumentChunk sorted by similarity descending
    """
    # Convert list to pgvector format string
    embedding_str = f"[{','.join(str(v) for v in query_embedding)}]"

    if source_file_filter:
        sql = """
            SELECT
                id::text,
                source_file,
                chunk_index,
                content,
                metadata,
                1 - (embedding <=> $1::vector) AS similarity
            FROM document_chunks
            WHERE source_file = $3
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        params = [embedding_str, top_k, source_file_filter]
    else:
        sql = """
            SELECT
                id::text,
                source_file,
                chunk_index,
                content,
                metadata,
                1 - (embedding <=> $1::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        params = [embedding_str, top_k]

    try:
        async with get_connection() as conn:
            rows = await conn.fetch(sql, *params)
        return [
            DocumentChunk(
                id=row["id"],
                source_file=row["source_file"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                similarity=float(row["similarity"]),
                metadata=(
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else (dict(row["metadata"]) if row["metadata"] else {})
                ),
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise


async def insert_chunks(chunks: list[dict]) -> int:
    """
    Bulk-insert document chunks into pgvector table.
    Each dict must have: source_file, chunk_index, content, embedding, metadata.
    Uses INSERT ... ON CONFLICT DO UPDATE to support re-ingestion.

    Returns: number of rows inserted/updated
    """
    if not chunks:
        return 0

    sql = """
        INSERT INTO document_chunks (source_file, chunk_index, content, embedding, metadata)
        VALUES ($1, $2, $3, $4::vector, $5::jsonb)
        ON CONFLICT (source_file, chunk_index)
        DO UPDATE SET
            content   = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata  = EXCLUDED.metadata,
            created_at = NOW()
    """

    import json

    async with get_connection() as conn:
        await conn.executemany(
            sql,
            [
                (
                    c["source_file"],
                    c["chunk_index"],
                    c["content"],
                    f"[{','.join(str(v) for v in c['embedding'])}]",
                    json.dumps(c.get("metadata", {})),
                )
                for c in chunks
            ],
        )

    logger.info(f"Inserted/updated {len(chunks)} chunks")
    return len(chunks)
