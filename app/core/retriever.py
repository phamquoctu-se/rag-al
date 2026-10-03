from app.config import settings
from app.db.vector_store import search_similar_chunks, DocumentChunk
from app.utils.logger import logger


async def retrieve(
    query_embedding: list[float],
    top_k: int | None = None,
) -> list[DocumentChunk]:
    """
    Search for the most similar document chunks given a query embedding.

    Args:
        query_embedding: 768-dim vector from embedder
        top_k: Max number of chunks to return (default from config)

    Returns:
        List of DocumentChunk sorted by similarity descending
    """
    k = top_k if top_k is not None else settings.top_k_default

    chunks = await search_similar_chunks(
        query_embedding=query_embedding,
        top_k=k,
    )

    logger.debug(
        f"Retrieved {len(chunks)} chunks; "
        f"top_similarity={chunks[0].similarity:.4f}" if chunks else "no chunks found"
    )
    return chunks
