from __future__ import annotations

import asyncio

from app.config import settings
from app.utils.errors import EmbeddingError
from app.utils.logger import logger

# Embedding client (api_version=v1 required for Gemini embedding models)
_embed_client = None

# Output dimension - Matryoshka truncation for ivfflat compatibility
# gemini-embedding-2 native = 3072, but ivfflat max = 2000
# Use 768 via outputDimensionality (Matryoshka) -- same as old text-embedding-004
EMBEDDING_DIM = 768


def _get_embed_client():
    """Lazy-initialize Gemini embedding client."""
    global _embed_client
    if _embed_client is None:
        from google import genai
        _embed_client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"api_version": "v1"},
        )
    return _embed_client


def _embed_many(texts: list[str], task_type: str) -> list[list[float]]:
    """Internal sync batch embed call with Matryoshka truncation to 768 dims."""
    from google.genai import types as genai_types

    client = _get_embed_client()
    # For gemini-embedding-2, google-genai normalizes a plain list[str] as one
    # Content containing multiple Parts. Build explicit Content objects so the
    # batch endpoint returns exactly one embedding per input text.
    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=text)],
        )
        for text in texts
    ]
    result = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=contents,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return [list(item.values) for item in result.embeddings]


def _embed(text: str, task_type: str) -> list[float]:
    """Internal sync embed call for one text."""
    return _embed_many([text], task_type)[0]


async def embed_text(text: str) -> list[float]:
    """
    Generate a 768-dim embedding vector for the query text.
    Uses RETRIEVAL_QUERY task type + Matryoshka truncation.

    Returns: List of 768 floats
    Raises: EmbeddingError
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty text")

    try:
        embedding = await asyncio.to_thread(_embed, text, "RETRIEVAL_QUERY")
        logger.debug(f"Embedded query ({len(text)} chars) -> dim={len(embedding)}")
        return embedding
    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Embedding API call failed: {e}")
        raise EmbeddingError(f"Gemini embedding error: {str(e)}")


async def embed_document_chunk(text: str) -> list[float]:
    """
    Generate a 768-dim embedding vector for a document chunk.
    Uses RETRIEVAL_DOCUMENT task type + Matryoshka truncation.

    Returns: List of 768 floats
    Raises: EmbeddingError
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty chunk")

    try:
        embedding = await asyncio.to_thread(_embed, text, "RETRIEVAL_DOCUMENT")
        return embedding
    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Document embedding failed: {e}")
        raise EmbeddingError(f"Gemini document embedding error: {str(e)}")


async def embed_document_texts(
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed document texts in batches for semantic boundary detection."""
    if not texts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if any(not text or not text.strip() for text in texts):
        raise EmbeddingError("Cannot embed empty document text")

    try:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_embeddings = await asyncio.to_thread(
                _embed_many,
                batch,
                "RETRIEVAL_DOCUMENT",
            )
            if len(batch_embeddings) != len(batch):
                raise EmbeddingError(
                    "Embedding API returned an unexpected batch size "
                    f"(expected={len(batch)}, actual={len(batch_embeddings)})"
                )
            embeddings.extend(batch_embeddings)
        return embeddings
    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Document batch embedding failed: {e}")
        raise EmbeddingError(f"Gemini document embedding error: {str(e)}")
