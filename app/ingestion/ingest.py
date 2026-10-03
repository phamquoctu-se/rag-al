"""
Ingestion CLI — loads documents, embeds them, and stores in pgvector.

Usage:
    python -m app.ingestion.ingest --path data/documents/
    python -m app.ingestion.ingest --path data/documents/ky_thuat_nuoi_tom.pdf
    python -m app.ingestion.ingest --path data/documents/ --clear
"""

import asyncio
import argparse
import time
from pathlib import Path

# Load .env before importing app modules
from dotenv import load_dotenv
load_dotenv()

from app.utils.logger import logger
from app.ingestion.loader import load_document, load_directory
from app.ingestion.chunker import chunk_document
from app.core.embedder import embed_document_chunk
from app.db.session import init_db_pool, close_db_pool
from app.db.vector_store import insert_chunks


async def ingest_file(file_path: Path) -> int:
    """Load, chunk, embed and store one document. Returns chunk count."""
    logger.info(f"Loading: {file_path.name}")
    doc = load_document(file_path)

    chunks = await chunk_document(doc)
    logger.info(f"  -> {len(chunks)} chunks created")

    embedded_chunks = []
    for chunk in chunks:
        embedding = await embed_document_chunk(chunk.content)
        embedded_chunks.append({
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "embedding": embedding,
            "metadata": chunk.metadata,
        })
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.05)

    count = await insert_chunks(embedded_chunks)
    logger.info(f"  -> {count} chunks stored in pgvector")
    return count


async def main(path: str, clear: bool = False) -> None:
    """Main ingestion flow."""
    await init_db_pool()
    try:
        target = Path(path)
        if not target.exists():
            logger.error(f"Path does not exist: {target}")
            return

        start = time.monotonic()
        total_chunks = 0

        if target.is_file():
            total_chunks = await ingest_file(target)
        elif target.is_dir():
            docs = load_directory(target)
            if not docs:
                logger.warning(f"No supported documents found in {target}")
            for doc in docs:
                file_path = target / doc.source_file
                count = await ingest_file(file_path)
                total_chunks += count
        else:
            logger.error(f"Invalid path: {target}")

        elapsed = time.monotonic() - start
        logger.info(f"Ingestion complete: {total_chunks} chunks in {elapsed:.1f}s")
    finally:
        await close_db_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartShrimp RAG Document Ingestion")
    parser.add_argument(
        "--path",
        required=True,
        help="Path to a document file or directory of documents",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="(not implemented yet) Clear existing chunks before ingesting",
    )
    args = parser.parse_args()

    asyncio.run(main(args.path, args.clear))
