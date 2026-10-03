-- ============================================================================
-- SmartShrimp RAG -- pgvector Migration
-- Enable pgvector extension and create document_chunks table
-- Run once against the Supabase PostgreSQL database
-- ============================================================================

-- Enable pgvector (requires Supabase to have the extension available)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- document_chunks: Stores embedded text chunks from knowledge base documents
-- ============================================================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file     VARCHAR(500) NOT NULL,       -- e.g. "ky_thuat_nuoi_tom.pdf"
    chunk_index     INT         NOT NULL,        -- position within the source file
    content         TEXT        NOT NULL,        -- raw text of the chunk
    embedding       vector(768) NOT NULL,        -- gemini-embedding-2 with outputDimensionality=768 (Matryoshka) = 768 dims
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- page, section, etc.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_chunk_per_file UNIQUE (source_file, chunk_index),
    CONSTRAINT chk_content_nonempty CHECK (nullif(trim(content), '') IS NOT NULL)
);

-- IVFFlat index for approximate cosine similarity search
-- lists = 100 is suitable for up to ~1M rows; adjust as collection grows
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for filtering/lookup by source file
CREATE INDEX IF NOT EXISTS idx_chunks_source_file
    ON document_chunks (source_file);
