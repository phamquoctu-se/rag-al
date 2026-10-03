from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from app.config import settings
from app.utils.logger import logger

# Global connection pool — initialized on startup
_pool: asyncpg.Pool | None = None


async def init_db_pool() -> None:
    """Initialize asyncpg connection pool. Call on app startup."""
    global _pool
    if _pool is not None:
        return
    try:
        # asyncpg expects postgresql:// not postgresql+asyncpg://
        db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        _pool = await asyncpg.create_pool(
            dsn=db_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


async def close_db_pool() -> None:
    """Close the connection pool. Call on app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")


@asynccontextmanager
async def get_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Async context manager for getting a pooled DB connection."""
    global _pool
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db_pool() first.")
    async with _pool.acquire() as conn:
        yield conn
