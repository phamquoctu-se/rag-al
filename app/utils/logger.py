import sys
from loguru import logger
from app.config import settings

# Remove default handler
logger.remove()

# Configure format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Console handler with UTF-8 encoding for Windows (avoids charmap errors on Vietnamese text)
import io
stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "buffer") else sys.stdout

logger.add(
    stdout_utf8,
    format=log_format,
    level=settings.log_level,
    colorize=False,   # Disable colorize when wrapping to avoid ANSI on non-tty
    enqueue=True,
)

# File handler (production)
if settings.is_production:
    logger.add(
        "logs/rag_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="INFO",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        enqueue=True,
        encoding="utf-8",
    )

__all__ = ["logger"]