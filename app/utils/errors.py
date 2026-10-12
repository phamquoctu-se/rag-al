from fastapi import HTTPException, status


class RAGServiceError(Exception):
    """Base exception for RAG service."""

    def __init__(self, message: str, code: str = "RAG_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class EmbeddingError(RAGServiceError):
    """Failed to generate embedding."""

    def __init__(self, message: str = "Failed to generate text embedding"):
        super().__init__(message, code="EMBEDDING_ERROR")


class InvalidDocumentError(RAGServiceError):
    """The document cannot be parsed or contains no readable text."""


class DocumentConflictError(RAGServiceError):
    """A document with this source filename already exists."""


class LLMError(RAGServiceError):
    """Failed to call LLM."""

    def __init__(self, message: str = "Failed to generate answer from LLM"):
        super().__init__(message, code="LLM_ERROR")


class DatabaseError(RAGServiceError):
    """Database operation failed."""

    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message, code="DATABASE_ERROR")


class UnauthorizedError(HTTPException):
    """Invalid or missing internal API key."""

    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Invalid or missing API key"},
        )
