import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.dependencies import verify_internal_key
from app.ingestion.ingest import ingest_upload
from app.ingestion.managed import delete_vectors, index_document
from app.models.schemas import DocumentResponse
from app.utils.errors import DocumentConflictError, InvalidDocumentError
from app.utils.logger import logger

router = APIRouter()
MAX_FILE_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


@router.post(
    "",
    status_code=201,
    response_model=DocumentResponse,
    dependencies=[Depends(verify_internal_key)],
    summary="Index one new knowledge document",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        }
    },
)
async def create_document(
    request: Request, filename: str = Query(..., min_length=1, max_length=500)
):
    # Reject path components on both Windows and Unix.
    if not filename.strip() or any(
        char in "/\\" or ord(char) < 32 or ord(char) == 127 for char in filename
    ):
        raise HTTPException(400, detail="Invalid document filename")
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, detail="Unsupported document format")
    if (
        request.headers.get("content-type", "").split(";")[0].strip().lower()
        != "application/octet-stream"
    ):
        raise HTTPException(415, detail="Expected application/octet-stream")

    try:
        with TemporaryDirectory(prefix="rag-upload-") as directory:
            # Keep a short filesystem name; the logical source can be 500 characters.
            target = Path(directory) / ("document" + Path(filename).suffix.lower())
            size = 0
            with target.open("wb") as output:
                async for part in request.stream():
                    size += len(part)
                    if size > MAX_FILE_BYTES:
                        raise HTTPException(413, detail="Document exceeds 10 MB")
                    output.write(part)
            if not size:
                raise HTTPException(400, detail="Empty document")
            count = await ingest_upload(target, source_file=filename)
        return DocumentResponse(source_file=filename, chunks_count=count)
    except HTTPException:
        raise
    except InvalidDocumentError as exc:
        raise HTTPException(400, detail="Cannot read document or no readable text") from exc
    except DocumentConflictError as exc:
        raise HTTPException(409, detail="Document already exists") from exc
    except Exception as exc:
        logger.error("Document ingestion failed ({})", type(exc).__name__)
        raise HTTPException(503, detail="Document ingestion unavailable") from exc


@router.post("/{document_id}/ingest", dependencies=[Depends(verify_internal_key)])
async def ingest_managed(
    document_id: UUID, request: Request,
    filename: str = Query(..., min_length=1, max_length=500),
    title: str = Query(..., min_length=1, max_length=255),
    attempt: int = Query(..., ge=1),
):
    if any(char in "/\\" or ord(char) < 32 or ord(char) == 127 for char in filename):
        raise HTTPException(400, detail="Invalid filename")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, detail="Unsupported document format")
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/octet-stream":
        raise HTTPException(415, detail="Expected binary body")
    try:
        with TemporaryDirectory(prefix="rag-managed-") as directory:
            path = Path(directory) / ("document" + extension)
            size = 0
            with path.open("wb") as output:
                async for part in request.stream():
                    size += len(part)
                    if size > MAX_FILE_BYTES:
                        raise HTTPException(413, detail="Document exceeds 10 MiB")
                    output.write(part)
            if not size:
                raise HTTPException(400, detail="Empty document")
            count = await asyncio.wait_for(index_document(path, str(document_id), attempt, filename, title), timeout=270)
        return {"document_id": str(document_id), "attempt": attempt, "chunks_count": count}
    except HTTPException:
        raise
    except InvalidDocumentError as exc:
        raise HTTPException(400, detail="Cannot read document or no readable text") from exc
    except DocumentConflictError as exc:
        raise HTTPException(409, detail="Stale attempt or deleted document") from exc
    except Exception as exc:
        logger.error("Managed indexing failed ({})", type(exc).__name__)
        raise HTTPException(503, detail="Document indexing unavailable") from exc


@router.delete("/{document_id}/vectors", dependencies=[Depends(verify_internal_key)])
async def remove_managed_vectors(document_id: UUID):
    await delete_vectors(str(document_id))
    return {"deleted": True}
