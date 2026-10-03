from fastapi import APIRouter
from app.config import settings

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint — không yêu cầu authentication."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.app_env,
    }
