from fastapi import APIRouter

from app.api.v1.endpoints import chat, documents, health

router = APIRouter()

# Health (no auth)
router.include_router(health.router)

# Chat (requires X-Internal-Key)
router.include_router(chat.router, prefix="/chat", tags=["Chat"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
