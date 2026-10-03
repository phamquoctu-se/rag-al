from fastapi import APIRouter, Depends

from app.core.pipeline import run_rag_pipeline
from app.dependencies import verify_internal_key
from app.models.schemas import ChatRequest, ChatResponse
from app.utils.logger import logger

router = APIRouter()


@router.post(
    "",
    response_model=ChatResponse,
    summary="Gui cau hoi toi RAG pipeline",
    description=(
        "Nhan cau hoi tu KTV (qua smartshrimp_be), chay toan bo pipeline "
        "embed -> retrieve -> generate, tra ket qua de BE luu vao rag_queries."
    ),
    dependencies=[Depends(verify_internal_key)],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    UC-21: Gui cau hoi toi Chatbox
    - Chi smartshrimp_be duoc goi endpoint nay (internal API key guard)
    - KTV identity duoc xac thuc phia BE truoc khi goi sang day
    """
    logger.info(
        f"Chat request | user_id={request.user_id} "
        f"conversation_id={request.conversation_id} "
        f"question_len={len(request.question)}"
    )

    response = await run_rag_pipeline(
        question=request.question,
        top_k=request.top_k,
    )

    logger.info(
        f"Chat response | status={response.query_status} "
        f"top_sim={response.top_similarity} "
        f"time={response.processing_time_ms}ms"
    )
    return response