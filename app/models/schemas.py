from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID


# ─── Request ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body gửi từ smartshrimp_be tới RAG service."""

    conversation_id: Optional[UUID] = Field(
        default=None,
        description="UUID hội thoại đang tiếp tục; None nếu hội thoại mới"
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Câu hỏi của KTV"
    )
    user_id: UUID = Field(..., description="UUID của KTV đang hỏi")
    season_id: Optional[UUID] = Field(
        default=None,
        description="UUID vụ nuôi ngữ cảnh (tuỳ chọn, dùng để log)"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Số chunks muốn retrieve (mặc định theo config)"
    )


# ─── Response ────────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """Một chunk tài liệu được retrieve."""
    source_file: str
    chunk_index: int
    content: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    metadata: dict = {}


class ChatResponse(BaseModel):
    """Response trả về cho smartshrimp_be."""

    answer: Optional[str] = Field(
        default=None,
        description="Câu trả lời từ LLM, kể cả no_source/low_match; None khi có lỗi"
    )
    query_status: str = Field(
        ...,
        description="answered | no_source | low_match | error"
    )
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    top_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Độ tương đồng cao nhất tìm được"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Phiên bản LLM model dùng để generate"
    )
    processing_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Thời gian xử lý tổng (ms)"
    )
