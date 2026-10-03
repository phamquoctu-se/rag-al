from fastapi import Header, HTTPException, status
from app.config import settings


async def verify_internal_key(x_internal_key: str = Header(...)) -> None:
    """
    Dependency guard: chỉ cho phép smartshrimp_be gọi qua internal API key.
    Raise 401 nếu key sai hoặc thiếu.
    """
    if x_internal_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Invalid internal API key"},
        )
