from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.session import init_db_pool, close_db_pool
from app.api.v1.router import router as v1_router
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info(f"SmartShrimp RAG Service starting (env={settings.app_env})")
    await init_db_pool()
    logger.info("Ready to serve requests")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down RAG service...")
    await close_db_pool()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="SmartShrimp RAG Service",
        description=(
            "Internal FastAPI microservice cung cấp RAG pipeline "
            "cho tính năng Chatbox của hệ thống SmartShrimp."
        ),
        version=settings.app_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS (internal only — chỉ cho phép BE gọi) ───────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    # ── Global exception handler ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception on {request.method} {request.url}")
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "Internal server error"},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(v1_router, prefix="/v1")

    return app


app = create_app()
