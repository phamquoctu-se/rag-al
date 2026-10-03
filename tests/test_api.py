"""API integration tests for POST /v1/chat endpoint."""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from app.main import app
from app.models.schemas import ChatResponse


INTERNAL_KEY = "test-internal-key"
HEADERS = {"X-Internal-Key": INTERNAL_KEY}

BASE_URL = "http://test"

SAMPLE_REQUEST = {
    "question": "Làm thế nào để xử lý nước ao bị ô nhiễm?",
    "user_id": "00000000-0000-0000-0000-000000000001",
    "conversation_id": None,
    "season_id": None,
    "top_k": 3,
}


@pytest.fixture
def mock_pipeline_answered():
    """Mock pipeline to return answered status."""
    with patch("app.api.v1.endpoints.chat.run_rag_pipeline", new_callable=AsyncMock) as mock:
        mock.return_value = ChatResponse(
            answer="Để xử lý nước ao bị ô nhiễm, cần...",
            query_status="answered",
            retrieved_chunks=[],
            top_similarity=0.88,
            model_version="gemini-2.0-flash",
            processing_time_ms=1200,
        )
        yield mock


@pytest.fixture
def mock_settings_key():
    """Override internal API key for tests."""
    with patch("app.dependencies.settings") as mock_settings:
        mock_settings.internal_api_key = INTERNAL_KEY
        yield mock_settings


# ── Normal Cases ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_returns_answered(mock_settings_key, mock_pipeline_answered):
    """N-01: request hợp lệ với key đúng → 200 status=answered"""
    # Skip DB pool init in test
    with patch("app.db.session.init_db_pool", new_callable=AsyncMock), \
         patch("app.db.session.close_db_pool", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
            response = await client.post("/v1/chat", json=SAMPLE_REQUEST, headers=HEADERS)

    assert response.status_code == 200
    data = response.json()
    assert data["query_status"] == "answered"
    assert data["answer"] is not None
    assert data["top_similarity"] == 0.88


@pytest.mark.asyncio
async def test_health_check():
    """N-02: GET /v1/health trả về 200 và status=ok"""
    with patch("app.db.session.init_db_pool", new_callable=AsyncMock), \
         patch("app.db.session.close_db_pool", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
            response = await client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Abnormal Cases ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_unauthorized_when_key_missing():
    """A-01: thiếu X-Internal-Key header → 422 (header required)"""
    with patch("app.db.session.init_db_pool", new_callable=AsyncMock), \
         patch("app.db.session.close_db_pool", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
            response = await client.post("/v1/chat", json=SAMPLE_REQUEST)

    assert response.status_code == 422  # FastAPI validation: missing required header


@pytest.mark.asyncio
async def test_chat_unauthorized_when_key_wrong(mock_settings_key):
    """A-02: X-Internal-Key sai → 401"""
    with patch("app.db.session.init_db_pool", new_callable=AsyncMock), \
         patch("app.db.session.close_db_pool", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
            response = await client.post(
                "/v1/chat",
                json=SAMPLE_REQUEST,
                headers={"X-Internal-Key": "wrong-key"},
            )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_validation_error_empty_question(mock_settings_key):
    """A-03: question rỗng → 422 validation error"""
    bad_request = {**SAMPLE_REQUEST, "question": ""}
    with patch("app.db.session.init_db_pool", new_callable=AsyncMock), \
         patch("app.db.session.close_db_pool", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
            response = await client.post("/v1/chat", json=bad_request, headers=HEADERS)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_validation_error_top_k_too_large(mock_settings_key):
    """B-01: top_k > 20 → 422 validation error"""
    bad_request = {**SAMPLE_REQUEST, "top_k": 21}
    with patch("app.db.session.init_db_pool", new_callable=AsyncMock), \
         patch("app.db.session.close_db_pool", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
            response = await client.post("/v1/chat", json=bad_request, headers=HEADERS)

    assert response.status_code == 422
