"""API测试"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.llm.gateway import LLMGatewayError


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert data["docs"] == "/docs"


def test_ask_direct(client):
    payload = {
        "user_id": "test_user",
        "question": "你好",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "session_id" in data
    assert data["route"] == "direct"
    assert "answer" in data
    assert "sources" in data
    assert "tool_trace" in data
    assert "timing" in data


def test_ask_mixed_greeting_routes_to_rag(client):
    payload = {
        "user_id": "test_user",
        "question": "你好，什么是机器学习？",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "rag"


def test_ask_rag(client):
    payload = {
        "user_id": "test_user",
        "question": "什么是机器学习",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "session_id" in data
    assert data["route"] == "rag"
    assert "answer" in data
    assert isinstance(data["sources"], list)
    assert "timing" in data


def test_ask_with_session(client):
    payload = {
        "user_id": "test_user",
        "session_id": "custom_session_123",
        "question": "测试问题",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "custom_session_123"


def test_ask_validation_error(client):
    payload = {
        "user_id": "test_user",
        "question": "",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 422


def test_ask_blank_question(client):
    payload = {
        "user_id": "test_user",
        "question": "   ",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 422


def test_ask_missing_user_id(client):
    payload = {
        "question": "测试问题",
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 422


def test_ask_llm_gateway_error_returns_clear_response(client):
    payload = {
        "user_id": "test_user",
        "question": "你好",
    }
    with patch(
        "app.orchestrator.orchestrator.llm.generate",
        new=AsyncMock(side_effect=LLMGatewayError("未配置DEEPSEEK_API_KEY")),
    ):
        response = client.post("/ask", json=payload)

    assert response.status_code == 503
    data = response.json()
    assert data["detail"] == "未配置DEEPSEEK_API_KEY"
    assert "request_id" in data


def test_orchestrator_uses_direct_prompt_template(client):
    payload = {
        "user_id": "test_user",
        "question": "你好",
    }
    with patch(
        "app.orchestrator.orchestrator.llm.generate",
        new=AsyncMock(return_value="ok"),
    ) as mock_generate:
        response = client.post("/ask", json=payload)

    assert response.status_code == 200
    _prompt = mock_generate.call_args.args[0]
    assert "请直接回答以下问题" in _prompt
    assert mock_generate.call_args.kwargs["system_prompt"] == "你是一个有用的中文助手。"


def test_orchestrator_uses_rag_prompt_template(client):
    payload = {
        "user_id": "test_user",
        "question": "什么是机器学习",
    }
    with patch(
        "app.orchestrator.orchestrator.llm.generate",
        new=AsyncMock(return_value="ok"),
    ) as mock_generate:
        response = client.post("/ask", json=payload)

    assert response.status_code == 200
    prompt = mock_generate.call_args.args[0]
    system_prompt = mock_generate.call_args.kwargs["system_prompt"]
    assert "基于以下文档回答问题" in prompt
    assert "你是一个严谨的中文问答助手" in system_prompt


def test_upload_pdf(client):
    with patch("app.main.ingest_file") as mock_ingest:
        mock_ingest.return_value = {
            "document_id": "test123",
            "filename": "test.pdf",
            "chunks": 5,
        }
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", b"pdf content", "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks"] == 5


def test_upload_txt(client):
    with patch("app.main.ingest_file") as mock_ingest:
        mock_ingest.return_value = {
            "document_id": "test456",
            "filename": "test.txt",
            "chunks": 3,
        }
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"text content", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_upload_invalid_extension(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.doc", b"doc content", "application/msword")},
    )
    assert response.status_code == 400
    assert "仅支持" in response.json()["detail"]
