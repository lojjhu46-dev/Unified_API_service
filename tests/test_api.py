"""API测试"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


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
