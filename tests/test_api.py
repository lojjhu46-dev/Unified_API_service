"""API测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.llm.gateway import LLMGatewayError
from app.schemas import ToolTrace


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
    with patch("app.main.ingest_file") as mock_ingest, patch(
        "app.main.orchestrator.retriever.refresh"
    ) as mock_refresh:
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
    mock_refresh.assert_called_once()


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


def test_upload_too_large(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("big.txt", b"x" * (10 * 1024 * 1024 + 1), "text/plain")},
    )
    assert response.status_code == 400
    assert "10MB" in response.json()["detail"]


def test_upload_processing_error_is_sanitized(client):
    with patch("app.main.save_uploaded_file", return_value="D:/secret/path/test.txt"), patch(
        "app.main.ingest_file", side_effect=RuntimeError("secret path D:/secret/path/test.txt")
    ):
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"text content", "text/plain")},
        )

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "文档处理失败，请稍后重试"
    assert "request_id" in data
    assert "secret" not in data["detail"]


def test_create_session(client):
    response = client.post("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data


def test_get_session_history(client):
    with patch("app.main.orchestrator.memory") as mock_memory:
        mock_memory.create_session = AsyncMock(return_value="shared_session")
        mock_memory.get_history = AsyncMock(return_value=[
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ])

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]
        response = client.get(f"/sessions/{session_id}/history")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["messages"][0]["content"] == "问题"
    mock_memory.get_history.assert_awaited_once_with("shared_session")


def test_delete_session(client):
    with patch("app.main.orchestrator.memory") as mock_memory:
        mock_memory.create_session = AsyncMock(return_value="shared_session")
        mock_memory.clear_session = AsyncMock()

        create_resp = client.post("/sessions")
        session_id = create_resp.json()["session_id"]
        response = client.delete(f"/sessions/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    mock_memory.clear_session.assert_awaited_once_with("shared_session")


def test_ask_with_standalone_question(client):
    with patch("app.orchestrator.orchestrator.memory") as mock_memory, \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")), \
         patch("app.orchestrator.orchestrator.retriever.search", new=AsyncMock(return_value=[])):

        mock_memory.get_history = AsyncMock(return_value=[
            {"role": "user", "content": "iPhone 15有什么特点"},
            {"role": "assistant", "content": "特点..."},
        ])
        mock_memory.create_session = AsyncMock(return_value="test_session")
        mock_memory.append_turn = AsyncMock()

        with patch("app.orchestrator.rewrite_question", new=AsyncMock(return_value="iPhone 15的价格")):
            payload = {
                "user_id": "test_user",
                "question": "它的价格呢",
            }
            response = client.post("/ask", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["standalone_question"] == "iPhone 15的价格"


def test_rag_prompt_includes_history_original_and_standalone_question(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[
        {"role": "user", "content": "iPhone 15有什么特点"},
        {"role": "assistant", "content": "特点..."},
    ])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    with patch("app.orchestrator.orchestrator.memory", mock_memory), patch(
        "app.orchestrator.orchestrator.llm.generate",
        new=AsyncMock(return_value="回答"),
    ) as mock_generate, patch(
        "app.orchestrator.orchestrator.retriever.search",
        new=AsyncMock(return_value=[]),
    ), patch("app.orchestrator.rewrite_question", new=AsyncMock(return_value="iPhone 15的续航")):
        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "它的续航怎么样"},
        )

    assert response.status_code == 200
    prompt = mock_generate.call_args.args[0]
    assert "最近对话历史" in prompt
    assert "iPhone 15有什么特点" in prompt
    assert "用户原始问题" in prompt
    assert "它的续航怎么样" in prompt
    assert "独立检索问题" in prompt
    assert "iPhone 15的续航" in prompt


def test_ask_routes_to_web_for_news(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute = AsyncMock(return_value=ToolTrace(
        tool_name="web_search",
        tool_input={"query": "今天的新闻"},
        status="success",
        output_preview="新闻标题",
        latency_ms=100,
    ))
    mock_tools._run_web_search = AsyncMock(return_value={
        "success": True,
        "results": [
            {"title": "新闻标题", "url": "https://example.com", "snippet": "新闻内容"},
        ],
    })

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")):

        payload = {
            "user_id": "test_user",
            "question": "今天的新闻",
        }
        response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "web"


def test_ask_routes_to_tool_for_calculation(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute = AsyncMock(return_value=ToolTrace(
        tool_name="calculator",
        tool_input={"expression": "计算 2+3 等于多少"},
        status="success",
        output_preview="5",
        latency_ms=10,
    ))
    mock_tools._run_calculator = AsyncMock(return_value={
        "success": True,
        "result": "5",
    })

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")):

        payload = {
            "user_id": "test_user",
            "question": "计算 2+3 等于多少",
        }
        response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "tool"


def test_ask_need_web_never_skips_search(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")), \
         patch("app.orchestrator.orchestrator.retriever.search", new=AsyncMock(return_value=[])):

        payload = {
            "user_id": "test_user",
            "question": "今天天气",
            "need_web": "never",
        }
        response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "rag"
