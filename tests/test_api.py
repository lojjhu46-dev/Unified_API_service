"""API测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.llm.gateway import LLMGatewayError
from app.schemas import SourceItem, ToolTrace
from app.tools.registry import ToolExecution


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
    assert mock_ingest.call_args.kwargs["knowledge_base_type"] == "enterprise"
    assert mock_ingest.call_args.kwargs["channel"] == "api"
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


def test_upload_docx(client):
    with patch("app.main.ingest_file") as mock_ingest, patch(
        "app.main.orchestrator.retriever.refresh"
    ) as mock_refresh:
        mock_ingest.return_value = {
            "document_id": "test789",
            "filename": "test.docx",
            "chunks": 4,
        }
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test.docx",
                    b"docx content",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test.docx"
    assert data["chunks"] == 4
    mock_refresh.assert_called_once()


def test_upload_xlsx_defaults_to_enterprise(client):
    with patch("app.main.ingest_file") as mock_ingest, patch(
        "app.main.orchestrator.retriever.refresh"
    ) as mock_refresh:
        mock_ingest.return_value = {
            "document_id": "xlsx123",
            "filename": "test.xlsx",
            "chunks": 6,
        }
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test.xlsx",
                    b"xlsx content",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.xlsx"
    assert data["chunks"] == 6
    assert mock_ingest.call_args.kwargs["knowledge_base_type"] == "enterprise"
    assert mock_ingest.call_args.kwargs["owner_open_id"] is None
    assert mock_ingest.call_args.kwargs["channel"] == "api"
    mock_refresh.assert_called_once()


def test_upload_xlsx_to_personal_requires_owner(client):
    response = client.post(
        "/documents/upload",
        data={"knowledge_base_type": "personal"},
        files={
            "file": (
                "test.xlsx",
                b"xlsx content",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 400
    assert "owner_open_id" in response.json()["detail"]


def test_upload_xlsx_to_personal(client):
    with patch("app.main.ingest_file") as mock_ingest:
        mock_ingest.return_value = {
            "document_id": "xlsx456",
            "filename": "test.xlsx",
            "chunks": 2,
        }
        response = client.post(
            "/documents/upload",
            data={"knowledge_base_type": "personal", "owner_open_id": "ou_test123"},
            files={
                "file": (
                    "test.xlsx",
                    b"xlsx content",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

    assert response.status_code == 200
    assert mock_ingest.call_args.kwargs["knowledge_base_type"] == "personal"
    assert mock_ingest.call_args.kwargs["owner_open_id"] == "ou_test123"
    assert mock_ingest.call_args.kwargs["channel"] == "api"


def test_upload_invalid_extension(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.doc", b"doc content", "application/msword")},
    )
    assert response.status_code == 400
    assert "仅支持" in response.json()["detail"]
    assert "DOCX" in response.json()["detail"]
    assert "XLSX" in response.json()["detail"]


def test_upload_docm_is_rejected(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.docm", b"docm content", "application/vnd.ms-word.document.macroEnabled.12")},
    )
    assert response.status_code == 400


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


def test_rag_prompt_uses_full_internal_content_not_display_snippet(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    source = SourceItem(
        title="4.pdf",
        snippet="5. 儒家大同思想的历史影响",
        content="5. 儒家大同思想的历史影响\n先小康后大同。具有阶级特征。需要高度的生产力。",
        score=1.0,
    )

    with patch("app.orchestrator.orchestrator.memory", mock_memory), patch(
        "app.orchestrator.orchestrator.retriever.search",
        new=AsyncMock(return_value=[source]),
    ), patch(
        "app.orchestrator.orchestrator.llm.generate",
        new=AsyncMock(return_value="回答"),
    ) as mock_generate:
        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "儒家大同思想的历史影响"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "content" not in data["sources"][0]
    prompt = mock_generate.call_args.args[0]
    system_prompt = mock_generate.call_args.kwargs["system_prompt"]
    assert "先小康后大同" in prompt
    assert "阶级特征" in prompt
    assert "高度的生产力" in prompt
    assert "全局汇总任务" in system_prompt


def test_pdf_follow_up_prompt_can_include_main_content_evidence(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[
        {"role": "user", "content": "什么是儒家大同思想"},
        {"role": "assistant", "content": "儒家大同思想是理想社会思想。"},
    ])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    source = SourceItem(
        title="4.pdf",
        snippet="儒家大同思想的基本内容主要包括以下五个方面。",
        content="儒家大同思想的基本内容主要包括以下五个方面：社会制度：全民公有；管理制度：选贤与能；人际关系：讲信修睦；社会保障：人人得其所；劳动态度：各尽其力。",
        score=1.0,
    )

    with patch("app.orchestrator.orchestrator.memory", mock_memory), patch(
        "app.orchestrator.rewrite_question",
        new=AsyncMock(return_value="儒家大同思想的主要内容"),
    ), patch(
        "app.orchestrator.orchestrator.retriever.search",
        new=AsyncMock(return_value=[source]),
    ) as mock_search, patch(
        "app.orchestrator.orchestrator.llm.generate",
        new=AsyncMock(return_value="回答"),
    ) as mock_generate:
        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "简介他的主要内容"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["standalone_question"] == "儒家大同思想的主要内容"
    mock_search.assert_awaited_once()
    assert mock_search.await_args.args[0] == "儒家大同思想的主要内容"
    prompt = mock_generate.call_args.args[0]
    assert "全民公有" in prompt
    assert "选贤与能" in prompt
    assert "讲信修睦" in prompt
    assert "人人得其所" in prompt
    assert "各尽其力" in prompt


def test_pdf_follow_up_prompt_can_include_historical_evolution(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[
        {"role": "user", "content": "介绍儒家大同思想"},
        {"role": "assistant", "content": "儒家大同思想源远流长。"},
    ])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    source = SourceItem(
        title="4.pdf",
        snippet="儒家大同思想的历史演变分别经历了七个发展时期。",
        content="儒家大同思想的历史演变分别经历了七个发展时期，分别是先秦孔子、汉代董仲舒与何休、宋代二程、太平天国时期洪秀全、康有为和孙中山、当代熊十力。",
        score=1.0,
    )

    with patch("app.orchestrator.orchestrator.memory", mock_memory), patch(
        "app.orchestrator.rewrite_question",
        new=AsyncMock(return_value="儒家大同思想的时代演变"),
    ), patch(
        "app.orchestrator.orchestrator.retriever.search",
        new=AsyncMock(return_value=[source]),
    ), patch(
        "app.orchestrator.orchestrator.llm.generate",
        new=AsyncMock(return_value="回答"),
    ) as mock_generate:
        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "那时代演变呢"},
        )

    assert response.status_code == 200
    prompt = mock_generate.call_args.args[0]
    assert "先秦孔子" in prompt
    assert "董仲舒与何休" in prompt
    assert "宋代二程" in prompt
    assert "洪秀全" in prompt
    assert "康有为和孙中山" in prompt
    assert "熊十力" in prompt


def test_pdf_follow_up_prompt_can_include_background_relation(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[
        {"role": "user", "content": "儒家大同思想产生的时代背景是什么"},
        {"role": "assistant", "content": "它产生于社会制度剧烈变动的时期。"},
    ])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    source = SourceItem(
        title="4.pdf",
        snippet="儒家大同社会产生的时代背景",
        content="儒家大同社会产生的时代背景，和当时的经济、政治、文化关系密不可分。",
        score=1.0,
    )

    with patch("app.orchestrator.orchestrator.memory", mock_memory), patch(
        "app.orchestrator.rewrite_question",
        new=AsyncMock(return_value="儒家大同思想产生的时代背景和什么密不可分"),
    ), patch(
        "app.orchestrator.orchestrator.retriever.search",
        new=AsyncMock(return_value=[source]),
    ), patch(
        "app.orchestrator.orchestrator.llm.generate",
        new=AsyncMock(return_value="回答"),
    ) as mock_generate:
        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "那它和什么密不可分"},
        )

    assert response.status_code == 200
    prompt = mock_generate.call_args.args[0]
    assert "经济、政治、文化关系密不可分" in prompt


def test_pdf_question_prompt_can_include_utopia_evidence(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    source = SourceItem(
        title="4.pdf",
        snippet="和西方国家的“乌托邦”大同小异。",
        content="“大同”思想源远流长，最早源自孔子的《礼记·礼运》。它和西方国家的“乌托邦”大同小异。",
        score=1.0,
    )

    with patch("app.orchestrator.orchestrator.memory", mock_memory), patch(
        "app.orchestrator.orchestrator.retriever.search",
        new=AsyncMock(return_value=[source]),
    ), patch(
        "app.orchestrator.orchestrator.llm.generate",
        new=AsyncMock(return_value="回答"),
    ) as mock_generate:
        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "“大同”思想和西方国家什么思想大同小异"},
        )

    assert response.status_code == 200
    prompt = mock_generate.call_args.args[0]
    assert "乌托邦" in prompt


def test_ask_routes_to_agentic_rag_for_news(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
            tool_name="web_search",
            tool_input={"query": "今天的新闻"},
            status="success",
            output_preview="新闻标题",
            latency_ms=100,
        ),
        result={
            "success": True,
            "results": [
                {"title": "新闻标题", "url": "https://example.com", "snippet": "新闻内容"},
            ],
        },
    ))

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.retriever.search", new=AsyncMock(return_value=[])), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")):

        payload = {
            "user_id": "test_user",
            "question": "今天的新闻",
        }
        response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "agentic_rag"
    mock_tools.execute_with_result.assert_awaited_once()


def test_ask_need_web_always_routes_to_web_before_calculation_words(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
            tool_name="web_search",
            tool_input={"query": "今天北京气温多少"},
            status="success",
            output_preview="天气",
            latency_ms=100,
        ),
        result={
            "success": True,
            "results": [
                {"title": "天气", "url": "https://example.com", "snippet": "北京天气"},
            ],
        },
    ))

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")):

        response = client.post(
            "/ask",
            json={
                "user_id": "test_user",
                "question": "今天北京气温多少",
                "need_web": "always",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "web"


def test_ask_web_route_calls_search_once(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
            tool_name="web_search",
            tool_input={"query": "最新AI新闻"},
            status="success",
            output_preview="AI新闻",
            latency_ms=100,
        ),
        result={
            "success": True,
            "results": [
                {"title": "AI新闻", "url": "https://example.com", "snippet": "新闻内容"},
            ],
        },
    ))
    mock_tools._run_web_search = AsyncMock()

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")):

        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "最新AI新闻", "need_web": "always"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "web"
    mock_tools.execute_with_result.assert_awaited_once()
    mock_tools._run_web_search.assert_not_called()


def test_ask_agentic_rag_calls_search_once_and_includes_sources(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
            tool_name="web_search",
            tool_input={"query": "今天AI新闻"},
            status="success",
            output_preview="AI新闻",
            latency_ms=100,
        ),
        result={
            "success": True,
            "results": [
                {"title": "AI新闻", "url": "https://example.com", "snippet": "联网内容"},
            ],
        },
    ))
    mock_tools._run_web_search = AsyncMock()
    local_source = SourceItem(
        title="本地文档",
        source_type="knowledge_base",
        snippet="本地内容",
        score=0.9,
    )

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.retriever.search", new=AsyncMock(return_value=[local_source])), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")) as mock_generate:

        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "今天AI新闻"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "agentic_rag"
    assert {s["source_type"] for s in data["sources"]} == {"knowledge_base", "web_search"}
    prompt = mock_generate.call_args.args[0]
    assert "本地知识库片段" in prompt
    assert "联网搜索片段" in prompt
    mock_tools.execute_with_result.assert_awaited_once()
    mock_tools._run_web_search.assert_not_called()


def test_ask_web_failure_falls_back_to_local_rag(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
            tool_name="web_search",
            tool_input={"query": "最新产品价格"},
            status="error",
            output_preview="",
            latency_ms=100,
            error_message="搜索请求异常: ConnectError",
        ),
        result={"success": False, "error": "搜索请求异常: ConnectError", "results": []},
    ))
    local_source = SourceItem(
        title="本地价格文档",
        source_type="knowledge_base",
        snippet="本地价格信息",
        score=0.8,
    )

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.retriever.search", new=AsyncMock(return_value=[local_source])), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="基于本地知识库回答")) as mock_generate:

        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "最新产品价格", "need_web": "always"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "web"
    assert data["answer"] == "基于本地知识库回答"
    assert data["sources"][0]["source_type"] == "knowledge_base"
    assert data["tool_trace"][0]["status"] == "error"
    assert "联网搜索失败" in mock_generate.call_args.args[0]


def test_ask_web_failure_without_local_rag_returns_friendly_answer(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
            tool_name="web_search",
            tool_input={"query": "最新新闻"},
            status="error",
            output_preview="",
            latency_ms=100,
            error_message="搜索服务认证失败（HTTP 401），SERPER_API_KEY 无效",
        ),
        result={"success": False, "error": "搜索服务认证失败（HTTP 401），SERPER_API_KEY 无效", "results": []},
    ))

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.retriever.search", new=AsyncMock(return_value=[])), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="不应调用")) as mock_generate:

        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "最新新闻", "need_web": "always"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "web"
    assert "暂时无法获取联网搜索结果" in data["answer"]
    assert "SERPER_API_KEY" not in data["answer"]
    assert data["tool_trace"][0]["status"] == "error"
    mock_generate.assert_not_called()


def test_ask_auto_weather_question_does_not_route_to_tool(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
            tool_name="web_search",
            tool_input={"query": "今天北京气温多少"},
            status="success",
            output_preview="天气",
            latency_ms=100,
        ),
        result={
            "success": True,
            "results": [
                {"title": "天气", "url": "https://example.com", "snippet": "北京天气"},
            ],
        },
    ))

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.retriever.search", new=AsyncMock(return_value=[])), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")):

        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "今天北京气温多少"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "agentic_rag"


def test_ask_auto_weather_question_calls_web_search(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
            tool_name="web_search",
            tool_input={"query": "今天韶关天气怎么样"},
            status="success",
            output_preview="韶关天气",
            latency_ms=100,
        ),
        result={
            "success": True,
            "results": [
                {"title": "韶关天气", "url": "https://example.com", "snippet": "韶关天气信息"},
            ],
        },
    ))

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.retriever.search", new=AsyncMock(return_value=[])), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")):

        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "今天韶关天气怎么样"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "agentic_rag"
    mock_tools.execute_with_result.assert_awaited_once_with("web_search", {"query": "今天韶关天气怎么样"})


def test_ask_auto_web_search_keyword_routes_to_agentic_rag(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
            tool_name="web_search",
            tool_input={"query": "联网搜索北京"},
            status="success",
            output_preview="北京信息",
            latency_ms=100,
        ),
        result={
            "success": True,
            "results": [
                {"title": "北京", "url": "https://example.com", "snippet": "北京信息"},
            ],
        },
    ))

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.retriever.search", new=AsyncMock(return_value=[])), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="联网回答")):

        response = client.post(
            "/ask",
            json={"user_id": "test_user", "question": "联网搜索北京"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "agentic_rag"
    assert data["answer"] == "联网回答"
    mock_tools.execute_with_result.assert_awaited_once_with("web_search", {"query": "联网搜索北京"})


def test_ask_routes_to_web_for_news(client):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.create_session = AsyncMock(return_value="test_session")
    mock_memory.append_turn = AsyncMock()

    mock_tools = MagicMock()
    mock_tools.execute_with_result = AsyncMock(return_value=ToolExecution(
        trace=ToolTrace(
        tool_name="web_search",
        tool_input={"query": "今天的新闻"},
        status="success",
        output_preview="新闻标题",
        latency_ms=100,
        ),
        result={
            "success": True,
            "results": [
                {"title": "新闻标题", "url": "https://example.com", "snippet": "新闻内容"},
            ],
        },
    ))

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.tools", mock_tools), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")):

        payload = {
            "user_id": "test_user",
            "question": "今天的新闻",
            "need_web": "always",
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

    with patch("app.orchestrator.orchestrator.memory", mock_memory), \
         patch("app.orchestrator.orchestrator.llm.generate", new=AsyncMock(return_value="回答")):

        payload = {
            "user_id": "test_user",
            "question": "计算 2+3 等于多少",
        }
        response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "tool"
    assert data["answer"] == "计算结果：5"


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
