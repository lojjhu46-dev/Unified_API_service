"""文档工具接入主编排路径测试"""

import json
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.config import settings
from app.main import app
from app.documents.tools import set_executor, _reset_executor
from app.documents.executor import DocumentOperationAgent
from app.documents.models import BackendType
from app.documents.adapters.docx import MockDocxBackend
from app.documents.adapters.txt import MockTxtBackend
from app.documents.adapters.pdf import MockPdfBackend


@pytest.fixture
def client():
    with patch.object(settings, "api_key", None), patch.object(settings, "llm_provider", "mock"):
        yield TestClient(app)


@pytest.fixture(autouse=True)
def _isolate_executor(monkeypatch):
    """重置 executor 并 mock 路径安全（集成测试用虚拟路径）"""
    from pathlib import Path
    _reset_executor()
    # 集成测试使用 /tmp/ 虚拟路径，mock 路径安全使其通过
    monkeypatch.setattr(
        "app.documents.tools.validate_file_path",
        lambda fp: (Path(fp), None),
    )
    monkeypatch.setattr(
        "app.documents.tools.validate_edit_permission",
        lambda resolved_path, owner_user_id: None,
    )
    yield
    _reset_executor()


def _setup_executor_with_txt():
    """注入带 TXT mock 后端的执行器"""
    executor = DocumentOperationAgent()
    backend = MockTxtBackend()
    backend.load_file("/tmp/test.txt", ["第一行", "第二行", "第三行"])
    executor.register_backend(BackendType.TEXT_ADAPTER, backend)
    set_executor(executor)
    return executor


# ---------------------------------------------------------------------------
# 路由识别
# ---------------------------------------------------------------------------

class TestDocumentRouting:
    def test_structured_file_path_routes_to_tool(self, client):
        """传 document_file_path 应走 tool 路由"""
        _setup_executor_with_txt()
        payload = {
            "user_id": "test_user",
            "question": "审阅文档",
            "document_file_path": "/tmp/test.txt",
            "document_action": "review",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "审阅" in data["answer"] or "文档" in data["answer"]

    def test_natural_language_doc_path_routes_to_tool(self, client):
        """自然语言中的文件路径 + 意图词应走 tool 路由"""
        _setup_executor_with_txt()
        payload = {
            "user_id": "test_user",
            "question": "总结 /tmp/test.txt 的内容",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"

    def test_document_plan_routes_to_tool(self, client):
        """传 document_plan 应走 tool 路由"""
        payload = {
            "user_id": "test_user",
            "question": "执行编辑",
            "document_plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": "/tmp/test.txt",
                "backend_required": "text_adapter",
                "operations": [
                    {"action": "replace_line", "target": {"line": 1}, "value": "新内容"},
                ],
            },
            "document_confirmed": True,
        }
        _setup_executor_with_txt()
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"

    def test_normal_question_not_routed_to_document(self, client):
        """普通问题不应走文档路由"""
        payload = {
            "user_id": "test_user",
            "question": "什么是机器学习？",
        }
        with patch(
            "app.orchestrator.orchestrator.llm.generate",
            new=AsyncMock(return_value="机器学习是..."),
        ):
            response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "rag"

    def test_personal_knowledge_query_not_routed_to_document_list(self, client):
        """查询个人知识库内容不应被列文件意图劫持。"""
        payload = {
            "user_id": "test_user",
            "question": "查看个人知识库文档里的排序算法",
        }
        with patch(
            "app.orchestrator.orchestrator.llm.generate",
            new=AsyncMock(return_value="排序算法相关内容..."),
        ):
            response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "rag"

    def test_document_id_routes_to_tool(self, client, monkeypatch, tmp_path):
        """传 document_id 应走文档工具，由服务端解析内部路径。"""
        from app.retrieval.document_registry import DocumentRegistry

        f = tmp_path / "test.txt"
        f.write_text("第一行\n第二行", encoding="utf-8")
        reg = DocumentRegistry(db_path=str(tmp_path / "reg.sqlite3"))
        reg.init()
        reg.create_processing(
            document_id="doc_public_id",
            tenant_id="default",
            knowledge_base_type="personal",
            owner_user_id="test_user",
            original_filename="test.txt",
            stored_filename="test.txt",
            stored_path=str(f),
        )
        reg.mark_ready("doc_public_id", chunk_count=2)
        monkeypatch.setattr("app.retrieval.document_registry.document_registry", reg)

        payload = {
            "user_id": "test_user",
            "question": "审阅文档",
            "document_id": "doc_public_id",
            "document_action": "review",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "文档审阅结果" in data["answer"]

    def test_calculator_not_affected(self, client):
        """计算题不受文档路由影响"""
        payload = {
            "user_id": "test_user",
            "question": "计算 2+3 等于多少",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "计算结果" in data["answer"]

    def test_document_delete_content_without_path_does_not_route_to_calculator(self, client):
        """删除文档内容但缺少路径时，应提示补充文档路径，不应把编号列表误判为计算。"""
        payload = {
            "user_id": "test_user",
            "question": (
                "删除文档中的内容：\n"
                "实验目的：\n"
                "（1）掌握 Pandas 读取数据及 Matplotlib 绘制散点图、折线图、饼图等多类型图表的方法；\n"
                "（2）学会生成随机数据并绘制带折线的直方图，掌握数据分布可视化技巧；"
            ),
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "文档文件路径" in data["answer"] or "document_file_path" in data["answer"]
        assert "计算结果" not in data["answer"]


# ---------------------------------------------------------------------------
# 文档操作执行
# ---------------------------------------------------------------------------

class TestDocumentOperations:
    def test_list_personal_files_via_ask_shows_document_id(self, client, monkeypatch, tmp_path):
        """列表入口只展示 document_id、文件名和保存时间，不展示 stored_path。"""
        from app.retrieval.document_registry import DocumentRegistry

        f = tmp_path / "test.txt"
        f.write_text("内容", encoding="utf-8")
        reg = DocumentRegistry(db_path=str(tmp_path / "reg.sqlite3"))
        reg.init()
        reg.create_processing(
            document_id="doc_public_id",
            tenant_id="default",
            knowledge_base_type="personal",
            owner_user_id="test_user",
            original_filename="test.txt",
            stored_filename="test.txt",
            stored_path=str(f),
        )
        reg.mark_ready("doc_public_id", chunk_count=1)
        monkeypatch.setattr("app.retrieval.document_registry.document_registry", reg)

        payload = {
            "user_id": "test_user",
            "question": "列出我的文件",
            "document_action": "list",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "doc_public_id" in data["answer"]
        assert "test.txt" in data["answer"]
        assert str(f) not in data["answer"]

    def test_extract_via_ask(self, client):
        """通过 /ask 提取文档结构"""
        _setup_executor_with_txt()
        payload = {
            "user_id": "test_user",
            "question": "提取文档结构",
            "document_file_path": "/tmp/test.txt",
            "document_action": "extract",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "提取成功" in data["answer"] or "结构" in data["answer"]

    def test_review_via_ask(self, client):
        """通过 /ask 审阅文档"""
        _setup_executor_with_txt()
        payload = {
            "user_id": "test_user",
            "question": "审阅文档",
            "document_file_path": "/tmp/test.txt",
            "document_action": "review",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "审阅" in data["answer"] or "文档" in data["answer"]

    def test_plan_via_ask_returns_edit_plan(self, client):
        """通过 /ask 生成编辑方案，应返回可执行 edit plan"""
        _setup_executor_with_txt()
        # 模拟 planner 返回可执行 edit plan
        mock_plan_data = {
            "intent": "edit",
            "file_type": "txt",
            "file_path": "/tmp/test.txt",
            "backend_required": "text_adapter",
            "operations": [
                {"action": "replace_line", "target": {"line": 1}, "value": "新内容", "description": "替换第一行"},
            ],
            "risk_level": "low",
            "requires_confirmation": False,
            "clarification_question": None,
            "unsupported_reason": None,
        }
        with patch(
            "app.documents.planner.llm_gateway.generate",
            new=AsyncMock(return_value=json.dumps(mock_plan_data)),
        ):
            payload = {
                "user_id": "test_user",
                "question": "把第一行改成新内容",
                "document_file_path": "/tmp/test.txt",
                "document_action": "plan",
            }
            response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "编辑方案" in data["answer"]
        assert "确认" in data["answer"]

    def test_plan_unsupported_via_ask(self, client):
        """PDF 编辑方案应返回 unsupported 提示"""
        payload = {
            "user_id": "test_user",
            "question": "删除第三段",
            "document_file_path": "/tmp/test.pdf",
            "document_action": "plan",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "无法生成" in data["answer"] or "不支持" in data["answer"]

    def test_plan_clarification_via_ask(self, client):
        """模糊命令应返回澄清问题"""
        mock_plan_data = {
            "intent": "edit",
            "file_type": "txt",
            "file_path": "/tmp/test.txt",
            "backend_required": "text_adapter",
            "operations": [],
            "risk_level": "low",
            "requires_confirmation": False,
            "clarification_question": "请指定要删除的行号范围",
            "unsupported_reason": None,
        }
        with patch(
            "app.documents.planner.llm_gateway.generate",
            new=AsyncMock(return_value=json.dumps(mock_plan_data)),
        ):
            payload = {
                "user_id": "test_user",
                "question": "删除实验体会的内容",
                "document_file_path": "/tmp/test.txt",
                "document_action": "plan",
            }
            response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "补充信息" in data["answer"] or "请指定" in data["answer"]

    def test_apply_confirmed_via_ask(self, client):
        """通过 /ask 确认执行编辑"""
        _setup_executor_with_txt()
        payload = {
            "user_id": "test_user",
            "question": "执行编辑",
            "document_plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": "/tmp/test.txt",
                "backend_required": "text_adapter",
                "operations": [
                    {"action": "replace_line", "target": {"line": 1}, "value": "新标题"},
                ],
            },
            "document_confirmed": True,
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "成功" in data["answer"]

    def test_apply_unconfirmed_edit_requires_confirmation(self, client):
        """低风险编辑未确认也应返回确认提示"""
        payload = {
            "user_id": "test_user",
            "question": "执行编辑",
            "document_plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": "/tmp/test.txt",
                "backend_required": "text_adapter",
                "risk_level": "low",
                "requires_confirmation": False,
                "operations": [
                    {"action": "replace_line", "target": {"line": 1}, "value": "新内容"},
                ],
            },
            "document_confirmed": False,
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "确认" in data["answer"]

    def test_apply_unconfirmed_high_risk_requires_confirmation(self, client):
        """高风险编辑未确认也应返回确认提示"""
        payload = {
            "user_id": "test_user",
            "question": "执行编辑",
            "document_plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": "/tmp/test.txt",
                "backend_required": "text_adapter",
                "risk_level": "high",
                "requires_confirmation": True,
                "operations": [
                    {"action": "delete_line", "target": {"line": 1}},
                ],
            },
            "document_confirmed": False,
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "确认" in data["answer"]

    def test_apply_without_plan_returns_hint(self, client):
        """apply 但没有 plan 应返回提示"""
        payload = {
            "user_id": "test_user",
            "question": "执行编辑",
            "document_action": "apply",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "tool"
        assert "方案" in data["answer"]


# ---------------------------------------------------------------------------
# 工具轨迹
# ---------------------------------------------------------------------------

class TestToolTrace:
    def test_document_tool_trace_present(self, client):
        """文档操作应有 tool_trace"""
        _setup_executor_with_txt()
        payload = {
            "user_id": "test_user",
            "question": "审阅文档",
            "document_file_path": "/tmp/test.txt",
            "document_action": "review",
            "return_trace": True,
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["tool_trace"]) > 0
        assert data["tool_trace"][0]["tool_name"] == "document_review"
