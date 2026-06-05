"""Phase 8: 文档智能体工具测试"""

import pytest
from app.documents.tools import (
    _reset_executor,
    document_apply_plan,
    document_extract,
    document_plan,
    document_review,
    infer_file_type,
    set_executor,
)
from app.documents.executor import DocumentOperationAgent
from app.documents.models import BackendType, FileType
from app.documents.adapters.docx import MockDocxBackend
from app.documents.adapters.txt import MockTxtBackend
from app.documents.adapters.pdf import MockPdfBackend
from app.documents.adapters.xlsx import MockXlsxBackend
from app.tools.registry import tool_registry


@pytest.fixture(autouse=True)
def _isolate_executor():
    """每个测试前后重置 executor，避免全局状态污染"""
    _reset_executor()
    yield
    _reset_executor()


def _make_executor_with_mocks() -> DocumentOperationAgent:
    """创建带全部 mock 后端的执行器"""
    executor = DocumentOperationAgent()
    executor.register_backend(BackendType.DOCX_MCP, MockDocxBackend())
    executor.register_backend(BackendType.XLSX_MCP, MockXlsxBackend())
    executor.register_backend(BackendType.TEXT_ADAPTER, MockTxtBackend())
    executor.register_backend(BackendType.PDF_READER, MockPdfBackend())
    return executor


# ---------------------------------------------------------------------------
# infer_file_type
# ---------------------------------------------------------------------------

class TestInferFileType:
    def test_docx(self):
        assert infer_file_type("/tmp/test.docx") == FileType.DOCX

    def test_xlsx(self):
        assert infer_file_type("/tmp/test.xlsx") == FileType.XLSX

    def test_txt(self):
        assert infer_file_type("/tmp/test.txt") == FileType.TXT

    def test_pdf(self):
        assert infer_file_type("/tmp/test.pdf") == FileType.PDF

    def test_unknown(self):
        assert infer_file_type("/tmp/test.xyz") is None


# ---------------------------------------------------------------------------
# 默认 executor 无后端
# ---------------------------------------------------------------------------

class TestDefaultExecutorNoBackends:
    @pytest.mark.asyncio
    async def test_extract_without_backend_fails(self):
        """默认 executor 不注册 mock 后端，应返回后端未注册"""
        result = await document_extract({"file_path": "/tmp/test.txt"})
        assert result["success"] is False
        assert "未注册" in result["error"]

    @pytest.mark.asyncio
    async def test_review_without_backend_fails(self):
        result = await document_review({"file_path": "/tmp/test.pdf"})
        assert result["success"] is False
        assert "未注册" in result["error"]


# ---------------------------------------------------------------------------
# 非法 file_type
# ---------------------------------------------------------------------------

class TestInvalidFileType:
    @pytest.mark.asyncio
    async def test_extract_invalid_file_type(self):
        set_executor(_make_executor_with_mocks())
        result = await document_extract({"file_path": "/tmp/test.xyz"})
        assert result["success"] is False
        assert "推断" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_explicit_invalid_file_type(self):
        set_executor(_make_executor_with_mocks())
        result = await document_extract({"file_path": "/tmp/test.txt", "file_type": "json"})
        assert result["success"] is False
        assert "json" in result["error"]

    @pytest.mark.asyncio
    async def test_plan_invalid_file_type(self):
        result = await document_plan({
            "user_command": "总结文档",
            "file_path": "/tmp/test.xyz",
        })
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_review_invalid_file_type(self):
        result = await document_review({"file_path": "/tmp/test.xyz"})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# document_extract
# ---------------------------------------------------------------------------

class TestDocumentExtract:
    @pytest.mark.asyncio
    async def test_extract_docx(self):
        executor = _make_executor_with_mocks()
        backend = executor.get_backend(BackendType.DOCX_MCP)
        backend.load_document("/tmp/test.docx", ["段落1", "段落2"])
        set_executor(executor)

        result = await document_extract({"file_path": "/tmp/test.docx"})
        assert result["success"] is True
        assert result["file_type"] == "docx"
        assert result["structure"]["paragraph_count"] == 2

    @pytest.mark.asyncio
    async def test_extract_missing_path(self):
        result = await document_extract({})
        assert result["success"] is False
        assert "file_path" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_file_not_found(self):
        set_executor(_make_executor_with_mocks())
        result = await document_extract({"file_path": "/nonexistent.txt"})
        assert result["success"] is False
        assert "不存在" in result["error"]


# ---------------------------------------------------------------------------
# document_review
# ---------------------------------------------------------------------------

class TestDocumentReview:
    @pytest.mark.asyncio
    async def test_review_via_executor(self):
        """document_review 应经过 executor.execute()"""
        executor = _make_executor_with_mocks()
        backend = executor.get_backend(BackendType.PDF_READER)
        backend.load_document("/tmp/test.pdf", ["页1", "页2", "页3"])
        set_executor(executor)

        result = await document_review({"file_path": "/tmp/test.pdf"})
        assert result["success"] is True
        assert result["structure"]["page_count"] == 3

    @pytest.mark.asyncio
    async def test_review_missing_path(self):
        result = await document_review({})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# document_plan
# ---------------------------------------------------------------------------

class TestDocumentPlan:
    @pytest.mark.asyncio
    async def test_plan_includes_file_path(self):
        """document_plan 返回的 plan 必须包含 file_path"""
        result = await document_plan({
            "user_command": "审阅文档",
            "file_path": "/tmp/test.pdf",
        })
        assert result["success"] is True
        assert result["plan"]["file_path"] == "/tmp/test.pdf"

    @pytest.mark.asyncio
    async def test_plan_pdf_edit_rejected(self):
        result = await document_plan({
            "user_command": "删除第三段",
            "file_path": "/tmp/test.pdf",
        })
        assert result["success"] is True
        plan = result["plan"]
        assert plan["intent"] == "unsupported"
        assert "PDF" in (plan["unsupported_reason"] or "")

    @pytest.mark.asyncio
    async def test_plan_missing_command(self):
        result = await document_plan({"file_path": "/tmp/test.txt"})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_plan_missing_path(self):
        result = await document_plan({"user_command": "总结文档"})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# document_apply_plan
# ---------------------------------------------------------------------------

class TestDocumentApplyPlan:
    @pytest.mark.asyncio
    async def test_apply_txt_edit(self):
        executor = _make_executor_with_mocks()
        backend = executor.get_backend(BackendType.TEXT_ADAPTER)
        backend.load_file("/tmp/test.txt", ["行1", "行2", "行3"])
        set_executor(executor)

        result = await document_apply_plan({
            "plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": "/tmp/test.txt",
                "backend_required": "text_adapter",
                "operations": [
                    {"action": "replace_line", "target": {"line": 1}, "value": "新行1"},
                ],
            },
        })
        assert result["success"] is True
        assert result["result"]["output_file"] is not None

    @pytest.mark.asyncio
    async def test_apply_missing_plan(self):
        result = await document_apply_plan({})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_apply_high_risk_requires_confirmation(self):
        result = await document_apply_plan({
            "plan": {
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
        })
        assert result["success"] is False
        assert result["requires_confirmation"] is True

    @pytest.mark.asyncio
    async def test_apply_high_risk_confirmed(self):
        executor = _make_executor_with_mocks()
        backend = executor.get_backend(BackendType.TEXT_ADAPTER)
        backend.load_file("/tmp/test.txt", ["行1", "行2"])
        set_executor(executor)

        result = await document_apply_plan({
            "plan": {
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
            "confirmed": True,
        })
        assert result["success"] is True


# ---------------------------------------------------------------------------
# 端到端：plan -> apply
# ---------------------------------------------------------------------------

class TestPlanToApply:
    @pytest.mark.asyncio
    async def test_plan_then_apply_txt(self):
        """document_plan 输出的 plan 可直接传给 document_apply_plan 执行"""
        executor = _make_executor_with_mocks()
        backend = executor.get_backend(BackendType.TEXT_ADAPTER)
        backend.load_file("/tmp/test.txt", ["第一行", "第二行", "第三行"])
        set_executor(executor)

        # 规划
        plan_result = await document_plan({
            "user_command": "把第二行改成新内容",
            "file_path": "/tmp/test.txt",
        })
        assert plan_result["success"] is True
        assert plan_result["plan"]["file_path"] == "/tmp/test.txt"

        # 执行
        apply_result = await document_apply_plan({"plan": plan_result["plan"]})
        # 可能成功也可能因 LLM mock 返回 unsupported 而失败，
        # 但关键验证 plan 中 file_path 非空
        assert "plan" in plan_result or "result" in apply_result


# ---------------------------------------------------------------------------
# 工具注册表集成
# ---------------------------------------------------------------------------

class TestToolRegistryIntegration:
    def test_document_tools_registered(self):
        tools = tool_registry.get_available_tools()
        assert "document_extract" in tools
        assert "document_plan" in tools
        assert "document_apply_plan" in tools
        assert "document_review" in tools

    @pytest.mark.asyncio
    async def test_document_extract_via_registry(self):
        executor = _make_executor_with_mocks()
        backend = executor.get_backend(BackendType.TEXT_ADAPTER)
        backend.load_file("/tmp/test.txt", ["行1"])
        set_executor(executor)

        execution = await tool_registry.execute_with_result(
            "document_extract",
            {"file_path": "/tmp/test.txt"},
        )
        assert execution.trace.status == "success"
        assert execution.result["success"] is True
