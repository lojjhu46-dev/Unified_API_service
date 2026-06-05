"""Phase 8: 文档智能体工具测试"""

import pytest
import app.documents.tools as document_tools
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
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperation,
    DocumentOperationResult,
    DocumentPlan,
    FileType,
)
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


class FixedPlanner:
    """返回固定 DocumentPlan 的测试 planner，避免单测依赖真实 LLM。"""

    def __init__(self, plan: DocumentPlan) -> None:
        self._plan = plan

    async def plan(self, user_command: str, file_type: FileType, structure: dict | None = None) -> DocumentPlan:
        return self._plan


class SpyExecutor(DocumentOperationAgent):
    """记录 execute() 调用的测试 executor。"""

    def __init__(self) -> None:
        super().__init__()
        self.execute_called = False
        self.received_plan: DocumentPlan | None = None

    async def execute(self, plan: DocumentPlan) -> DocumentOperationResult:
        self.execute_called = True
        self.received_plan = plan
        return DocumentOperationResult(
            success=True,
            summary="via executor",
            verification={"structure": {"source": "executor"}},
        )


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
        executor = SpyExecutor()
        set_executor(executor)

        result = await document_review({"file_path": "/tmp/test.pdf"})
        assert result["success"] is True
        assert result["summary"] == "via executor"
        assert result["structure"] == {"source": "executor"}
        assert executor.execute_called is True
        assert executor.received_plan is not None
        assert executor.received_plan.intent == DocumentIntent.REVIEW
        assert executor.received_plan.file_type == FileType.PDF
        assert executor.received_plan.file_path == "/tmp/test.pdf"
        assert executor.received_plan.backend_required == BackendType.PDF_READER

    @pytest.mark.asyncio
    async def test_review_missing_path(self):
        result = await document_review({})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# document_plan
# ---------------------------------------------------------------------------

class TestDocumentPlan:
    @pytest.mark.asyncio
    async def test_plan_includes_file_path(self, monkeypatch):
        """document_plan 返回的 plan 必须包含 file_path"""
        monkeypatch.setattr(
            document_tools,
            "_planner",
            FixedPlanner(
                DocumentPlan(
                    intent=DocumentIntent.REVIEW,
                    file_type=FileType.PDF,
                    backend_required=BackendType.PDF_READER,
                )
            ),
        )
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
    async def test_apply_clarification_plan_returns_clarification(self):
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
                "clarification_question": "请指定要修改的行号",
            },
        })
        assert result["success"] is False
        assert result["requires_clarification"] is True
        assert result["clarification_question"] == "请指定要修改的行号"
        assert "需要澄清" in result["error"]
        assert result["plan"]["file_path"] == "/tmp/test.txt"

    @pytest.mark.asyncio
    async def test_apply_empty_edit_plan_rejected_before_executor(self):
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
            },
        })
        assert result["success"] is False
        assert "没有可执行操作" in result["error"]
        assert "requires_confirmation" not in result

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
    async def test_plan_then_apply_txt(self, monkeypatch):
        """document_plan 输出的 plan 可直接传给 document_apply_plan 执行"""
        executor = _make_executor_with_mocks()
        backend = executor.get_backend(BackendType.TEXT_ADAPTER)
        backend.load_file("/tmp/test.txt", ["第一行", "第二行", "第三行"])
        set_executor(executor)
        monkeypatch.setattr(
            document_tools,
            "_planner",
            FixedPlanner(
                DocumentPlan(
                    intent=DocumentIntent.EDIT,
                    file_type=FileType.TXT,
                    backend_required=BackendType.TEXT_ADAPTER,
                    operations=[
                        DocumentOperation(
                            action="replace_line",
                            target={"line": 2},
                            value="新内容",
                        ),
                    ],
                )
            ),
        )

        # 规划
        plan_result = await document_plan({
            "user_command": "把第二行改成新内容",
            "file_path": "/tmp/test.txt",
        })
        assert plan_result["success"] is True
        assert plan_result["plan"]["file_path"] == "/tmp/test.txt"

        # 执行
        apply_result = await document_apply_plan({"plan": plan_result["plan"]})
        assert apply_result["success"] is True
        assert apply_result["result"]["success"] is True
        assert apply_result["result"]["output_file"] is not None


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
