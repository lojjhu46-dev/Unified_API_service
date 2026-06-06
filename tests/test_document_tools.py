"""Phase 8: 文档智能体工具测试"""

import pytest
import app.documents.tools as document_tools
from app.config import settings
from app.documents.tools import (
    _reset_executor,
    document_apply_plan,
    document_extract,
    document_list_personal_files,
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
def _isolate_executor(tmp_path, monkeypatch):
    """每个测试前后重置 executor，避免全局状态污染。
    同时将 upload_dir 指向 tmp_path，让路径安全校验通过。"""
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(settings, "document_mcp_enabled", False)
    monkeypatch.setattr(
        "app.documents.tools.validate_edit_permission",
        lambda resolved_path, owner_user_id: None,
    )
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


def _create_file(tmp_path, name, content):
    """在 tmp_path 下创建文件并返回路径字符串"""
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return str(f)


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
# 默认 executor 含 TXT/PDF 真实后端
# ---------------------------------------------------------------------------

class TestDefaultExecutorHasRealBackends:
    @pytest.mark.asyncio
    async def test_default_has_txt_backend(self):
        executor = document_tools._get_executor()
        assert executor.get_backend(BackendType.TEXT_ADAPTER) is not None

    @pytest.mark.asyncio
    async def test_default_has_pdf_backend(self):
        executor = document_tools._get_executor()
        assert executor.get_backend(BackendType.PDF_READER) is not None

    @pytest.mark.asyncio
    async def test_default_no_docx_backend(self):
        executor = document_tools._get_executor()
        assert executor.get_backend(BackendType.DOCX_MCP) is None

    @pytest.mark.asyncio
    async def test_default_no_xlsx_backend(self):
        executor = document_tools._get_executor()
        assert executor.get_backend(BackendType.XLSX_MCP) is None


# ---------------------------------------------------------------------------
# 非法 file_type
# ---------------------------------------------------------------------------

class TestInvalidFileType:
    @pytest.mark.asyncio
    async def test_extract_invalid_file_type(self, tmp_path):
        set_executor(_make_executor_with_mocks())
        f = _create_file(tmp_path, "test.xyz", "内容")
        result = await document_extract({"file_path": f})
        assert result["success"] is False
        assert "推断" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_explicit_invalid_file_type(self, tmp_path):
        set_executor(_make_executor_with_mocks())
        f = _create_file(tmp_path, "test.txt", "内容")
        result = await document_extract({"file_path": f, "file_type": "json"})
        assert result["success"] is False
        assert "json" in result["error"]

    @pytest.mark.asyncio
    async def test_plan_invalid_file_type(self, tmp_path):
        result = await document_plan({
            "user_command": "总结文档",
            "file_path": str(tmp_path / "test.xyz"),
        })
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_review_invalid_file_type(self, tmp_path):
        result = await document_review({"file_path": str(tmp_path / "test.xyz")})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# document_extract
# ---------------------------------------------------------------------------

class TestDocumentExtract:
    @pytest.mark.asyncio
    async def test_extract_docx(self, tmp_path):
        f = _create_file(tmp_path, "test.docx", "")  # 占位文件让路径安全通过
        executor = _make_executor_with_mocks()
        executor.get_backend(BackendType.DOCX_MCP).load_document(f, ["段落1", "段落2"])
        set_executor(executor)

        result = await document_extract({"file_path": f})
        assert result["success"] is True
        assert result["file_type"] == "docx"
        assert result["structure"]["paragraph_count"] == 2

    @pytest.mark.asyncio
    async def test_extract_missing_path(self):
        result = await document_extract({})
        assert result["success"] is False
        assert "file_path" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_file_not_found(self, tmp_path):
        set_executor(_make_executor_with_mocks())
        result = await document_extract({"file_path": str(tmp_path / "nonexistent.txt")})
        assert result["success"] is False
        assert "不存在" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_real_txt(self, tmp_path):
        """真实 TXT 后端提取结构"""
        f = _create_file(tmp_path, "test.txt", "第一行\n第二行\n第三行")
        result = await document_extract({"file_path": f})
        assert result["success"] is True
        assert result["structure"]["line_count"] == 3


# ---------------------------------------------------------------------------
# document_review
# ---------------------------------------------------------------------------

class TestDocumentReview:
    @pytest.mark.asyncio
    async def test_review_via_executor(self, tmp_path):
        """document_review 应经过 executor.execute()"""
        f = _create_file(tmp_path, "test.pdf", "")
        executor = SpyExecutor()
        set_executor(executor)

        result = await document_review({"file_path": f})
        assert result["success"] is True
        assert result["summary"] == "via executor"
        assert executor.execute_called is True
        assert executor.received_plan.intent == DocumentIntent.REVIEW
        assert executor.received_plan.file_type == FileType.PDF

    @pytest.mark.asyncio
    async def test_review_missing_path(self):
        result = await document_review({})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_review_real_txt(self, tmp_path):
        """真实 TXT 后端审阅（使用默认 executor，含 RealTxtBackend）"""
        f = _create_file(tmp_path, "test.txt", "第一行\n第二行")
        result = await document_review({"file_path": f})
        assert result["success"] is True
        assert result["structure"]["line_count"] == 2


# ---------------------------------------------------------------------------
# document_plan
# ---------------------------------------------------------------------------

class TestDocumentPlan:
    @pytest.mark.asyncio
    async def test_plan_includes_file_path(self, monkeypatch, tmp_path):
        """document_plan 返回的 plan 必须包含 file_path"""
        f = _create_file(tmp_path, "test.pdf", "")
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
            "file_path": f,
        })
        assert result["success"] is True
        assert result["plan"]["file_path"] == f

    @pytest.mark.asyncio
    async def test_plan_pdf_edit_rejected(self, tmp_path):
        f = _create_file(tmp_path, "test.pdf", "")
        result = await document_plan({
            "user_command": "删除第三段",
            "file_path": f,
        })
        assert result["success"] is True
        plan = result["plan"]
        assert plan["intent"] == "unsupported"
        assert "PDF" in (plan["unsupported_reason"] or "")

    @pytest.mark.asyncio
    async def test_plan_missing_command(self, tmp_path):
        f = _create_file(tmp_path, "test.txt", "")
        result = await document_plan({"file_path": f})
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
    async def test_apply_txt_edit_requires_confirmation(self, tmp_path):
        f = _create_file(tmp_path, "test.txt", "行1\n行2\n行3")

        result = await document_apply_plan({
            "plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": f,
                "backend_required": "text_adapter",
                "operations": [
                    {"action": "replace_line", "target": {"line": 1}, "value": "新行1"},
                ],
            },
        })
        assert result["success"] is False
        assert result["requires_confirmation"] is True

    @pytest.mark.asyncio
    async def test_apply_txt_edit_confirmed(self, tmp_path):
        f = _create_file(tmp_path, "test.txt", "行1\n行2\n行3")

        result = await document_apply_plan({
            "plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": f,
                "backend_required": "text_adapter",
                "operations": [
                    {"action": "replace_line", "target": {"line": 1}, "value": "新行1"},
                ],
            },
            "confirmed": True,
        })
        assert result["success"] is True
        assert result["result"]["output_file"] is not None

    @pytest.mark.asyncio
    async def test_apply_missing_plan(self):
        result = await document_apply_plan({})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_apply_clarification_plan_returns_clarification(self, tmp_path):
        f = _create_file(tmp_path, "test.txt", "行1\n行2")

        result = await document_apply_plan({
            "plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": f,
                "backend_required": "text_adapter",
                "clarification_question": "请指定要修改的行号",
            },
        })
        assert result["success"] is False
        assert result["requires_clarification"] is True
        assert result["clarification_question"] == "请指定要修改的行号"

    @pytest.mark.asyncio
    async def test_apply_empty_edit_plan_rejected_before_executor(self, tmp_path):
        f = _create_file(tmp_path, "test.txt", "行1\n行2")

        result = await document_apply_plan({
            "plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": f,
                "backend_required": "text_adapter",
            },
        })
        assert result["success"] is False
        assert "没有可执行操作" in result["error"]

    @pytest.mark.asyncio
    async def test_apply_high_risk_requires_confirmation(self, tmp_path):
        f = _create_file(tmp_path, "test.txt", "行1\n行2")
        result = await document_apply_plan({
            "plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": f,
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
    async def test_apply_high_risk_confirmed(self, tmp_path):
        f = _create_file(tmp_path, "test.txt", "行1\n行2")
        result = await document_apply_plan({
            "plan": {
                "intent": "edit",
                "file_type": "txt",
                "file_path": f,
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
    async def test_plan_then_apply_txt(self, monkeypatch, tmp_path):
        """document_plan 输出的 plan 可直接传给 document_apply_plan 执行"""
        f = _create_file(tmp_path, "test.txt", "第一行\n第二行\n第三行")
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
            "file_path": f,
        })
        assert plan_result["success"] is True
        assert plan_result["plan"]["file_path"] == f

        # 执行
        apply_result = await document_apply_plan({"plan": plan_result["plan"], "confirmed": True})
        assert apply_result["success"] is True
        assert apply_result["result"]["output_file"] is not None


# ---------------------------------------------------------------------------
# 路径安全
# ---------------------------------------------------------------------------

class TestPathSecurityInTools:
    @pytest.mark.asyncio
    async def test_extract_outside_upload_dir_rejected(self, tmp_path):
        """目录外文件被拒绝"""
        outside = tmp_path.parent / "outside_doc_tools"
        outside.mkdir(exist_ok=True)
        f = outside / "test.txt"
        f.write_text("内容")
        result = await document_extract({"file_path": str(f)})
        assert result["success"] is False
        assert "不在允许的目录内" in result["error"]
        f.unlink()
        outside.rmdir()


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
    async def test_document_extract_via_registry(self, tmp_path):
        f = _create_file(tmp_path, "test.txt", "行1")
        executor = _make_executor_with_mocks()
        executor.get_backend(BackendType.TEXT_ADAPTER).load_file(f, ["行1"])
        set_executor(executor)

        execution = await tool_registry.execute_with_result(
            "document_extract",
            {"file_path": f},
        )
        assert execution.trace.status == "success"
        assert execution.result["success"] is True


# ---------------------------------------------------------------------------
# 真实权限校验（不 mock validate_edit_permission）
# ---------------------------------------------------------------------------

class TestRealEditPermission:
    """直接单测 validate_edit_permission，不依赖 autouse fixture 的 mock"""

    @pytest.fixture
    def seeded_registry(self, tmp_path, monkeypatch):
        from app.retrieval.document_registry import DocumentRegistry
        db_path = str(tmp_path / "reg.sqlite3")
        reg = DocumentRegistry(db_path=db_path)
        reg.init()
        # 本人 personal ready 文件
        reg.create_processing(
            document_id="doc_personal", tenant_id="default",
            knowledge_base_type="personal", owner_user_id="owner1",
            original_filename="test.txt", stored_filename="test.txt",
            stored_path=str(tmp_path / "test.txt"),
        )
        reg.mark_ready("doc_personal", chunk_count=3)
        # 企业库文件
        reg.create_processing(
            document_id="doc_enterprise", tenant_id="default",
            knowledge_base_type="enterprise", owner_user_id="owner1",
            original_filename="ent.txt", stored_filename="ent.txt",
            stored_path=str(tmp_path / "ent.txt"),
        )
        reg.mark_ready("doc_enterprise", chunk_count=3)
        # 其他用户的文件
        reg.create_processing(
            document_id="doc_other", tenant_id="default",
            knowledge_base_type="personal", owner_user_id="owner2",
            original_filename="other.txt", stored_filename="other.txt",
            stored_path=str(tmp_path / "other.txt"),
        )
        reg.mark_ready("doc_other", chunk_count=3)
        # processing 文件
        reg.create_processing(
            document_id="doc_proc", tenant_id="default",
            knowledge_base_type="personal", owner_user_id="owner1",
            original_filename="proc.txt", stored_filename="proc.txt",
            stored_path=str(tmp_path / "proc.txt"),
        )
        monkeypatch.setattr("app.retrieval.document_registry.document_registry", reg)
        return tmp_path

    def test_own_personal_ready_passes(self, seeded_registry):
        from app.documents.path_security import validate_edit_permission
        result = validate_edit_permission(seeded_registry / "test.txt", "owner1")
        assert result is None

    def test_enterprise_file_rejected(self, seeded_registry):
        from app.documents.path_security import validate_edit_permission
        result = validate_edit_permission(seeded_registry / "ent.txt", "owner1")
        assert result is not None
        assert "只能编辑" in result

    def test_other_user_rejected(self, seeded_registry):
        from app.documents.path_security import validate_edit_permission
        result = validate_edit_permission(seeded_registry / "other.txt", "owner1")
        assert result is not None
        assert "只能编辑" in result

    def test_unregistered_file_rejected(self, seeded_registry):
        from app.documents.path_security import validate_edit_permission
        result = validate_edit_permission(seeded_registry / "nope.txt", "owner1")
        assert result is not None
        assert "只能编辑" in result

    def test_processing_file_rejected(self, seeded_registry):
        from app.documents.path_security import validate_edit_permission
        result = validate_edit_permission(seeded_registry / "proc.txt", "owner1")
        assert result is not None
        assert "只能编辑" in result

    def test_empty_owner_rejected(self, seeded_registry):
        from app.documents.path_security import validate_edit_permission
        result = validate_edit_permission(seeded_registry / "test.txt", "")
        assert result is not None
        assert "认证" in result


# ---------------------------------------------------------------------------
# document_id 入口与列表脱敏
# ---------------------------------------------------------------------------

class TestPersonalDocumentId:
    @pytest.fixture
    def registry_with_personal_file(self, tmp_path, monkeypatch):
        from app.retrieval.document_registry import DocumentRegistry

        f = tmp_path / "test.txt"
        f.write_text("第一行\n第二行", encoding="utf-8")

        reg = DocumentRegistry(db_path=str(tmp_path / "reg.sqlite3"))
        reg.init()
        reg.create_processing(
            document_id="doc_public_id",
            tenant_id="default",
            knowledge_base_type="personal",
            owner_user_id="owner1",
            original_filename="test.txt",
            stored_filename="test.txt",
            stored_path=str(f),
        )
        reg.mark_ready("doc_public_id", chunk_count=2)
        monkeypatch.setattr("app.retrieval.document_registry.document_registry", reg)
        return f

    @pytest.mark.asyncio
    async def test_document_list_does_not_expose_stored_path(self, registry_with_personal_file):
        result = await document_list_personal_files({
            "owner_user_id": "owner1",
        })
        assert result["success"] is True
        assert result["files"] == [{
            "document_id": "doc_public_id",
            "original_filename": "test.txt",
            "created_at": result["files"][0]["created_at"],
        }]
        assert "stored_path" not in result["files"][0]

    @pytest.mark.asyncio
    async def test_document_plan_resolves_document_id(self, monkeypatch, registry_with_personal_file):
        monkeypatch.setattr(
            document_tools,
            "_planner",
            FixedPlanner(
                DocumentPlan(
                    intent=DocumentIntent.REVIEW,
                    file_type=FileType.TXT,
                    backend_required=BackendType.TEXT_ADAPTER,
                )
            ),
        )
        result = await document_plan({
            "user_command": "审阅文档",
            "document_id": "doc_public_id",
            "owner_user_id": "owner1",
        })
        assert result["success"] is True
        assert result["plan"]["file_path"] == str(registry_with_personal_file.resolve())

    @pytest.mark.asyncio
    async def test_document_review_resolves_document_id(self, registry_with_personal_file):
        result = await document_review({
            "document_id": "doc_public_id",
            "owner_user_id": "owner1",
        })
        assert result["success"] is True
        assert result["structure"]["line_count"] == 2
