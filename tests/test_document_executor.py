"""Phase 7: 执行智能体测试"""

from typing import Any

import pytest
from app.documents.adapters.base import DocumentBackend
from app.documents.adapters.docx import MockDocxBackend
from app.documents.adapters.pdf import MockPdfBackend
from app.documents.adapters.txt import MockTxtBackend
from app.documents.adapters.xlsx import MockXlsxBackend
from app.documents.executor import DocumentOperationAgent
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperation,
    DocumentOperationResult,
    DocumentPlan,
    FileType,
    RiskLevel,
)


class FixedResultBackend(DocumentBackend):
    """返回固定结果的测试后端，用于覆盖执行后校验分支。"""

    def __init__(self, result: DocumentOperationResult) -> None:
        self._result = result

    @property
    def name(self) -> str:
        return "fixed_result_backend"

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, plan: DocumentPlan) -> DocumentOperationResult:
        return self._result

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        return {}


@pytest.fixture
def agent():
    a = DocumentOperationAgent()
    a.register_backend(BackendType.DOCX_MCP, MockDocxBackend())
    a.register_backend(BackendType.XLSX_MCP, MockXlsxBackend())
    a.register_backend(BackendType.TEXT_ADAPTER, MockTxtBackend())
    a.register_backend(BackendType.PDF_READER, MockPdfBackend())
    return a


# ---------------------------------------------------------------------------
# 路由分发
# ---------------------------------------------------------------------------

class TestRouting:
    @pytest.mark.asyncio
    async def test_docx_routed_to_docx_backend(self, agent):
        backend = agent.get_backend(BackendType.DOCX_MCP)
        backend.load_document("/tmp/test.docx", ["段落1", "段落2"])
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path="/tmp/test.docx",
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="新内容"),
            ],
        )
        result = await agent.execute(plan)
        assert result.success

    @pytest.mark.asyncio
    async def test_xlsx_routed_to_xlsx_backend(self, agent):
        backend = agent.get_backend(BackendType.XLSX_MCP)
        backend.load_workbook("/tmp/test.xlsx", {"Sheet1": {"A1": "值"}})
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path="/tmp/test.xlsx",
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(action="modify_cell", target={"sheet": "Sheet1", "cell": "A1"}, value="新值"),
            ],
        )
        result = await agent.execute(plan)
        assert result.success

    @pytest.mark.asyncio
    async def test_txt_routed_to_txt_backend(self, agent):
        backend = agent.get_backend(BackendType.TEXT_ADAPTER)
        backend.load_file("/tmp/test.txt", ["行1", "行2"])
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path="/tmp/test.txt",
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(action="replace_line", target={"line": 1}, value="新行"),
            ],
        )
        result = await agent.execute(plan)
        assert result.success

    @pytest.mark.asyncio
    async def test_pdf_routed_to_pdf_backend(self, agent):
        backend = agent.get_backend(BackendType.PDF_READER)
        backend.load_document("/tmp/test.pdf", ["页1", "页2"])
        plan = DocumentPlan(
            intent=DocumentIntent.REVIEW,
            file_type=FileType.PDF,
            file_path="/tmp/test.pdf",
            backend_required=BackendType.PDF_READER,
        )
        result = await agent.execute(plan)
        assert result.success


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------

class TestErrors:
    @pytest.mark.asyncio
    async def test_unsupported_intent(self, agent):
        plan = DocumentPlan(
            intent=DocumentIntent.UNSUPPORTED,
            file_type=FileType.DOCX,
            unsupported_reason="不支持",
        )
        result = await agent.execute(plan)
        assert not result.success
        assert "不支持" in result.error

    @pytest.mark.asyncio
    async def test_none_backend_required(self, agent):
        plan = DocumentPlan.model_construct(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            backend_required=None,
        )
        result = await agent.execute(plan)
        assert not result.success
        assert "为空" in result.error

    @pytest.mark.asyncio
    async def test_unregistered_backend(self):
        agent = DocumentOperationAgent()  # 无后端注册
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path="/tmp/test.docx",
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="x"),
            ],
        )
        result = await agent.execute(plan)
        assert not result.success
        assert "未注册" in result.error

    @pytest.mark.asyncio
    async def test_clarification_plan_not_executed(self, agent):
        backend = agent.get_backend(BackendType.TEXT_ADAPTER)
        backend.load_file("/tmp/test.txt", ["行1", "行2"])
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path="/tmp/test.txt",
            backend_required=BackendType.TEXT_ADAPTER,
            clarification_question="请指定要修改的行号",
        )
        result = await agent.execute(plan)
        assert not result.success
        assert result.output_file is None
        assert "需要澄清" in result.error
        assert "请指定要修改的行号" in result.error

    @pytest.mark.asyncio
    async def test_empty_edit_plan_not_executed(self, agent):
        backend = agent.get_backend(BackendType.TEXT_ADAPTER)
        backend.load_file("/tmp/test.txt", ["行1", "行2"])
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path="/tmp/test.txt",
            backend_required=BackendType.TEXT_ADAPTER,
        )
        result = await agent.execute(plan)
        assert not result.success
        assert result.output_file is None
        assert "没有可执行操作" in result.error

    @pytest.mark.asyncio
    async def test_readonly_without_operations_still_executes(self, agent):
        backend = agent.get_backend(BackendType.TEXT_ADAPTER)
        backend.load_file("/tmp/test.txt", ["行1", "行2"])
        plan = DocumentPlan(
            intent=DocumentIntent.REVIEW,
            file_type=FileType.TXT,
            file_path="/tmp/test.txt",
            backend_required=BackendType.TEXT_ADAPTER,
        )
        result = await agent.execute(plan)
        assert result.success
        assert result.output_file is None
        assert result.summary == "只读操作完成"


# ---------------------------------------------------------------------------
# 执行后校验 warnings
# ---------------------------------------------------------------------------

class TestPostValidation:
    @pytest.mark.asyncio
    async def test_edit_success_without_output_file_warns(self):
        """编辑成功时应生成 output_file，否则记录 warning。"""
        agent = DocumentOperationAgent()
        agent.register_backend(
            BackendType.DOCX_MCP,
            FixedResultBackend(DocumentOperationResult(success=True)),
        )
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path="/tmp/test.docx",
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="新内容"),
            ],
        )
        result = await agent.execute(plan)
        assert result.success
        assert result.output_file is None
        assert "编辑操作成功但 output_file 为空" in result.warnings

    @pytest.mark.asyncio
    async def test_readonly_pdf_with_output_file_warns(self):
        """只读 PDF 生成 output_file 时应同时记录两个 warning。"""
        agent = DocumentOperationAgent()
        agent.register_backend(
            BackendType.PDF_READER,
            FixedResultBackend(DocumentOperationResult(success=True, output_file="/tmp/bad.pdf")),
        )
        plan = DocumentPlan(
            intent=DocumentIntent.SUMMARIZE,
            file_type=FileType.PDF,
            file_path="/tmp/test.pdf",
            backend_required=BackendType.PDF_READER,
        )
        result = await agent.execute(plan)
        assert result.success
        assert result.output_file == "/tmp/bad.pdf"
        assert "只读操作不应生成 output_file" in result.warnings
        assert "PDF 操作不应生成 output_file" in result.warnings


# ---------------------------------------------------------------------------
# 集成测试：完整流程
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.asyncio
    async def test_docx_edit_flow(self, agent):
        """DOCX 编辑完整流程：plan → execute → 副本生成"""
        backend = agent.get_backend(BackendType.DOCX_MCP)
        backend.load_document("/tmp/report.docx", [
            "标题：项目报告",
            "第一章 背景",
            "第二章 方法",
        ])
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path="/tmp/report.docx",
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="replace_paragraph",
                    target={"paragraph_index": 1},
                    value="第一章 项目背景",
                    description="补充标题",
                ),
                DocumentOperation(
                    action="append_paragraph",
                    target={},
                    value="第三章 结论",
                    description="追加结论",
                ),
            ],
        )
        result = await agent.execute(plan)
        assert result.success
        assert result.output_file is not None
        assert result.output_file != "/tmp/report.docx"
        # 原文件不变
        original = await backend.read_paragraph("/tmp/report.docx", 1)
        assert original == "第一章 背景"
        # 副本已修改
        copy_text = await backend.read_paragraph(result.output_file, 1)
        assert copy_text == "第一章 项目背景"

    @pytest.mark.asyncio
    async def test_xlsx_edit_flow(self, agent):
        """XLSX 编辑完整流程"""
        backend = agent.get_backend(BackendType.XLSX_MCP)
        backend.load_workbook("/tmp/data.xlsx", {
            "Sheet1": {"A1": "姓名", "B1": "工资", "A2": "张三", "B2": "10000"},
        })
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path="/tmp/data.xlsx",
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="modify_cell",
                    target={"sheet": "Sheet1", "cell": "B2"},
                    value="15000",
                    description="涨工资",
                ),
            ],
        )
        result = await agent.execute(plan)
        assert result.success
        assert await backend.read_cell(result.output_file, "Sheet1", "B2") == "15000"

    @pytest.mark.asyncio
    async def test_pdf_review_flow(self, agent):
        """PDF 审阅完整流程"""
        backend = agent.get_backend(BackendType.PDF_READER)
        backend.load_document("/tmp/paper.pdf", ["封面", "正文内容", "结论"])
        plan = DocumentPlan(
            intent=DocumentIntent.SUMMARIZE,
            file_type=FileType.PDF,
            file_path="/tmp/paper.pdf",
            backend_required=BackendType.PDF_READER,
        )
        result = await agent.execute(plan)
        assert result.success
        assert "3 页" in result.summary
        assert result.output_file is None
