"""Phase 3: DOCX 后端 adapter 测试"""

import pytest
from unittest.mock import AsyncMock

from app.documents.adapters.base import BackendUnavailableError
from app.documents.adapters.docx import MockDocxBackend
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperation,
    DocumentPlan,
    FileType,
    RiskLevel,
)


@pytest.fixture
def backend():
    return MockDocxBackend()


@pytest.fixture
def sample_doc(backend):
    """加载模拟文档"""
    backend.load_document("/tmp/test.docx", [
        "标题：项目报告",
        "第一章 背景",
        "本项目旨在提升系统性能。",
        "第二章 方法",
        "采用分布式架构。",
        "第三章 结果",
        "性能提升50%。",
    ])
    return "/tmp/test.docx"


# ---------------------------------------------------------------------------
# read_structure
# ---------------------------------------------------------------------------

class TestReadStructure:
    @pytest.mark.asyncio
    async def test_structure_returns_paragraph_count(self, backend, sample_doc):
        structure = await backend.read_structure(sample_doc)
        assert structure["paragraph_count"] == 7
        assert structure["type"] == "docx"

    @pytest.mark.asyncio
    async def test_unavailable_raises(self, backend, sample_doc):
        backend.set_available(False)
        with pytest.raises(BackendUnavailableError):
            await backend.read_structure(sample_doc)


# ---------------------------------------------------------------------------
# 原子操作
# ---------------------------------------------------------------------------

class TestAtomicOperations:
    @pytest.mark.asyncio
    async def test_read_paragraph(self, backend, sample_doc):
        text = await backend.read_paragraph(sample_doc, 0)
        assert text == "标题：项目报告"

    @pytest.mark.asyncio
    async def test_replace_paragraph(self, backend, sample_doc):
        await backend.replace_paragraph(sample_doc, 2, "本项目旨在降低成本。")
        text = await backend.read_paragraph(sample_doc, 2)
        assert text == "本项目旨在降低成本。"

    @pytest.mark.asyncio
    async def test_delete_paragraph(self, backend, sample_doc):
        await backend.delete_paragraph(sample_doc, 2)
        structure = await backend.read_structure(sample_doc)
        assert structure["paragraph_count"] == 6
        # 原来的第3段变成第2段
        text = await backend.read_paragraph(sample_doc, 2)
        assert text == "第二章 方法"

    @pytest.mark.asyncio
    async def test_append_paragraph(self, backend, sample_doc):
        await backend.append_paragraph(sample_doc, "第四章 结论")
        structure = await backend.read_structure(sample_doc)
        assert structure["paragraph_count"] == 8
        text = await backend.read_paragraph(sample_doc, 7)
        assert text == "第四章 结论"

    @pytest.mark.asyncio
    async def test_index_out_of_range(self, backend, sample_doc):
        with pytest.raises(IndexError):
            await backend.read_paragraph(sample_doc, 99)

    @pytest.mark.asyncio
    async def test_file_not_loaded(self, backend):
        with pytest.raises(FileNotFoundError):
            await backend.read_paragraph("/nonexistent.docx", 0)

    @pytest.mark.asyncio
    async def test_read_structure_file_not_loaded(self, backend):
        with pytest.raises(FileNotFoundError):
            await backend.read_structure("/nonexistent.docx")


# ---------------------------------------------------------------------------
# execute() 编辑操作
# ---------------------------------------------------------------------------

class TestExecuteEdit:
    @pytest.mark.asyncio
    async def test_replace_paragraph_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="replace_paragraph",
                    target={"paragraph_index": 2},
                    value="本项目旨在降低成本。",
                    description="替换第3段",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        assert result.output_file is not None
        assert result.output_file != sample_doc  # 副本路径不同
        assert "1 项操作" in result.summary
        # 原文件不变
        original_text = await backend.read_paragraph(sample_doc, 2)
        assert original_text == "本项目旨在提升系统性能。"
        # 副本已修改
        copy_text = await backend.read_paragraph(result.output_file, 2)
        assert copy_text == "本项目旨在降低成本。"

    @pytest.mark.asyncio
    async def test_delete_paragraph_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="delete_paragraph",
                    target={"paragraph_index": 4},
                    description="删除第5段",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        # 原文件段落数不变
        original_structure = await backend.read_structure(sample_doc)
        assert original_structure["paragraph_count"] == 7
        # 副本少了一段
        copy_structure = await backend.read_structure(result.output_file)
        assert copy_structure["paragraph_count"] == 6

    @pytest.mark.asyncio
    async def test_append_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="append_text",
                    target={},
                    value="附录：参考资料",
                    description="追加附录",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        copy_structure = await backend.read_structure(result.output_file)
        assert copy_structure["paragraph_count"] == 8

    @pytest.mark.asyncio
    async def test_replace_first_paragraph(self, backend, sample_doc):
        """paragraph_index=0 是合法索引，不应被误判为缺失"""
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="replace_paragraph",
                    target={"paragraph_index": 0},
                    value="新标题",
                    description="替换第1段",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        copy_text = await backend.read_paragraph(result.output_file, 0)
        assert copy_text == "新标题"

    @pytest.mark.asyncio
    async def test_delete_first_paragraph(self, backend, sample_doc):
        """paragraph_index=0 删除第一段"""
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="delete_paragraph",
                    target={"paragraph_index": 0},
                    description="删除第1段",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        copy_structure = await backend.read_structure(result.output_file)
        assert copy_structure["paragraph_count"] == 6

    @pytest.mark.asyncio
    async def test_multiple_operations(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="replace_paragraph",
                    target={"paragraph_index": 2},
                    value="新内容",
                    description="替换",
                ),
                DocumentOperation(
                    action="append_text",
                    target={},
                    value="追加段落",
                    description="追加",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        assert "2 项操作" in result.summary


# ---------------------------------------------------------------------------
# execute() 只读操作
# ---------------------------------------------------------------------------

class TestExecuteReadonly:
    @pytest.mark.asyncio
    async def test_review_returns_structure(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.REVIEW,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
        )
        result = await backend.execute(plan)
        assert result.success
        assert result.output_file is None  # 只读不生成文件

    @pytest.mark.asyncio
    async def test_summarize_returns_structure(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.SUMMARIZE,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
        )
        result = await backend.execute(plan)
        assert result.success


# ---------------------------------------------------------------------------
# execute() 错误处理
# ---------------------------------------------------------------------------

class TestExecuteErrors:
    @pytest.mark.asyncio
    async def test_unavailable_backend_raises(self, backend, sample_doc):
        backend.set_available(False)
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "不可用" in result.error

    @pytest.mark.asyncio
    async def test_unsupported_intent_returns_error(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.UNSUPPORTED,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            unsupported_reason="不支持",
        )
        result = await backend.execute(plan)
        assert not result.success

    @pytest.mark.asyncio
    async def test_invalid_action_returns_error(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(action="invalid_action", target={}),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "不支持的操作动作" in result.error

    @pytest.mark.asyncio
    async def test_missing_index_returns_error(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(action="replace_paragraph", target={}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "paragraph_index" in result.error

    @pytest.mark.asyncio
    async def test_partial_edit_no_output_file(self, backend, sample_doc):
        """多操作失败时不应返回可交付的 output_file"""
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="replace_paragraph",
                    target={"paragraph_index": 0},
                    value="新标题",
                    description="成功操作",
                ),
                DocumentOperation(
                    action="replace_paragraph",
                    target={"paragraph_index": 999},
                    value="失败",
                    description="索引越界",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert result.output_file is None
        assert result.verification.get("partial_output") is not None


# ---------------------------------------------------------------------------
# 文件类型 / 后端校验
# ---------------------------------------------------------------------------

class TestFileTypeValidation:
    """execute() 校验 plan.file_type 和 backend_required"""

    @pytest.mark.asyncio
    async def test_non_docx_plan_rejected(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(action="modify_cell", target={"cell": "A1"}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "xlsx" in result.error

    @pytest.mark.asyncio
    async def test_mismatched_backend_rejected(self, backend, sample_doc):
        # 用 model_construct 绕过 model 层校验，测试 adapter 层防御
        plan = DocumentPlan.model_construct(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "xlsx_mcp" in result.error

    @pytest.mark.asyncio
    async def test_missing_backend_rejected(self, backend, sample_doc):
        # 用 model_construct 绕过 model 层校验，测试 adapter 层防御
        plan = DocumentPlan.model_construct(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=None,
            operations=[
                DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "None" in result.error


# ---------------------------------------------------------------------------
# 输出副本唯一性
# ---------------------------------------------------------------------------

class TestOutputUniqueness:
    """连续执行生成不同副本路径"""

    @pytest.mark.asyncio
    async def test_consecutive_executions_different_paths(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="append_text",
                    target={},
                    value="追加",
                    description="追加",
                ),
            ],
        )
        result1 = await backend.execute(plan)
        result2 = await backend.execute(plan)
        assert result1.success and result2.success
        assert result1.output_file != result2.output_file
