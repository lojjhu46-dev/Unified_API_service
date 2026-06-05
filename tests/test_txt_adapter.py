"""Phase 5: TXT 后端 adapter 测试"""

import pytest
from app.documents.adapters.base import BackendUnavailableError
from app.documents.adapters.txt import MockTxtBackend
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperation,
    DocumentPlan,
    FileType,
)


@pytest.fixture
def backend():
    return MockTxtBackend()


@pytest.fixture
def sample_doc(backend):
    """加载模拟文本文件"""
    backend.load_file("/tmp/test.txt", [
        "第一行：标题",
        "第二行：正文内容",
        "第三行：实验体会的具体描述",
        "第四行：更多内容",
        "第五行：结尾",
    ])
    return "/tmp/test.txt"


# ---------------------------------------------------------------------------
# read_lines
# ---------------------------------------------------------------------------

class TestReadLines:
    @pytest.mark.asyncio
    async def test_read_all_lines(self, backend, sample_doc):
        lines = await backend.read_lines(sample_doc)
        assert len(lines) == 5
        assert lines[0] == "第一行：标题"

    @pytest.mark.asyncio
    async def test_unavailable_raises(self, backend, sample_doc):
        backend.set_available(False)
        with pytest.raises(BackendUnavailableError):
            await backend.read_lines(sample_doc)


# ---------------------------------------------------------------------------
# 原子操作
# ---------------------------------------------------------------------------

class TestAtomicOperations:
    @pytest.mark.asyncio
    async def test_replace_line(self, backend, sample_doc):
        await backend.replace_line(sample_doc, 3, "新内容")
        lines = await backend.read_lines(sample_doc)
        assert lines[2] == "新内容"

    @pytest.mark.asyncio
    async def test_delete_line(self, backend, sample_doc):
        await backend.delete_line(sample_doc, 3)
        lines = await backend.read_lines(sample_doc)
        assert len(lines) == 4
        assert lines[2] == "第四行：更多内容"

    @pytest.mark.asyncio
    async def test_delete_lines_range(self, backend, sample_doc):
        await backend.delete_lines(sample_doc, 2, 4)
        lines = await backend.read_lines(sample_doc)
        assert len(lines) == 2
        assert lines[0] == "第一行：标题"
        assert lines[1] == "第五行：结尾"

    @pytest.mark.asyncio
    async def test_append_lines(self, backend, sample_doc):
        await backend.append_lines(sample_doc, ["第六行", "第七行"])
        lines = await backend.read_lines(sample_doc)
        assert len(lines) == 7
        assert lines[5] == "第六行"

    @pytest.mark.asyncio
    async def test_replace_text(self, backend, sample_doc):
        count = await backend.replace_text(sample_doc, "实验体会", "研究心得")
        assert count == 1
        lines = await backend.read_lines(sample_doc)
        assert "研究心得" in lines[2]

    @pytest.mark.asyncio
    async def test_replace_text_no_match_raises(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(
                    action="replace_text",
                    target={"old": "不存在的文本"},
                    value="新文本",
                    description="替换不存在的文本",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "未找到匹配" in result.error
        assert result.output_file is None

    @pytest.mark.asyncio
    async def test_file_not_loaded(self, backend):
        with pytest.raises(FileNotFoundError):
            await backend.read_lines("/nonexistent.txt")

    @pytest.mark.asyncio
    async def test_line_out_of_range(self, backend, sample_doc):
        with pytest.raises(IndexError):
            await backend.replace_line(sample_doc, 99, "x")

    @pytest.mark.asyncio
    async def test_delete_lines_invalid_range(self, backend, sample_doc):
        with pytest.raises(IndexError):
            await backend.delete_lines(sample_doc, 4, 2)


# ---------------------------------------------------------------------------
# execute() 编辑操作
# ---------------------------------------------------------------------------

class TestExecuteEdit:
    @pytest.mark.asyncio
    async def test_replace_line_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(
                    action="replace_line",
                    target={"line": 3},
                    value="新内容",
                    description="替换第3行",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        assert result.output_file != sample_doc
        # 原文件不变
        original = await backend.read_lines(sample_doc)
        assert original[2] == "第三行：实验体会的具体描述"
        # 副本已修改
        copy_lines = await backend.read_lines(result.output_file)
        assert copy_lines[2] == "新内容"

    @pytest.mark.asyncio
    async def test_delete_line_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(
                    action="delete_line",
                    target={"line": 3},
                    description="删除第3行",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        copy_lines = await backend.read_lines(result.output_file)
        assert len(copy_lines) == 4

    @pytest.mark.asyncio
    async def test_delete_lines_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(
                    action="delete_lines",
                    target={"start": 2, "end": 4},
                    description="删除第2-4行",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        copy_lines = await backend.read_lines(result.output_file)
        assert len(copy_lines) == 2

    @pytest.mark.asyncio
    async def test_append_lines_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(
                    action="append_lines",
                    target={"lines": ["第六行", "第七行"]},
                    description="追加两行",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        copy_lines = await backend.read_lines(result.output_file)
        assert len(copy_lines) == 7

    @pytest.mark.asyncio
    async def test_replace_text_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(
                    action="replace_text",
                    target={"old": "实验体会"},
                    value="研究心得",
                    description="替换文本片段",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        copy_lines = await backend.read_lines(result.output_file)
        assert "研究心得" in copy_lines[2]

    @pytest.mark.asyncio
    async def test_multiple_operations(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(
                    action="replace_line",
                    target={"line": 1},
                    value="新标题",
                    description="替换标题",
                ),
                DocumentOperation(
                    action="append_lines",
                    target={"lines": ["附录"]},
                    description="追加附录",
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
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
        )
        result = await backend.execute(plan)
        assert result.success
        assert result.output_file is None
        assert result.verification["line_count"] == 5

    @pytest.mark.asyncio
    async def test_summarize_returns_line_count(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.SUMMARIZE,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
        )
        result = await backend.execute(plan)
        assert result.success
        assert "5 行" in result.summary


# ---------------------------------------------------------------------------
# execute() 错误处理
# ---------------------------------------------------------------------------

class TestExecuteErrors:
    @pytest.mark.asyncio
    async def test_unavailable_backend(self, backend, sample_doc):
        backend.set_available(False)
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(action="replace_line", target={"line": 1}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "不可用" in result.error

    @pytest.mark.asyncio
    async def test_non_txt_plan_rejected(self, backend, sample_doc):
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
        assert "docx" in result.error

    @pytest.mark.asyncio
    async def test_mismatched_backend_rejected(self, backend, sample_doc):
        plan = DocumentPlan.model_construct(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(action="replace_line", target={"line": 1}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "docx_mcp" in result.error

    @pytest.mark.asyncio
    async def test_none_backend_rejected(self, backend, sample_doc):
        """TXT 也要求 BackendType.TEXT_ADAPTER，None 不被接受"""
        plan = DocumentPlan.model_construct(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=None,
            operations=[
                DocumentOperation(action="replace_line", target={"line": 1}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "None" in result.error

    @pytest.mark.asyncio
    async def test_unsupported_intent(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.UNSUPPORTED,
            file_type=FileType.TXT,
            file_path=sample_doc,
            unsupported_reason="不支持",
        )
        result = await backend.execute(plan)
        assert not result.success

    @pytest.mark.asyncio
    async def test_invalid_action(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(action="invalid_action", target={}),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "不支持的操作动作" in result.error

    @pytest.mark.asyncio
    async def test_replace_line_missing_line(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(action="replace_line", target={}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "line" in result.error

    @pytest.mark.asyncio
    async def test_replace_text_missing_old(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(action="replace_text", target={}, value="新文本"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "old" in result.error

    @pytest.mark.asyncio
    async def test_append_lines_empty_input_raises(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(action="append_lines", target={}, description="空追加"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "缺少" in result.error

    @pytest.mark.asyncio
    async def test_partial_edit_no_output_file(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(
                    action="replace_line",
                    target={"line": 1},
                    value="新标题",
                    description="成功操作",
                ),
                DocumentOperation(
                    action="replace_line",
                    target={"line": 999},
                    value="失败",
                    description="行号越界",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert result.output_file is None
        assert result.verification.get("partial_output") is not None


# ---------------------------------------------------------------------------
# 输出副本唯一性
# ---------------------------------------------------------------------------

class TestOutputUniqueness:
    @pytest.mark.asyncio
    async def test_consecutive_executions_different_paths(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            file_path=sample_doc,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[
                DocumentOperation(
                    action="append_lines",
                    target={"lines": ["追加"]},
                    description="追加",
                ),
            ],
        )
        result1 = await backend.execute(plan)
        result2 = await backend.execute(plan)
        assert result1.success and result2.success
        assert result1.output_file != result2.output_file
