"""Phase 6: PDF 只读后端 adapter 测试"""

import pytest
from app.documents.adapters.base import BackendUnavailableError
from app.documents.adapters.pdf import MockPdfBackend
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperation,
    DocumentPlan,
    FileType,
)


@pytest.fixture
def backend():
    return MockPdfBackend()


@pytest.fixture
def sample_doc(backend):
    """加载模拟 PDF"""
    backend.load_document("/tmp/test.pdf", [
        "第一页：封面\n项目报告",
        "第二页：目录\n第一章 背景\n第二章 方法",
        "第三页：正文\n本项目旨在提升系统性能。",
        "第四页：结论\n性能提升50%。",
    ])
    return "/tmp/test.pdf"


# ---------------------------------------------------------------------------
# read_structure
# ---------------------------------------------------------------------------

class TestReadStructure:
    @pytest.mark.asyncio
    async def test_structure_returns_page_count(self, backend, sample_doc):
        structure = await backend.read_structure(sample_doc)
        assert structure["type"] == "pdf"
        assert structure["page_count"] == 4

    @pytest.mark.asyncio
    async def test_unavailable_raises(self, backend, sample_doc):
        backend.set_available(False)
        with pytest.raises(BackendUnavailableError):
            await backend.read_structure(sample_doc)


# ---------------------------------------------------------------------------
# 只读原子操作
# ---------------------------------------------------------------------------

class TestReadOnlyOperations:
    @pytest.mark.asyncio
    async def test_read_page(self, backend, sample_doc):
        text = await backend.read_page(sample_doc, 1)
        assert "封面" in text

    @pytest.mark.asyncio
    async def test_extract_text_all(self, backend, sample_doc):
        text = await backend.extract_text(sample_doc)
        assert "封面" in text
        assert "结论" in text

    @pytest.mark.asyncio
    async def test_extract_text_range(self, backend, sample_doc):
        text = await backend.extract_text(sample_doc, 2, 3)
        assert "目录" in text
        assert "正文" in text
        assert "封面" not in text

    @pytest.mark.asyncio
    async def test_page_out_of_range(self, backend, sample_doc):
        with pytest.raises(IndexError):
            await backend.read_page(sample_doc, 99)

    @pytest.mark.asyncio
    async def test_file_not_loaded(self, backend):
        with pytest.raises(FileNotFoundError):
            await backend.read_page("/nonexistent.pdf", 1)


# ---------------------------------------------------------------------------
# execute() 编辑被拒绝
# ---------------------------------------------------------------------------

class TestEditRejected:
    @pytest.mark.asyncio
    async def test_edit_intent_rejected(self, backend, sample_doc):
        """PDF 编辑请求必须被拒绝（adapter 层 defense-in-depth）"""
        plan = DocumentPlan.model_construct(
            intent=DocumentIntent.EDIT,
            file_type=FileType.PDF,
            file_path=sample_doc,
            backend_required=BackendType.PDF_READER,
            operations=[
                DocumentOperation(
                    action="replace_text",
                    target={"page": 1},
                    value="新内容",
                    description="尝试编辑",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "不支持编辑" in result.error
        assert result.output_file is None

    @pytest.mark.asyncio
    async def test_unsupported_intent(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.UNSUPPORTED,
            file_type=FileType.PDF,
            file_path=sample_doc,
            unsupported_reason="不支持",
        )
        result = await backend.execute(plan)
        assert not result.success


# ---------------------------------------------------------------------------
# execute() 只读操作
# ---------------------------------------------------------------------------

class TestExecuteReadonly:
    @pytest.mark.asyncio
    async def test_review_returns_structure(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.REVIEW,
            file_type=FileType.PDF,
            file_path=sample_doc,
            backend_required=BackendType.PDF_READER,
        )
        result = await backend.execute(plan)
        assert result.success
        assert result.output_file is None
        assert result.verification["structure"]["page_count"] == 4

    @pytest.mark.asyncio
    async def test_summarize_returns_page_count(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.SUMMARIZE,
            file_type=FileType.PDF,
            file_path=sample_doc,
            backend_required=BackendType.PDF_READER,
        )
        result = await backend.execute(plan)
        assert result.success
        assert "4 页" in result.summary

    @pytest.mark.asyncio
    async def test_extract_text_operation(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EXTRACT,
            file_type=FileType.PDF,
            file_path=sample_doc,
            backend_required=BackendType.PDF_READER,
            operations=[
                DocumentOperation(
                    action="extract_text",
                    target={"page_range": [1, 2]},
                    description="提取前两页",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        assert "封面" in result.summary

    @pytest.mark.asyncio
    async def test_read_page_operation(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EXTRACT,
            file_type=FileType.PDF,
            file_path=sample_doc,
            backend_required=BackendType.PDF_READER,
            operations=[
                DocumentOperation(
                    action="read_page",
                    target={"page": 3},
                    description="读取第3页",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        assert "正文" in result.summary


# ---------------------------------------------------------------------------
# execute() 错误处理
# ---------------------------------------------------------------------------

class TestExecuteErrors:
    @pytest.mark.asyncio
    async def test_unavailable_backend(self, backend, sample_doc):
        backend.set_available(False)
        plan = DocumentPlan(
            intent=DocumentIntent.REVIEW,
            file_type=FileType.PDF,
            file_path=sample_doc,
            backend_required=BackendType.PDF_READER,
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "不可用" in result.error

    @pytest.mark.asyncio
    async def test_non_pdf_plan_rejected(self, backend, sample_doc):
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
            intent=DocumentIntent.REVIEW,
            file_type=FileType.PDF,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "docx_mcp" in result.error
