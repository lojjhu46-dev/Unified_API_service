"""Phase 4: XLSX 后端 adapter 测试"""

import pytest
from app.documents.adapters.base import BackendUnavailableError
from app.documents.adapters.xlsx import MockXlsxBackend, _column_name
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
    return MockXlsxBackend()


@pytest.fixture
def sample_doc(backend):
    """加载模拟 workbook"""
    backend.load_workbook("/tmp/test.xlsx", {
        "Sheet1": {
            "A1": "姓名", "B1": "部门", "C1": "工资",
            "A2": "张三", "B2": "技术部", "C2": "10000",
            "A3": "李四", "B3": "市场部", "C3": "8000",
        },
        "汇总": {
            "A1": "项目", "B1": "金额",
            "A2": "总计", "B2": "18000",
        },
    })
    return "/tmp/test.xlsx"


# ---------------------------------------------------------------------------
# read_structure
# ---------------------------------------------------------------------------

class TestReadStructure:
    @pytest.mark.asyncio
    async def test_structure_returns_sheets(self, backend, sample_doc):
        structure = await backend.read_structure(sample_doc)
        assert structure["type"] == "xlsx"
        assert "Sheet1" in structure["sheets"]
        assert "汇总" in structure["sheets"]

    @pytest.mark.asyncio
    async def test_structure_returns_headers(self, backend, sample_doc):
        structure = await backend.read_structure(sample_doc)
        sheet1_info = structure["sheets_info"]["Sheet1"]
        assert "姓名" in sheet1_info["headers"]
        assert sheet1_info["row_count"] == 3

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
    async def test_read_cell(self, backend, sample_doc):
        assert await backend.read_cell(sample_doc, "Sheet1", "A1") == "姓名"
        assert await backend.read_cell(sample_doc, "Sheet1", "C2") == "10000"

    @pytest.mark.asyncio
    async def test_modify_cell(self, backend, sample_doc):
        await backend.modify_cell(sample_doc, "Sheet1", "C2", "12000")
        assert await backend.read_cell(sample_doc, "Sheet1", "C2") == "12000"

    @pytest.mark.asyncio
    async def test_append_row(self, backend, sample_doc):
        await backend.append_row(sample_doc, "Sheet1", ["王五", "财务部", "9000"])
        assert await backend.read_cell(sample_doc, "Sheet1", "A4") == "王五"
        assert await backend.read_cell(sample_doc, "Sheet1", "C4") == "9000"

    @pytest.mark.asyncio
    async def test_delete_row(self, backend, sample_doc):
        await backend.delete_row(sample_doc, "Sheet1", 2)
        # 原第3行上移成为新第2行
        assert await backend.read_cell(sample_doc, "Sheet1", "A2") == "李四"
        assert await backend.read_cell(sample_doc, "Sheet1", "C2") == "8000"

    @pytest.mark.asyncio
    async def test_delete_row_zero_rejected(self, backend, sample_doc):
        with pytest.raises(ValueError, match="行号必须 >= 1"):
            await backend.delete_row(sample_doc, "Sheet1", 0)

    @pytest.mark.asyncio
    async def test_delete_row_negative_rejected(self, backend, sample_doc):
        with pytest.raises(ValueError, match="行号必须 >= 1"):
            await backend.delete_row(sample_doc, "Sheet1", -1)

    @pytest.mark.asyncio
    async def test_file_not_loaded(self, backend):
        with pytest.raises(FileNotFoundError):
            await backend.read_cell("/nonexistent.xlsx", "Sheet1", "A1")

    @pytest.mark.asyncio
    async def test_sheet_not_exist(self, backend, sample_doc):
        with pytest.raises(KeyError):
            await backend.read_cell(sample_doc, "不存在的表", "A1")

    @pytest.mark.asyncio
    async def test_append_row_27_columns(self, backend, sample_doc):
        """第27列应为 AA，不是 [ """
        values = [f"col{i}" for i in range(27)]
        await backend.append_row(sample_doc, "Sheet1", values)
        assert await backend.read_cell(sample_doc, "Sheet1", "A4") == "col0"
        assert await backend.read_cell(sample_doc, "Sheet1", "Z4") == "col25"
        assert await backend.read_cell(sample_doc, "Sheet1", "AA4") == "col26"


# ---------------------------------------------------------------------------
# _column_name 辅助函数
# ---------------------------------------------------------------------------

class TestColumnName:
    def test_single_letter(self):
        assert _column_name(0) == "A"
        assert _column_name(25) == "Z"

    def test_double_letter(self):
        assert _column_name(26) == "AA"
        assert _column_name(27) == "AB"
        assert _column_name(51) == "AZ"

    def test_triple_letter(self):
        assert _column_name(52) == "BA"
        assert _column_name(701) == "ZZ"
        assert _column_name(702) == "AAA"

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="列索引必须 >= 0"):
            _column_name(-1)


# ---------------------------------------------------------------------------
# 表头推断
# ---------------------------------------------------------------------------

class TestHeaderInference:
    @pytest.mark.asyncio
    async def test_a11_not_misidentified_as_header(self, backend):
        """A11 不应被误认为第一行表头"""
        backend.load_workbook("/tmp/header_test.xlsx", {
            "Sheet1": {
                "A1": "正确表头",
                "A11": "第11行数据",
                "B21": "第21行数据",
            },
        })
        structure = await backend.read_structure("/tmp/header_test.xlsx")
        headers = structure["sheets_info"]["Sheet1"]["headers"]
        assert headers == ["正确表头"]
        assert "第11行数据" not in headers


# ---------------------------------------------------------------------------
# execute() 编辑操作
# ---------------------------------------------------------------------------

class TestExecuteEdit:
    @pytest.mark.asyncio
    async def test_modify_cell_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="modify_cell",
                    target={"sheet": "Sheet1", "cell": "C2"},
                    value="12000",
                    description="涨工资",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        assert result.output_file != sample_doc
        # 原文件不变
        assert await backend.read_cell(sample_doc, "Sheet1", "C2") == "10000"
        # 副本已修改
        assert await backend.read_cell(result.output_file, "Sheet1", "C2") == "12000"

    @pytest.mark.asyncio
    async def test_append_row_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="append_row",
                    target={"sheet": "Sheet1", "values": ["王五", "财务部", "9000"]},
                    description="新增一行",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        assert await backend.read_cell(result.output_file, "Sheet1", "A4") == "王五"

    @pytest.mark.asyncio
    async def test_delete_row_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="delete_row",
                    target={"sheet": "Sheet1", "row": 2},
                    description="删除第2行",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        # 原文件不变
        assert await backend.read_cell(sample_doc, "Sheet1", "A2") == "张三"
        # 副本已删除
        assert await backend.read_cell(result.output_file, "Sheet1", "A2") == "李四"

    @pytest.mark.asyncio
    async def test_replace_range_via_execute(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="replace_range",
                    target={
                        "sheet": "Sheet1",
                        "replacements": [
                            {"cell": "B2", "value": "研发部"},
                            {"cell": "B3", "value": "销售部"},
                        ],
                    },
                    description="批量替换部门",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        assert await backend.read_cell(result.output_file, "Sheet1", "B2") == "研发部"
        assert await backend.read_cell(result.output_file, "Sheet1", "B3") == "销售部"

    @pytest.mark.asyncio
    async def test_multiple_operations(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="modify_cell",
                    target={"sheet": "Sheet1", "cell": "C2"},
                    value="15000",
                    description="修改工资",
                ),
                DocumentOperation(
                    action="append_row",
                    target={"sheet": "Sheet1", "values": ["王五", "财务部", "9000"]},
                    description="新增一行",
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
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
        )
        result = await backend.execute(plan)
        assert result.success
        assert result.output_file is None

    @pytest.mark.asyncio
    async def test_read_cell_operation(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EXTRACT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="read_cell",
                    target={"sheet": "Sheet1", "cell": "A1"},
                    description="读取表头",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert result.success
        assert "姓名" in result.summary


# ---------------------------------------------------------------------------
# execute() 错误处理
# ---------------------------------------------------------------------------

class TestExecuteErrors:
    @pytest.mark.asyncio
    async def test_unavailable_backend(self, backend, sample_doc):
        backend.set_available(False)
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(action="modify_cell", target={"sheet": "Sheet1", "cell": "A1"}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "不可用" in result.error

    @pytest.mark.asyncio
    async def test_non_xlsx_plan_rejected(self, backend, sample_doc):
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
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(action="modify_cell", target={"sheet": "Sheet1", "cell": "A1"}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "docx_mcp" in result.error

    @pytest.mark.asyncio
    async def test_missing_backend_rejected(self, backend, sample_doc):
        plan = DocumentPlan.model_construct(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=None,
            operations=[
                DocumentOperation(action="modify_cell", target={"sheet": "Sheet1", "cell": "A1"}, value="x"),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success

    @pytest.mark.asyncio
    async def test_unsupported_intent(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.UNSUPPORTED,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            unsupported_reason="不支持",
        )
        result = await backend.execute(plan)
        assert not result.success

    @pytest.mark.asyncio
    async def test_invalid_action(self, backend, sample_doc):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(action="invalid_action", target={}),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "不支持的操作动作" in result.error

    @pytest.mark.asyncio
    async def test_modify_cell_missing_sheet(self, backend, sample_doc):
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
        assert "sheet" in result.error

    @pytest.mark.asyncio
    async def test_replace_range_missing_cell_raises(self, backend, sample_doc):
        """replace_range 中任何 replacement 缺少 cell 都应报错"""
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="replace_range",
                    target={
                        "sheet": "Sheet1",
                        "replacements": [
                            {"cell": "A1", "value": "新值"},
                            {"value": "没有cell"},  # 缺少 cell
                        ],
                    },
                    description="批量替换",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "缺少 cell" in result.error

    @pytest.mark.asyncio
    async def test_replace_range_empty_replacements_raises(self, backend, sample_doc):
        """replace_range 的 replacements 不能为空"""
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="replace_range",
                    target={"sheet": "Sheet1", "replacements": []},
                    description="空批量替换",
                ),
            ],
        )
        result = await backend.execute(plan)
        assert not result.success
        assert "缺少 replacements" in result.error

    @pytest.mark.asyncio
    async def test_partial_edit_no_output_file(self, backend, sample_doc):
        """多操作失败时不应返回可交付的 output_file"""
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="modify_cell",
                    target={"sheet": "Sheet1", "cell": "A1"},
                    value="新值",
                    description="成功操作",
                ),
                DocumentOperation(
                    action="modify_cell",
                    target={"sheet": "不存在的表", "cell": "A1"},
                    value="失败",
                    description="工作表不存在",
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
            file_type=FileType.XLSX,
            file_path=sample_doc,
            backend_required=BackendType.XLSX_MCP,
            operations=[
                DocumentOperation(
                    action="append_row",
                    target={"sheet": "Sheet1", "values": ["测试"]},
                    description="追加",
                ),
            ],
        )
        result1 = await backend.execute(plan)
        result2 = await backend.execute(plan)
        assert result1.success and result2.success
        assert result1.output_file != result2.output_file
