"""Phase 1: 统一文档任务模型测试"""

import pytest
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperation,
    DocumentOperationResult,
    DocumentPlan,
    FileType,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# DocumentPlan 构造与序列化
# ---------------------------------------------------------------------------

class TestDocumentPlan:
    """DocumentPlan 基本构造与属性"""

    def test_edit_plan_is_actionable(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            backend_required=BackendType.DOCX_MCP,
            operations=[
                DocumentOperation(
                    action="replace_paragraph",
                    target={"paragraph_index": 3},
                    value="新内容",
                    description="替换第4段",
                ),
            ],
        )
        assert plan.is_actionable
        assert not plan.needs_confirmation

    def test_review_plan_is_actionable_without_operations(self):
        plan = DocumentPlan(
            intent=DocumentIntent.REVIEW,
            file_type=FileType.PDF,
            backend_required=BackendType.PDF_READER,
        )
        assert plan.is_actionable
        assert plan.intent == DocumentIntent.REVIEW

    def test_extract_plan_is_actionable_without_operations(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EXTRACT,
            file_type=FileType.PDF,
            backend_required=BackendType.PDF_READER,
        )
        assert plan.is_actionable

    def test_summarize_plan_is_actionable_without_operations(self):
        plan = DocumentPlan(
            intent=DocumentIntent.SUMMARIZE,
            file_type=FileType.DOCX,
            backend_required=BackendType.DOCX_MCP,
        )
        assert plan.is_actionable

    def test_edit_plan_not_actionable_without_operations(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            backend_required=BackendType.TEXT_ADAPTER,
        )
        assert not plan.is_actionable

    def test_unsupported_plan(self):
        plan = DocumentPlan(
            intent=DocumentIntent.UNSUPPORTED,
            file_type=FileType.PDF,
            unsupported_reason="PDF 不支持编辑",
        )
        assert not plan.is_actionable
        assert plan.backend_required is None

    def test_clarification_plan(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            backend_required=BackendType.TEXT_ADAPTER,
            clarification_question="请指定要删除的行号范围",
        )
        assert not plan.is_actionable
        assert plan.clarification_question == "请指定要删除的行号范围"

    def test_high_risk_needs_confirmation(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            backend_required=BackendType.XLSX_MCP,
            risk_level=RiskLevel.HIGH,
            operations=[
                DocumentOperation(action="delete_row", target={"row": 5}),
            ],
        )
        assert plan.needs_confirmation

    def test_explicit_requires_confirmation(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            backend_required=BackendType.DOCX_MCP,
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=True,
            operations=[
                DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="x"),
            ],
        )
        assert plan.needs_confirmation

    def test_serialization_roundtrip(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EXTRACT,
            file_type=FileType.PDF,
            backend_required=BackendType.PDF_READER,
            operations=[
                DocumentOperation(
                    action="extract_text",
                    target={"page_range": [1, 3]},
                    description="提取前3页文本",
                ),
            ],
        )
        data = plan.model_dump()
        restored = DocumentPlan.model_validate(data)
        assert restored.intent == plan.intent
        assert restored.file_type == plan.file_type
        assert len(restored.operations) == 1
        assert restored.operations[0].action == "extract_text"


# ---------------------------------------------------------------------------
# DocumentOperation
# ---------------------------------------------------------------------------

class TestDocumentOperation:
    """单条操作"""

    def test_replace_operation(self):
        op = DocumentOperation(
            action="replace_paragraph",
            target={"paragraph_index": 2},
            value="替换后的文本",
            description="替换第3段",
        )
        assert op.action == "replace_paragraph"
        assert op.target["paragraph_index"] == 2
        assert op.value == "替换后的文本"

    def test_delete_operation_no_value(self):
        op = DocumentOperation(
            action="delete_paragraph",
            target={"paragraph_index": 4},
            description="删除第5段",
        )
        assert op.value is None

    def test_cell_operation(self):
        op = DocumentOperation(
            action="modify_cell",
            target={"sheet": "Sheet1", "cell": "B3"},
            value="新值",
        )
        assert op.target["cell"] == "B3"


# ---------------------------------------------------------------------------
# DocumentOperationResult
# ---------------------------------------------------------------------------

class TestDocumentOperationResult:
    """执行结果"""

    def test_success_result(self):
        result = DocumentOperationResult(
            success=True,
            output_file="/tmp/output.docx",
            summary="已替换3个段落",
            verification={"original_unchanged": True, "target_modified": True},
        )
        assert result.success
        assert result.error is None
        assert len(result.warnings) == 0

    def test_failure_result(self):
        result = DocumentOperationResult(
            success=False,
            error="MCP 后端不可用",
        )
        assert not result.success
        assert result.output_file is None

    def test_result_with_warnings(self):
        result = DocumentOperationResult(
            success=True,
            output_file="/tmp/output.xlsx",
            summary="已修改单元格",
            warnings=["公式引用的单元格被修改，可能影响计算结果"],
        )
        assert result.success
        assert len(result.warnings) == 1

    def test_result_serialization(self):
        result = DocumentOperationResult(
            success=True,
            output_file="/tmp/output.txt",
            summary="完成",
        )
        data = result.model_dump()
        assert data["success"] is True
        assert data["output_file"] == "/tmp/output.txt"


# ---------------------------------------------------------------------------
# 规则验证
# ---------------------------------------------------------------------------

class TestPlanRules:
    """Phase 1 规则：模型层强制校验 + 非法输入拒绝"""

    # ---- 合规样例 ----

    def test_pdf_extract_uses_pdf_reader(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EXTRACT,
            file_type=FileType.PDF,
            backend_required=BackendType.PDF_READER,
            operations=[DocumentOperation(action="extract_text", target={"page_range": [1, 5]})],
        )
        assert plan.backend_required == BackendType.PDF_READER

    def test_docx_edit_requires_mcp_backend(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.DOCX,
            backend_required=BackendType.DOCX_MCP,
            operations=[DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="x")],
        )
        assert plan.backend_required == BackendType.DOCX_MCP

    def test_xlsx_edit_requires_mcp_backend(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.XLSX,
            backend_required=BackendType.XLSX_MCP,
            operations=[DocumentOperation(action="modify_cell", target={"cell": "A1"}, value="x")],
        )
        assert plan.backend_required == BackendType.XLSX_MCP

    def test_txt_edit_uses_text_adapter(self):
        plan = DocumentPlan(
            intent=DocumentIntent.EDIT,
            file_type=FileType.TXT,
            backend_required=BackendType.TEXT_ADAPTER,
            operations=[DocumentOperation(action="replace_line", target={"line": 1}, value="新内容")],
        )
        assert plan.backend_required == BackendType.TEXT_ADAPTER

    # ---- 非法输入必须被拒绝（P1）----

    def test_pdf_edit_rejected(self):
        """PDF 编辑请求必须被模型层拒绝"""
        with pytest.raises(ValueError, match="PDF 不支持编辑"):
            DocumentPlan(
                intent=DocumentIntent.EDIT,
                file_type=FileType.PDF,
                backend_required=BackendType.PDF_READER,
                operations=[DocumentOperation(action="replace_text", target={"page": 1}, value="x")],
            )

    def test_docx_edit_with_wrong_backend_rejected(self):
        """DOCX 编辑使用错误后端必须被拒绝"""
        with pytest.raises(ValueError, match="docx 文件必须使用 docx_mcp"):
            DocumentPlan(
                intent=DocumentIntent.EDIT,
                file_type=FileType.DOCX,
                backend_required=BackendType.TEXT_ADAPTER,
                operations=[DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="x")],
            )

    def test_xlsx_edit_with_wrong_backend_rejected(self):
        """XLSX 编辑使用错误后端必须被拒绝"""
        with pytest.raises(ValueError, match="xlsx 文件必须使用 xlsx_mcp"):
            DocumentPlan(
                intent=DocumentIntent.EDIT,
                file_type=FileType.XLSX,
                backend_required=BackendType.DOCX_MCP,
                operations=[DocumentOperation(action="modify_cell", target={"cell": "A1"}, value="x")],
            )

    def test_txt_edit_with_wrong_backend_rejected(self):
        """TXT 编辑使用错误后端必须被拒绝"""
        with pytest.raises(ValueError, match="txt 文件必须使用 text_adapter"):
            DocumentPlan(
                intent=DocumentIntent.EDIT,
                file_type=FileType.TXT,
                backend_required=BackendType.DOCX_MCP,
                operations=[DocumentOperation(action="replace_line", target={"line": 1}, value="x")],
            )

    def test_docx_edit_without_backend_rejected(self):
        """DOCX 编辑未声明后端必须被拒绝"""
        with pytest.raises(ValueError, match="docx 文件必须使用 docx_mcp"):
            DocumentPlan(
                intent=DocumentIntent.EDIT,
                file_type=FileType.DOCX,
                operations=[DocumentOperation(action="replace_paragraph", target={"paragraph_index": 0}, value="x")],
            )

    def test_pdf_review_with_wrong_backend_rejected(self):
        """PDF 只读任务也必须使用 pdf_reader 后端"""
        with pytest.raises(ValueError, match="pdf 文件必须使用 pdf_reader"):
            DocumentPlan(
                intent=DocumentIntent.REVIEW,
                file_type=FileType.PDF,
                backend_required=BackendType.DOCX_MCP,
            )

    def test_pdf_summarize_with_wrong_backend_rejected(self):
        """PDF 总结任务不能被错误路由到 XLSX MCP"""
        with pytest.raises(ValueError, match="pdf 文件必须使用 pdf_reader"):
            DocumentPlan(
                intent=DocumentIntent.SUMMARIZE,
                file_type=FileType.PDF,
                backend_required=BackendType.XLSX_MCP,
            )

    def test_docx_extract_with_wrong_backend_rejected(self):
        """DOCX 只读任务也必须使用 docx_mcp 后端"""
        with pytest.raises(ValueError, match="docx 文件必须使用 docx_mcp"):
            DocumentPlan(
                intent=DocumentIntent.EXTRACT,
                file_type=FileType.DOCX,
                backend_required=BackendType.PDF_READER,
            )

    def test_xlsx_review_with_wrong_backend_rejected(self):
        """XLSX 审阅任务也必须使用 xlsx_mcp 后端"""
        with pytest.raises(ValueError, match="xlsx 文件必须使用 xlsx_mcp"):
            DocumentPlan(
                intent=DocumentIntent.REVIEW,
                file_type=FileType.XLSX,
                backend_required=BackendType.PDF_READER,
            )

    def test_unsupported_bypasses_validation(self):
        """UNSUPPORTED 意图跳过校验（用于返回拒绝说明）"""
        plan = DocumentPlan(
            intent=DocumentIntent.UNSUPPORTED,
            file_type=FileType.PDF,
            unsupported_reason="PDF 不支持编辑",
        )
        assert plan.intent == DocumentIntent.UNSUPPORTED


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    """枚举值完整性"""

    def test_intent_values(self):
        assert set(DocumentIntent) == {
            DocumentIntent.EDIT,
            DocumentIntent.REVIEW,
            DocumentIntent.EXTRACT,
            DocumentIntent.SUMMARIZE,
            DocumentIntent.UNSUPPORTED,
        }

    def test_file_type_values(self):
        assert set(FileType) == {
            FileType.DOCX,
            FileType.XLSX,
            FileType.TXT,
            FileType.PDF,
        }

    def test_backend_values(self):
        assert set(BackendType) == {
            BackendType.DOCX_MCP,
            BackendType.XLSX_MCP,
            BackendType.TEXT_ADAPTER,
            BackendType.PDF_READER,
        }

    def test_risk_level_values(self):
        assert set(RiskLevel) == {
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
        }
