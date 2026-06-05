"""Phase 2: 规划智能体测试"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from app.documents.models import (
    BackendType,
    DocumentIntent,
    FileType,
    RiskLevel,
)
from app.documents.planner import DocumentPlanningAgent, build_structure_brief


planner = DocumentPlanningAgent()


# ---------------------------------------------------------------------------
# build_structure_brief
# ---------------------------------------------------------------------------

class TestBuildStructureBrief:
    """文件结构摘要"""

    def test_docx_structure(self):
        brief = build_structure_brief({
            "type": "docx",
            "headings": ["第一章", "1.1 背景", "1.2 目标"],
            "paragraph_count": 42,
            "tables": ["表1"],
        })
        assert "第一章" in brief
        assert "42" in brief
        assert "1 个表格" in brief

    def test_xlsx_structure(self):
        brief = build_structure_brief({
            "type": "xlsx",
            "sheets": ["Sheet1", "汇总"],
            "headers": ["姓名", "部门", "工资"],
            "row_count": 100,
        })
        assert "Sheet1" in brief
        assert "姓名" in brief
        assert "100" in brief

    def test_txt_structure(self):
        brief = build_structure_brief({
            "type": "txt",
            "line_count": 50,
            "preview": "第一行内容",
        })
        assert "50" in brief
        assert "第一行内容" in brief

    def test_pdf_structure(self):
        brief = build_structure_brief({
            "type": "pdf",
            "page_count": 10,
            "toc": ["目录", "第一章"],
        })
        assert "10" in brief
        assert "目录" in brief

    def test_empty_structure(self):
        assert "无结构信息" in build_structure_brief({})

    def test_none_structure(self):
        assert "无结构信息" in build_structure_brief(None)


# ---------------------------------------------------------------------------
# 硬规则：不依赖 LLM
# ---------------------------------------------------------------------------

class TestHardRules:
    """硬规则：PDF 编辑拒绝等"""

    @pytest.mark.asyncio
    async def test_pdf_edit_rejected_without_llm(self):
        """PDF 编辑请求应直接拒绝，不调用 LLM"""
        plan = await planner.plan(
            user_command="把第三段的内容改成新文本",
            file_type=FileType.PDF,
        )
        assert plan.intent == DocumentIntent.UNSUPPORTED
        assert plan.is_actionable is False
        assert "PDF" in (plan.unsupported_reason or "")

    @pytest.mark.asyncio
    async def test_pdf_review_passes_through(self):
        """PDF 审阅不被拒绝（需要 LLM 返回结果）"""
        mock_plan_data = {
            "intent": "review",
            "operations": [],
            "risk_level": "low",
            "requires_confirmation": False,
            "clarification_question": None,
            "unsupported_reason": None,
        }
        with patch.object(
            planner, "_extract_json", return_value=mock_plan_data,
        ):
            plan = await planner.plan(
                user_command="审阅这个PDF文档",
                file_type=FileType.PDF,
            )
        assert plan.intent == DocumentIntent.REVIEW
        assert plan.backend_required == BackendType.PDF_READER


# ---------------------------------------------------------------------------
# LLM 调用失败降级
# ---------------------------------------------------------------------------

class TestLLMFailureFallback:
    """LLM 调用失败时的降级处理"""

    @pytest.mark.asyncio
    async def test_llm_failure_returns_unsupported(self):
        with patch(
            "app.documents.planner.llm_gateway.generate",
            new=AsyncMock(side_effect=Exception("API 超时")),
        ):
            plan = await planner.plan(
                user_command="删除第三段",
                file_type=FileType.DOCX,
            )
        assert plan.intent == DocumentIntent.UNSUPPORTED
        assert "失败" in (plan.unsupported_reason or "")


# ---------------------------------------------------------------------------
# LLM 输出解析
# ---------------------------------------------------------------------------

class TestParseLLMOutput:
    """LLM 输出解析与后处理"""

    @pytest.mark.asyncio
    async def test_valid_edit_plan(self):
        mock_data = {
            "intent": "edit",
            "operations": [
                {
                    "action": "replace_paragraph",
                    "target": {"paragraph_index": 2},
                    "value": "新内容",
                    "description": "替换第3段",
                },
            ],
            "risk_level": "low",
            "requires_confirmation": False,
            "clarification_question": None,
            "unsupported_reason": None,
        }
        with patch(
            "app.documents.planner.llm_gateway.generate",
            new=AsyncMock(return_value=json.dumps(mock_data)),
        ):
            plan = await planner.plan(
                user_command="把第三段改成新内容",
                file_type=FileType.DOCX,
            )
        assert plan.intent == DocumentIntent.EDIT
        assert plan.backend_required == BackendType.DOCX_MCP
        assert len(plan.operations) == 1
        assert plan.operations[0].action == "replace_paragraph"

    @pytest.mark.asyncio
    async def test_high_risk_action_forced_confirmation(self):
        """删除操作强制高风险 + 需确认"""
        mock_data = {
            "intent": "edit",
            "operations": [
                {
                    "action": "replace_paragraph",
                    "target": {"paragraph_index": 2},
                    "value": None,
                    "description": "删除第3段",
                },
            ],
            "risk_level": "low",
            "requires_confirmation": False,
            "clarification_question": None,
            "unsupported_reason": None,
        }
        with patch(
            "app.documents.planner.llm_gateway.generate",
            new=AsyncMock(return_value=json.dumps(mock_data)),
        ):
            plan = await planner.plan(
                user_command="删除第三段",
                file_type=FileType.DOCX,
            )
        assert plan.risk_level == RiskLevel.HIGH
        assert plan.needs_confirmation

    @pytest.mark.asyncio
    async def test_vague_target_triggers_clarification(self):
        """模糊定位触发澄清"""
        mock_data = {
            "intent": "edit",
            "operations": [
                {
                    "action": "delete_paragraph",
                    "target": {},
                    "value": None,
                    "description": "删除相关内容",
                },
            ],
            "risk_level": "medium",
            "requires_confirmation": False,
            "clarification_question": None,
            "unsupported_reason": None,
        }
        with patch(
            "app.documents.planner.llm_gateway.generate",
            new=AsyncMock(return_value=json.dumps(mock_data)),
        ):
            plan = await planner.plan(
                user_command="删除实验体会的内容",
                file_type=FileType.DOCX,
            )
        assert plan.clarification_question is not None
        assert len(plan.operations) == 0

    @pytest.mark.asyncio
    async def test_malformed_json_returns_unsupported(self):
        """LLM 输出无法解析时返回 unsupported"""
        with patch(
            "app.documents.planner.llm_gateway.generate",
            new=AsyncMock(return_value="这不是JSON"),
        ):
            plan = await planner.plan(
                user_command="总结文档",
                file_type=FileType.TXT,
            )
        assert plan.intent == DocumentIntent.UNSUPPORTED
        assert "格式错误" in (plan.unsupported_reason or "")

    @pytest.mark.asyncio
    async def test_json_in_code_block(self):
        """LLM 输出包含在 ```json 代码块中"""
        mock_data = {
            "intent": "summarize",
            "operations": [],
            "risk_level": "low",
            "requires_confirmation": False,
            "clarification_question": None,
            "unsupported_reason": None,
        }
        wrapped = f"```json\n{json.dumps(mock_data, ensure_ascii=False)}\n```"
        with patch(
            "app.documents.planner.llm_gateway.generate",
            new=AsyncMock(return_value=wrapped),
        ):
            plan = await planner.plan(
                user_command="总结文档",
                file_type=FileType.TXT,
            )
        assert plan.intent == DocumentIntent.SUMMARIZE

    @pytest.mark.asyncio
    async def test_xlsx_plan_uses_xlsx_mcp(self):
        mock_data = {
            "intent": "edit",
            "operations": [
                {
                    "action": "modify_cell",
                    "target": {"sheet": "Sheet1", "cell": "B3"},
                    "value": "新值",
                    "description": "修改 B3",
                },
            ],
            "risk_level": "low",
            "requires_confirmation": False,
            "clarification_question": None,
            "unsupported_reason": None,
        }
        with patch(
            "app.documents.planner.llm_gateway.generate",
            new=AsyncMock(return_value=json.dumps(mock_data)),
        ):
            plan = await planner.plan(
                user_command="把 Sheet1 的 B3 改成新值",
                file_type=FileType.XLSX,
            )
        assert plan.backend_required == BackendType.XLSX_MCP


# ---------------------------------------------------------------------------
# JSON 提取
# ---------------------------------------------------------------------------

class TestExtractJson:
    """_extract_json 辅助方法"""

    def test_direct_json(self):
        data = {"intent": "edit"}
        assert planner._extract_json(json.dumps(data)) == data

    def test_json_in_code_block(self):
        data = {"intent": "review"}
        raw = f"好的，这是方案：\n```json\n{json.dumps(data)}\n```"
        assert planner._extract_json(raw) == data

    def test_json_embedded_in_text(self):
        data = {"intent": "extract"}
        raw = f"根据分析，方案如下：{json.dumps(data)}以上是方案。"
        assert planner._extract_json(raw) == data

    def test_no_json_returns_none(self):
        assert planner._extract_json("完全没有JSON内容") is None

    def test_invalid_json_returns_none(self):
        assert planner._extract_json("{invalid json}") is None
