"""Phase 2: 规划智能体

接收用户命令 + 文件类型 + 文件结构信息，输出 DocumentPlan。
规则校验优先于 LLM 调用：PDF 编辑、高风险操作等硬规则不依赖 LLM 判断。
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.documents.models import (
    BACKEND_BY_FILE_TYPE,
    BackendType,
    DocumentIntent,
    DocumentOperation,
    DocumentPlan,
    FileType,
    RiskLevel,
)
from app.llm.gateway import llm_gateway
from app.observability.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 高风险动作关键词
# ---------------------------------------------------------------------------

_HIGH_RISK_ACTIONS = frozenset({
    "delete", "remove", "clear", "replace_all", "bulk_replace",
    "删除", "清空", "批量替换", "替换全部", "移除",
})

_VAGUE_TARGET_MARKERS = frozenset({
    "相关内容", "相关部分", "有关内容", "那部分", "那一段",
    "实验体会", "随便哪里", "差不多的", "类似内容",
})

# ---------------------------------------------------------------------------
# 文件结构摘要（供 LLM 理解文档结构）
# ---------------------------------------------------------------------------

def build_structure_brief(structure: dict[str, Any]) -> str:
    """将文件结构信息压缩为 LLM 可消费的文本摘要。"""
    if not structure:
        return "（无结构信息）"

    parts: list[str] = []
    doc_type = structure.get("type", "")
    if doc_type == "docx":
        headings = structure.get("headings", [])
        if headings:
            parts.append("标题结构：" + " > ".join(headings[:10]))
        para_count = structure.get("paragraph_count", 0)
        if para_count:
            parts.append(f"共 {para_count} 个段落")

        # 段落详情（新增）
        paragraphs = structure.get("paragraphs", [])
        if paragraphs:
            parts.append("段落：")
            for p in paragraphs[:50]:  # 最多显示前 50 段
                index = p.get("index", 0)
                text = p.get("text", "")
                # 截断过长的文本显示
                if len(text) > 100:
                    text = text[:100] + "..."
                parts.append(f"  [{index}] {text}")

        # tables 兼容 list 和 int 两种格式
        tables = structure.get("tables", [])
        if isinstance(tables, list):
            if tables:
                parts.append(f"含 {len(tables)} 个表格")
                for t in tables[:5]:  # 最多显示前 5 个表格
                    if not isinstance(t, dict):
                        continue
                    idx = t.get("index", 0)
                    rows = t.get("rows", 0)
                    cols = t.get("cols", 0)
                    parts.append(f"  表格[{idx}]: {rows}行 x {cols}列")
                    cells = t.get("cells", [])
                    if isinstance(cells, list) and cells:
                        for cell in cells[:30]:
                            if not isinstance(cell, dict):
                                continue
                            text = (cell.get("text") or "").strip()
                            if not text:
                                continue
                            if len(text) > 400:
                                text = text[:400] + "..."
                            row = cell.get("row", 0)
                            col = cell.get("col", 0)
                            parts.append(f"    表格[{idx}] R{row}C{col}: {text}")
        elif isinstance(tables, int) and tables > 0:
            parts.append(f"含 {tables} 个表格")

    elif doc_type == "xlsx":
        sheets = structure.get("sheets", [])
        if sheets:
            parts.append("工作表：" + ", ".join(sheets[:10]))

        # 优先读取 sheets_info，同时兼容顶层 headers/row_count
        sheets_info = structure.get("sheets_info", {})
        if sheets_info:
            for sheet_name, info in list(sheets_info.items())[:5]:  # 最多显示前 5 个 sheet
                headers = info.get("headers", [])
                row_count = info.get("row_count", 0)
                if headers:
                    parts.append(f"[{sheet_name}] 表头：" + ", ".join(headers[:15]))
                if row_count:
                    parts.append(f"[{sheet_name}] 约 {row_count} 行数据")
        else:
            # 兼容顶层 headers/row_count
            headers = structure.get("headers", [])
            if headers:
                parts.append("表头：" + ", ".join(headers[:15]))
            row_count = structure.get("row_count", 0)
            if row_count:
                parts.append(f"约 {row_count} 行数据")

    elif doc_type == "txt":
        line_count = structure.get("line_count", 0)
        if line_count:
            parts.append(f"共 {line_count} 行")
        preview = structure.get("preview", "")
        if preview:
            parts.append(f"开头预览：{preview[:200]}")
    elif doc_type == "pdf":
        page_count = structure.get("page_count", 0)
        if page_count:
            parts.append(f"共 {page_count} 页")
        toc = structure.get("toc", [])
        if toc:
            parts.append("目录：" + " > ".join(toc[:10]))

    return "\n".join(parts) if parts else "（无结构信息）"


# ---------------------------------------------------------------------------
# LLM 提示词
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """你是文档编辑规划助手。根据用户命令和文档结构，输出一个 JSON 格式的编辑方案。

输出格式（严格 JSON，不要输出其他内容）：
{
  "intent": "edit|review|extract|summarize|unsupported",
  "operations": [
    {
      "action": "操作动作名",
      "target": {"定位键": "定位值"},
      "value": "写入值或null",
      "description": "操作说明"
    }
  ],
  "risk_level": "low|medium|high",
  "requires_confirmation": false,
  "clarification_question": null,
  "unsupported_reason": null
}

规则：
1. 如果用户命令定位模糊（如"删除实验体会的内容"但无法确定具体段落），必须将 clarification_question 设为具体问题，operations 留空。
2. 删除、替换、清空、批量修改操作的 risk_level 设为 high，requires_confirmation 设为 true。
3. intent 为 unsupported 时，说明原因。
4. operations 中的 target 必须使用文档结构中的具体定位信息（段落号、单元格引用、行号等）。
5. DOCX 表格单元格内容可以用于定位；如果目标位于表格单元格，target 必须包含 table_index、row、col。
6. 删除/清空 DOCX 表格单元格内容时，action 使用 clear_table_cell；替换 DOCX 表格单元格内容时，action 使用 replace_table_cell。
7. 只输出 JSON，不要输出任何解释。"""


def _build_planner_prompt(
    user_command: str,
    file_type: FileType,
    structure_brief: str,
) -> str:
    return (
        f"文件类型：{file_type.value}\n\n"
        f"文档结构：\n{structure_brief}\n\n"
        f"用户命令：{user_command}"
    )


# ---------------------------------------------------------------------------
# 规划智能体
# ---------------------------------------------------------------------------

class DocumentPlanningAgent:
    """规划智能体：分析用户命令，输出可执行的 DocumentPlan。"""

    async def plan(
        self,
        user_command: str,
        file_type: FileType,
        structure: dict[str, Any] | None = None,
    ) -> DocumentPlan:
        """分析用户命令并返回 DocumentPlan。

        Args:
            user_command: 用户自然语言命令
            file_type: 目标文件类型
            structure: 文件结构信息（段落列表、工作表名、页数等）

        Returns:
            DocumentPlan：可执行方案或澄清问题
        """
        # ---- 硬规则：PDF 编辑直接拒绝 ----
        if file_type == FileType.PDF and self._looks_like_edit(user_command):
            return DocumentPlan(
                intent=DocumentIntent.UNSUPPORTED,
                file_type=file_type,
                unsupported_reason="PDF 只支持提取/审阅，不支持编辑。建议输出审阅意见或修改建议文本。",
            )

        # ---- 调用 LLM 分析 ----
        structure_brief = build_structure_brief(structure or {})
        prompt = _build_planner_prompt(user_command, file_type, structure_brief)

        try:
            raw = await llm_gateway.generate(
                prompt,
                system_prompt=_PLANNER_SYSTEM,
                max_tokens=1024,
                temperature=0,
                allow_mock=False,
            )
        except Exception as e:
            logger.error(f"规划 LLM 调用失败: {e}")
            return DocumentPlan(
                intent=DocumentIntent.UNSUPPORTED,
                file_type=file_type,
                unsupported_reason=f"规划智能体调用失败: {e}",
            )

        return self._parse_plan(raw, file_type, user_command, structure or {})

    # ---- 解析 LLM 输出 ----

    def _parse_plan(
        self,
        raw: str,
        file_type: FileType,
        user_command: str,
        structure: dict[str, Any] | None = None,
    ) -> DocumentPlan:
        """解析 LLM 输出为 DocumentPlan，含防御性后处理。"""
        data = self._extract_json(raw)
        if data is None:
            logger.warning(f"规划 LLM 输出无法解析为 JSON: {raw[:200]}")
            return DocumentPlan(
                intent=DocumentIntent.UNSUPPORTED,
                file_type=file_type,
                unsupported_reason="规划智能体输出格式错误，无法解析。",
            )

        intent_str = data.get("intent", "unsupported")
        try:
            intent = DocumentIntent(intent_str)
        except ValueError:
            intent = DocumentIntent.UNSUPPORTED

        raw_operations = data.get("operations", [])
        if raw_operations is None:
            raw_operations = []
        if not isinstance(raw_operations, list):
            logger.warning(f"规划 LLM 输出 operations 不是列表: {raw_operations!r}")
            return DocumentPlan(
                intent=DocumentIntent.UNSUPPORTED,
                file_type=file_type,
                unsupported_reason="规划智能体输出 operation 格式错误",
            )

        operations: list[DocumentOperation] = []
        for op in raw_operations:
            if not isinstance(op, dict):
                logger.warning(f"规划 LLM 输出 operation 不是对象: {op!r}")
                return DocumentPlan(
                    intent=DocumentIntent.UNSUPPORTED,
                    file_type=file_type,
                    unsupported_reason="规划智能体输出 operation 格式错误",
                )
            try:
                operations.append(DocumentOperation(**op))
            except Exception as e:
                logger.warning(f"规划 LLM 输出 operation 格式错误: {e}")
                return DocumentPlan(
                    intent=DocumentIntent.UNSUPPORTED,
                    file_type=file_type,
                    unsupported_reason="规划智能体输出 operation 格式错误",
                )

        risk_str = data.get("risk_level", "low")
        try:
            risk_level = RiskLevel(risk_str)
        except ValueError:
            risk_level = RiskLevel.LOW

        clarification = data.get("clarification_question")
        unsupported_reason = data.get("unsupported_reason")
        requires_confirmation = data.get("requires_confirmation", False)

        # ---- unsupported 意图：清空所有执行相关字段 ----
        if intent == DocumentIntent.UNSUPPORTED:
            return DocumentPlan(
                intent=DocumentIntent.UNSUPPORTED,
                file_type=file_type,
                operations=[],
                backend_required=None,
                risk_level=RiskLevel.LOW,
                requires_confirmation=False,
                clarification_question=None,
                unsupported_reason=unsupported_reason or "规划智能体判定该操作不支持",
            )

        # ---- 后处理：强制规则 ----

        # 高风险动作强制确认
        if self._has_high_risk_action(user_command, operations):
            risk_level = RiskLevel.HIGH
            requires_confirmation = True

        # DOCX 相对表格定位：例如 “在「教师评语及成绩：」的上一个表格填充内容”
        relative_table_result = self._apply_relative_table_reference(
            user_command,
            file_type,
            structure or {},
            operations,
        )
        if relative_table_result.get("clarification"):
            clarification = relative_table_result["clarification"]
            operations = []
        else:
            operations = relative_table_result.get("operations", operations)

        # 模糊定位强制澄清
        if operations and self._is_vague_target(user_command, operations):
            clarification = (
                "您的操作指令定位不够具体，请补充以下信息之一：\n"
                "1. 具体段落编号或标题名称\n"
                "2. 要修改的原文片段\n"
                "3. 行号或单元格引用"
            )
            operations = []

        backend = BACKEND_BY_FILE_TYPE.get(file_type)

        return DocumentPlan(
            intent=intent,
            file_type=file_type,
            operations=operations,
            backend_required=backend if intent != DocumentIntent.UNSUPPORTED else None,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            clarification_question=clarification,
            unsupported_reason=unsupported_reason,
        )

    @classmethod
    def _apply_relative_table_reference(
        cls,
        user_command: str,
        file_type: FileType,
        structure: dict[str, Any],
        operations: list[DocumentOperation],
    ) -> dict[str, Any]:
        """处理“锚点文本的上一个表格”这类确定性相对定位。

        LLM 容易把包含锚点文本的表格当作目标；这里用文档结构强制把
        table_index 改为锚点表格之前的那个表格。
        """
        if file_type != FileType.DOCX or not operations:
            return {"operations": operations}

        anchor = cls._extract_previous_table_anchor(user_command)
        if not anchor:
            return {"operations": operations}

        tables = structure.get("tables", [])
        if not isinstance(tables, list) or not tables:
            return {"clarification": f"未找到表格结构，无法定位“{anchor}”的上一个表格。"}

        anchor_position = cls._find_table_position_containing_text(tables, anchor)
        if anchor_position is None:
            return {"clarification": f"未在文档表格中找到锚点文本“{anchor}”，无法定位上一个表格。"}
        if anchor_position <= 0:
            return {"clarification": f"已找到“{anchor}”所在表格，但它前面没有上一个表格。"}

        previous_table = tables[anchor_position - 1]
        if not isinstance(previous_table, dict):
            return {"clarification": f"“{anchor}”的上一个表格结构无效，无法定位。"}
        previous_table_index = previous_table.get("index", anchor_position - 1)

        adjusted: list[DocumentOperation] = []
        changed = False
        for op in operations:
            target = dict(op.target or {})
            if "table_index" in target:
                target["table_index"] = previous_table_index
                adjusted.append(op.model_copy(update={"target": target}))
                changed = True
            else:
                adjusted.append(op)

        if not changed:
            return {
                "clarification": (
                    f"已定位到“{anchor}”的上一个表格，但当前编辑方案没有表格单元格定位，"
                    "请补充要填充的具体单元格。"
                )
            }
        return {"operations": adjusted}

    @staticmethod
    def _extract_previous_table_anchor(command: str) -> str | None:
        patterns = [
            r"[\"“”「『](.+?)[\"“”」』]\s*的?\s*上一个表格",
            r"在\s*(.+?)\s*的?\s*上一个表格",
        ]
        for pattern in patterns:
            match = re.search(pattern, command)
            if match:
                anchor = match.group(1).strip()
                return anchor or None
        return None

    @staticmethod
    def _find_table_position_containing_text(tables: list[Any], anchor: str) -> int | None:
        normalized_anchor = re.sub(r"\s+", "", anchor)
        for position, table in enumerate(tables):
            if not isinstance(table, dict):
                continue
            cells = table.get("cells", [])
            if not isinstance(cells, list):
                continue
            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                text = re.sub(r"\s+", "", str(cell.get("text") or ""))
                if normalized_anchor and normalized_anchor in text:
                    return position
        return None

    # ---- 辅助方法 ----

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """从 LLM 输出中提取第一个合法 JSON 对象。

        优先尝试完整解析和 ```json 代码块，最后用 raw_decode
        从第一个 '{' 位置逐个尝试，避免贪婪正则吞掉多个 JSON 块。
        """
        text = (raw or "").strip()
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试提取 ```json ... ``` 块
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        # 用 raw_decode 从每个 '{' 位置尝试解析第一个合法 JSON 对象
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            brace = text.find("{", idx)
            if brace == -1:
                break
            try:
                obj, _ = decoder.raw_decode(text, brace)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = brace + 1
        return None

    @staticmethod
    def _looks_like_edit(command: str) -> bool:
        """判断命令是否看起来像编辑操作。"""
        edit_keywords = [
            "修改", "编辑", "替换", "删除", "改写", "添加", "插入",
            "清空", "移除", "更新", "改动", "改成", "改为", "edit",
            "replace", "delete", "modify", "remove", "clear", "insert", "append",
        ]
        text = command.lower()
        return any(kw in text for kw in edit_keywords)

    @staticmethod
    def _has_high_risk_action(command: str, operations: list[DocumentOperation]) -> bool:
        """检查命令或操作是否包含高风险动作。"""
        command_lower = command.lower()
        if any(kw in command_lower for kw in _HIGH_RISK_ACTIONS):
            return True
        for op in operations:
            action_lower = op.action.lower()
            if any(kw in action_lower for kw in _HIGH_RISK_ACTIONS):
                return True
        return False

    @staticmethod
    def _has_table_cell_target(operations: list[DocumentOperation]) -> bool:
        """检查 DOCX 操作是否定位到表格单元格。"""
        table_target_keys = {"table_index", "table", "row", "col", "cell_ref"}
        for op in operations:
            target = op.target or {}
            if "table_index" in target:
                return True
            if {"row", "col"}.issubset(target.keys()) and any(k in target for k in ("table", "table_index")):
                return True
            if target.get("target_type") in {"table_cell", "cell"}:
                return True
            action = op.action.lower()
            if "table" in action or "cell" in action:
                return True
            if table_target_keys.intersection(target.keys()) and "paragraph_index" not in target:
                return True
        return False

    @staticmethod
    def _is_vague_target(command: str, operations: list[DocumentOperation]) -> bool:
        """检查操作目标是否模糊。

        命令含模糊表述时，不信任 LLM 猜测的 target（如 paragraph_index: 3），
        除非 target 中包含用户提供的明确锚点（原文引号、具体行号等）。
        """
        command_lower = command.lower()
        has_vague_marker = any(marker in command_lower for marker in _VAGUE_TARGET_MARKERS)
        if not has_vague_marker:
            return False

        # 用户命令中是否包含明确锚点
        has_explicit_anchor = bool(
            re.search(r'["“”「」『』]', command)  # 引号包裹的原文片段（ASCII + 中文）
            or re.search(r"第\s*\d+\s*[段行节]", command)  # "第3段" "第5行"
            or re.search(r"(?:行|段)\s*\d+", command)  # "行3" "段5"
            or re.search(r"[A-Z]+\d+", command)  # 单元格引用如 B3
        )
        if has_explicit_anchor:
            return False

        # 有模糊标记且无明确锚点 → 强制澄清，不信任 LLM 猜测
        return True


planner = DocumentPlanningAgent()
