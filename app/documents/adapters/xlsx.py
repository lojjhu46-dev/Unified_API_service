"""XLSX 后端接口与 mock 实现

XlsxBackend 定义 XLSX 文档操作的抽象接口。
MockXlsxBackend 用于测试和无 MCP 环境时的开发验证。
真实 MCP 实现后续按 HTTP 或 subprocess 形态接入。
"""

from __future__ import annotations

import copy
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any

from app.documents.adapters.base import BackendUnavailableError, DocumentBackend
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperation,
    DocumentOperationResult,
    DocumentPlan,
    FileType,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)


class XlsxBackend(DocumentBackend):
    """XLSX 后端抽象接口

    所有 XLSX 操作方法由具体子类实现（MCP、mock 等）。
    execute() 是统一入口，内部按 operation.action 分发。
    """

    @property
    def name(self) -> str:
        return "xlsx_backend"

    # ---- 子类必须实现的原子操作 ----

    @abstractmethod
    async def read_cell(self, file_path: str, sheet: str, cell: str) -> str:
        """读取指定单元格的值

        Args:
            file_path: 文件路径
            sheet: 工作表名
            cell: 单元格引用，如 "A1"、"B3"
        """
        ...

    @abstractmethod
    async def modify_cell(self, file_path: str, sheet: str, cell: str, value: str) -> None:
        """修改指定单元格的值

        Args:
            file_path: 文件路径
            sheet: 工作表名
            cell: 单元格引用
            value: 新值
        """
        ...

    @abstractmethod
    async def append_row(self, file_path: str, sheet: str, values: list[str]) -> None:
        """在指定工作表末尾追加一行

        Args:
            file_path: 文件路径
            sheet: 工作表名
            values: 行数据
        """
        ...

    @abstractmethod
    async def delete_row(self, file_path: str, sheet: str, row: int) -> None:
        """删除指定行

        Args:
            file_path: 文件路径
            sheet: 工作表名
            row: 行号（1-based）
        """
        ...

    @abstractmethod
    async def save_copy(self, file_path: str, output_path: str) -> str:
        """保存文档副本到指定路径，返回输出路径"""
        ...

    # ---- 统一执行入口 ----

    async def execute(self, plan: DocumentPlan) -> DocumentOperationResult:
        """执行 DocumentPlan 中的所有操作。"""
        if not self.is_available:
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不可用",
            )

        if plan.file_type != FileType.XLSX:
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不支持 {plan.file_type.value} 文件",
            )

        if plan.intent == DocumentIntent.UNSUPPORTED:
            return DocumentOperationResult(
                success=False,
                error="不支持的操作意图",
            )

        if plan.backend_required != BackendType.XLSX_MCP:
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不匹配后端 {plan.backend_required}",
            )

        # 只读操作
        if plan.intent in (DocumentIntent.REVIEW, DocumentIntent.EXTRACT, DocumentIntent.SUMMARIZE):
            return await self._handle_readonly(plan)

        # 编辑操作：在副本上执行
        return await self._handle_edit(plan)

    async def _handle_readonly(self, plan: DocumentPlan) -> DocumentOperationResult:
        """处理只读操作"""
        try:
            structure = await self.read_structure(plan.file_path)
            summary_parts = []

            if plan.intent == DocumentIntent.SUMMARIZE:
                summary_parts.append(f"工作簿结构：{structure.get('sheets', [])}")

            for op in plan.operations:
                if op.action == "read_cell":
                    sheet = op.target.get("sheet", "")
                    cell = op.target.get("cell", "")
                    if sheet and cell:
                        value = await self.read_cell(plan.file_path, sheet, cell)
                        summary_parts.append(f"{sheet}!{cell} = {value}")

            return DocumentOperationResult(
                success=True,
                summary="；".join(summary_parts) if summary_parts else "只读操作完成",
                verification={"structure": structure},
            )
        except Exception as e:
            logger.error(f"只读操作失败: {e}")
            return DocumentOperationResult(success=False, error=str(e))

    async def _handle_edit(self, plan: DocumentPlan) -> DocumentOperationResult:
        """处理编辑操作（在副本上执行）"""
        output_path = self._build_output_path(plan.file_path)
        warnings: list[str] = []

        try:
            # 保存副本
            await self.save_copy(plan.file_path, output_path)

            # 执行操作
            for i, op in enumerate(plan.operations):
                try:
                    await self._dispatch_operation(output_path, op)
                except Exception as e:
                    return DocumentOperationResult(
                        success=False,
                        summary=f"操作 {i + 1}/{len(plan.operations)} 失败",
                        error=str(e),
                        warnings=warnings,
                        verification={"partial_output": output_path},
                    )

            return DocumentOperationResult(
                success=True,
                output_file=output_path,
                summary=f"已完成 {len(plan.operations)} 项操作",
                warnings=warnings,
                verification={"original_unchanged": True},
            )
        except Exception as e:
            logger.error(f"编辑操作失败: {e}")
            return DocumentOperationResult(
                success=False,
                error=str(e),
            )

    async def _dispatch_operation(self, file_path: str, op: DocumentOperation) -> None:
        """按 action 类型分发到对应的原子操作"""
        action = op.action.lower()

        if action == "modify_cell":
            sheet = op.target.get("sheet")
            cell = op.target.get("cell")
            if not sheet or not cell:
                raise ValueError(f"modify_cell 缺少 sheet 或 cell: {op.target}")
            await self.modify_cell(file_path, sheet, cell, op.value or "")

        elif action == "append_row":
            sheet = op.target.get("sheet")
            if not sheet:
                raise ValueError(f"append_row 缺少 sheet: {op.target}")
            values = op.target.get("values", [])
            await self.append_row(file_path, sheet, values)

        elif action == "delete_row":
            sheet = op.target.get("sheet")
            row = op.target.get("row")
            if not sheet or row is None:
                raise ValueError(f"delete_row 缺少 sheet 或 row: {op.target}")
            await self.delete_row(file_path, sheet, int(row))

        elif action == "replace_range":
            # 范围替换：批量修改多个单元格
            sheet = op.target.get("sheet")
            replacements = op.target.get("replacements", [])
            if not sheet:
                raise ValueError(f"replace_range 缺少 sheet: {op.target}")
            for rep in replacements:
                cell = rep.get("cell")
                value = rep.get("value", "")
                if cell:
                    await self.modify_cell(file_path, sheet, cell, value)

        else:
            raise ValueError(f"不支持的操作动作: {op.action}")

    @staticmethod
    def _build_output_path(file_path: str) -> str:
        """生成唯一副本路径：原文件名_副本_<uuid>.xlsx，避免连续执行覆盖。"""
        p = Path(file_path)
        short_id = uuid.uuid4().hex[:8]
        return str(p.parent / f"{p.stem}_副本_{short_id}{p.suffix}")


class MockXlsxBackend(XlsxBackend):
    """XLSX 后端的 mock 实现

    使用内存中的字典模拟 workbook 结构，用于测试和开发。
    不依赖任何外部库或 MCP 服务。
    """

    def __init__(self) -> None:
        self._available = True
        # file_path -> {sheet_name: {cell_ref: value}}
        self._workbooks: dict[str, dict[str, dict[str, str]]] = {}

    @property
    def is_available(self) -> bool:
        return self._available

    def set_available(self, value: bool) -> None:
        """测试用：切换可用状态"""
        self._available = value

    def load_workbook(self, file_path: str, sheets: dict[str, dict[str, str]]) -> None:
        """测试用：加载模拟 workbook

        Args:
            file_path: 文件路径
            sheets: {sheet_name: {cell_ref: value}}
        """
        self._workbooks[file_path] = copy.deepcopy(sheets)

    def _get_workbook(self, file_path: str) -> dict[str, dict[str, str]]:
        """获取 workbook，未加载则抛 FileNotFoundError"""
        if file_path not in self._workbooks:
            raise FileNotFoundError(f"文件未加载: {file_path}")
        return self._workbooks[file_path]

    def _get_sheet(self, file_path: str, sheet: str) -> dict[str, str]:
        """获取指定工作表，不存在则抛 KeyError"""
        wb = self._get_workbook(file_path)
        if sheet not in wb:
            raise KeyError(f"工作表不存在: {sheet}")
        return wb[sheet]

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        if not self.is_available:
            raise BackendUnavailableError("MockXlsxBackend 不可用")
        wb = self._get_workbook(file_path)
        sheets_info = {}
        for sheet_name, cells in wb.items():
            # 推断表头：第一行非空单元格
            row1 = {k: v for k, v in cells.items() if k[-1] == "1" or (len(k) > 1 and k[1:] == "1")}
            headers = [v for _, v in sorted(row1.items())]
            sheets_info[sheet_name] = {
                "headers": headers,
                "row_count": self._count_rows(cells),
            }
        return {
            "type": "xlsx",
            "sheets": list(wb.keys()),
            "sheets_info": sheets_info,
        }

    @staticmethod
    def _count_rows(cells: dict[str, str]) -> int:
        """从单元格引用中推断最大行号"""
        max_row = 0
        for ref in cells:
            row_str = "".join(c for c in ref if c.isdigit())
            if row_str:
                max_row = max(max_row, int(row_str))
        return max_row

    async def read_cell(self, file_path: str, sheet: str, cell: str) -> str:
        if not self.is_available:
            raise BackendUnavailableError("MockXlsxBackend 不可用")
        ws = self._get_sheet(file_path, sheet)
        return ws.get(cell.upper(), "")

    async def modify_cell(self, file_path: str, sheet: str, cell: str, value: str) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockXlsxBackend 不可用")
        ws = self._get_sheet(file_path, sheet)
        ws[cell.upper()] = value

    async def append_row(self, file_path: str, sheet: str, values: list[str]) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockXlsxBackend 不可用")
        ws = self._get_sheet(file_path, sheet)
        next_row = self._count_rows(ws) + 1
        for i, val in enumerate(values):
            col_letter = chr(ord("A") + i)
            ws[f"{col_letter}{next_row}"] = val

    async def delete_row(self, file_path: str, sheet: str, row: int) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockXlsxBackend 不可用")
        ws = self._get_sheet(file_path, sheet)
        # 删除指定行：移除该行所有单元格，下方行上移
        to_delete = [k for k in ws if self._row_of(k) == row]
        for k in to_delete:
            del ws[k]
        # 下方行上移
        to_shift = {k: v for k, v in ws.items() if self._row_of(k) > row}
        for k, v in to_shift.items():
            del ws[k]
            new_ref = self._shift_row(k, -1)
            ws[new_ref] = v

    async def save_copy(self, file_path: str, output_path: str) -> str:
        if not self.is_available:
            raise BackendUnavailableError("MockXlsxBackend 不可用")
        wb = self._get_workbook(file_path)
        self._workbooks[output_path] = copy.deepcopy(wb)
        return output_path

    @staticmethod
    def _row_of(cell_ref: str) -> int:
        """从单元格引用提取行号"""
        row_str = "".join(c for c in cell_ref if c.isdigit())
        return int(row_str) if row_str else 0

    @staticmethod
    def _shift_row(cell_ref: str, offset: int) -> str:
        """将单元格引用的行号偏移"""
        col = "".join(c for c in cell_ref if c.isalpha())
        row = int("".join(c for c in cell_ref if c.isdigit()))
        return f"{col}{row + offset}"
