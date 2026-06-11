"""DOCX 后端接口与 mock 实现

DocxBackend 定义 DOCX 文档操作的抽象接口。
MockDocxBackend 用于测试和无 MCP 环境时的开发验证。
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


class DocxBackend(DocumentBackend):
    """DOCX 后端抽象接口

    所有 DOCX 操作方法由具体子类实现（MCP、mock 等）。
    execute() 是统一入口，内部按 operation.action 分发。
    """

    @property
    def name(self) -> str:
        return "docx_backend"

    # ---- 子类必须实现的原子操作 ----

    @abstractmethod
    async def read_paragraph(self, file_path: str, index: int) -> str:
        """读取指定段落的文本"""
        ...

    @abstractmethod
    async def replace_paragraph(self, file_path: str, index: int, new_text: str) -> None:
        """替换指定段落的文本"""
        ...

    @abstractmethod
    async def delete_paragraph(self, file_path: str, index: int) -> None:
        """删除指定段落"""
        ...

    @abstractmethod
    async def append_paragraph(self, file_path: str, text: str) -> None:
        """在文档末尾追加段落"""
        ...

    @abstractmethod
    async def read_table_cell(self, file_path: str, table_index: int, row: int, col: int) -> str:
        """读取指定表格单元格文本"""
        ...

    @abstractmethod
    async def replace_table_cell(self, file_path: str, table_index: int, row: int, col: int, new_text: str) -> None:
        """替换指定表格单元格文本"""
        ...

    @abstractmethod
    async def clear_table_cell(self, file_path: str, table_index: int, row: int, col: int) -> None:
        """清空指定表格单元格文本"""
        ...

    @abstractmethod
    async def save_copy(self, file_path: str, output_path: str) -> str:
        """保存文档副本到指定路径，返回输出路径"""
        ...

    # ---- 统一执行入口 ----

    async def execute(self, plan: DocumentPlan) -> DocumentOperationResult:
        """执行 DocumentPlan 中的所有操作。

        对 EDIT 意图：先保存副本，再在副本上执行操作。
        对 REVIEW / EXTRACT / SUMMARIZE：调用 read_structure + read_paragraph。
        """
        if not self.is_available:
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不可用",
            )

        if plan.file_type != FileType.DOCX:
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不支持 {plan.file_type.value} 文件",
            )

        if plan.intent == DocumentIntent.UNSUPPORTED:
            return DocumentOperationResult(
                success=False,
                error="不支持的操作意图",
            )

        if plan.backend_required != BackendType.DOCX_MCP:
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
                summary_parts.append(f"文档结构：{structure}")

            for op in plan.operations:
                if op.action == "extract_text":
                    page_range = op.target.get("page_range")
                    if page_range:
                        summary_parts.append(f"提取范围：第 {page_range[0]}-{page_range[1]} 页")

            return DocumentOperationResult(
                success=True,
                summary="；".join(summary_parts) if summary_parts else "只读操作完成",
                verification={"structure": structure},
            )
        except Exception as e:
            logger.error(f"只读操作失败: {e}")
            return DocumentOperationResult(success=False, error=str(e))

    async def _handle_edit(self, plan: DocumentPlan) -> DocumentOperationResult:
        """处理编辑操作：源文档生成副本，系统副本可原地续编。"""
        output_path = plan.file_path if plan.edit_in_place else self._build_output_path(plan.file_path)
        warnings: list[str] = []

        try:
            if not plan.edit_in_place:
                await self.save_copy(plan.file_path, output_path)

            # 执行操作
            for i, op in enumerate(plan.operations):
                try:
                    await self._dispatch_operation(output_path, op)
                except Exception as e:
                    verification = {
                        "partial_output": output_path,
                        "edited_in_place": plan.edit_in_place,
                    }
                    return DocumentOperationResult(
                        success=False,
                        summary=f"操作 {i + 1}/{len(plan.operations)} 失败",
                        error=str(e),
                        warnings=warnings,
                        verification=verification,
                    )

            verification = {"edited_in_place": plan.edit_in_place}
            if not plan.edit_in_place:
                verification["original_unchanged"] = True
            return DocumentOperationResult(
                success=True,
                output_file=output_path,
                summary=f"已完成 {len(plan.operations)} 项操作",
                warnings=warnings,
                verification=verification,
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

        if action in ("replace_paragraph", "replace_text"):
            index = self._extract_index(op)
            if index is None:
                raise ValueError(f"replace_paragraph 缺少 paragraph_index: {op.target}")
            await self.replace_paragraph(file_path, index, op.value or "")

        elif action in ("delete_paragraph", "delete_text"):
            index = self._extract_index(op)
            if index is None:
                raise ValueError(f"delete_paragraph 缺少 paragraph_index: {op.target}")
            await self.delete_paragraph(file_path, index)

        elif action in ("append_text", "append_paragraph"):
            await self.append_paragraph(file_path, op.value or "")

        elif action == "replace_table_cell":
            table_index, row, col = self._extract_table_cell_target(op)
            await self.replace_table_cell(file_path, table_index, row, col, op.value or "")

        elif action == "clear_table_cell":
            table_index, row, col = self._extract_table_cell_target(op)
            await self.clear_table_cell(file_path, table_index, row, col)

        else:
            raise ValueError(f"不支持的操作动作: {op.action}")

    @staticmethod
    def _extract_index(op: DocumentOperation) -> int | None:
        """从 target 中提取段落索引，显式判断 key 存在（0 是合法索引）。"""
        target = op.target
        for key in ("paragraph_index", "index"):
            if key in target:
                return int(target[key])
        return None

    @staticmethod
    def _extract_table_cell_target(op: DocumentOperation) -> tuple[int, int, int]:
        """从 target 中提取 table_index/row/col，0 是合法索引。"""
        target = op.target
        missing = [key for key in ("table_index", "row", "col") if key not in target]
        if missing:
            raise ValueError(f"{op.action} 缺少 {', '.join(missing)}: {op.target}")
        return int(target["table_index"]), int(target["row"]), int(target["col"])

    @staticmethod
    def _build_output_path(file_path: str) -> str:
        """生成唯一副本路径：原文件名_副本_<uuid>.docx，避免连续执行覆盖。"""
        p = Path(file_path)
        short_id = uuid.uuid4().hex[:8]
        return str(p.parent / f"{p.stem}_副本_{short_id}{p.suffix}")


class MockDocxBackend(DocxBackend):
    """DOCX 后端的 mock 实现

    使用内存中的段落列表模拟文档操作，用于测试和开发。
    不依赖任何外部库或 MCP 服务。
    """

    def __init__(self) -> None:
        self._available = True
        # file_path -> list[str]（段落文本）
        self._documents: dict[str, list[str]] = {}
        # file_path -> list[table][row][col]（表格文本）
        self._tables: dict[str, list[list[list[str]]]] = {}

    @property
    def is_available(self) -> bool:
        return self._available

    def set_available(self, value: bool) -> None:
        """测试用：切换可用状态"""
        self._available = value

    def load_document(self, file_path: str, paragraphs: list[str]) -> None:
        """测试用：加载模拟文档"""
        self._documents[file_path] = list(paragraphs)
        self._tables.setdefault(file_path, [])

    def load_tables(self, file_path: str, tables: list[list[list[str]]]) -> None:
        """测试用：加载模拟表格"""
        self._tables[file_path] = copy.deepcopy(tables)

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        paras = self._get_paras(file_path)
        return {
            "type": "docx",
            "paragraph_count": len(paras),
            "headings": [p for p in paras if p.startswith("#")],
            "tables": self._build_tables_structure(file_path),
        }

    def _get_paras(self, file_path: str) -> list[str]:
        """获取文档段落列表，未加载则抛 FileNotFoundError"""
        if file_path not in self._documents:
            raise FileNotFoundError(f"文件未加载: {file_path}")
        return self._documents[file_path]

    async def read_paragraph(self, file_path: str, index: int) -> str:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        paras = self._get_paras(file_path)
        if index < 0 or index >= len(paras):
            raise IndexError(f"段落索引 {index} 超出范围（共 {len(paras)} 段）")
        return paras[index]

    async def replace_paragraph(self, file_path: str, index: int, new_text: str) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        paras = self._get_paras(file_path)
        if index < 0 or index >= len(paras):
            raise IndexError(f"段落索引 {index} 超出范围（共 {len(paras)} 段）")
        paras[index] = new_text

    async def delete_paragraph(self, file_path: str, index: int) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        paras = self._get_paras(file_path)
        if index < 0 or index >= len(paras):
            raise IndexError(f"段落索引 {index} 超出范围（共 {len(paras)} 段）")
        paras.pop(index)

    async def append_paragraph(self, file_path: str, text: str) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        paras = self._get_paras(file_path)
        paras.append(text)

    async def read_table_cell(self, file_path: str, table_index: int, row: int, col: int) -> str:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        return self._get_table_cell(file_path, table_index, row, col)

    async def replace_table_cell(self, file_path: str, table_index: int, row: int, col: int, new_text: str) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        self._set_table_cell(file_path, table_index, row, col, new_text)

    async def clear_table_cell(self, file_path: str, table_index: int, row: int, col: int) -> None:
        await self.replace_table_cell(file_path, table_index, row, col, "")

    async def save_copy(self, file_path: str, output_path: str) -> str:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        paras = self._documents.get(file_path)
        if paras is None:
            raise FileNotFoundError(f"文件未加载: {file_path}")
        # 保存副本到内存（复制段落列表）
        self._documents[output_path] = copy.deepcopy(paras)
        self._tables[output_path] = copy.deepcopy(self._tables.get(file_path, []))
        return output_path

    def _get_tables(self, file_path: str) -> list[list[list[str]]]:
        self._get_paras(file_path)
        return self._tables.setdefault(file_path, [])

    def _get_table_cell(self, file_path: str, table_index: int, row: int, col: int) -> str:
        if table_index < 0 or row < 0 or col < 0:
            raise IndexError(f"表格单元格索引越界: table_index={table_index}, row={row}, col={col}")
        tables = self._get_tables(file_path)
        try:
            return tables[table_index][row][col]
        except IndexError:
            raise IndexError(f"表格单元格索引越界: table_index={table_index}, row={row}, col={col}")

    def _set_table_cell(self, file_path: str, table_index: int, row: int, col: int, value: str) -> None:
        if table_index < 0 or row < 0 or col < 0:
            raise IndexError(f"表格单元格索引越界: table_index={table_index}, row={row}, col={col}")
        tables = self._get_tables(file_path)
        try:
            tables[table_index][row][col] = value
        except IndexError:
            raise IndexError(f"表格单元格索引越界: table_index={table_index}, row={row}, col={col}")

    def _build_tables_structure(self, file_path: str) -> list[dict[str, Any]]:
        tables = self._tables.get(file_path, [])
        result = []
        for table_index, table in enumerate(tables):
            cells = []
            max_cols = 0
            for row_index, row_cells in enumerate(table):
                max_cols = max(max_cols, len(row_cells))
                for col_index, text in enumerate(row_cells):
                    cells.append({
                        "row": row_index,
                        "col": col_index,
                        "text": text,
                        "merged": False,
                    })
            result.append({
                "index": table_index,
                "rows": len(table),
                "cols": max_cols,
                "cells": cells,
                "truncated": False,
            })
        return result
