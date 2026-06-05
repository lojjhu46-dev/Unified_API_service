"""TXT 后端接口与 mock 实现

TxtBackend 定义 TXT 文档操作的抽象接口。
MockTxtBackend 用于测试和无 MCP 环境时的开发验证。
TXT 使用本地确定性 adapter，不依赖外部 MCP 服务。
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


class TxtBackend(DocumentBackend):
    """TXT 后端抽象接口

    所有 TXT 操作方法由具体子类实现。
    execute() 是统一入口，内部按 operation.action 分发。
    """

    @property
    def name(self) -> str:
        return "txt_backend"

    # ---- 子类必须实现的原子操作 ----

    @abstractmethod
    async def read_lines(self, file_path: str) -> list[str]:
        """读取所有行"""
        ...

    @abstractmethod
    async def replace_line(self, file_path: str, line: int, new_text: str) -> None:
        """替换指定行（1-based）"""
        ...

    @abstractmethod
    async def delete_line(self, file_path: str, line: int) -> None:
        """删除指定行（1-based）"""
        ...

    @abstractmethod
    async def delete_lines(self, file_path: str, start: int, end: int) -> None:
        """删除指定范围的行（1-based，包含 start 和 end）"""
        ...

    @abstractmethod
    async def append_lines(self, file_path: str, lines: list[str]) -> None:
        """在文件末尾追加行"""
        ...

    @abstractmethod
    async def replace_text(self, file_path: str, old: str, new: str) -> int:
        """替换文本片段，返回替换次数"""
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

        if plan.file_type != FileType.TXT:
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不支持 {plan.file_type.value} 文件",
            )

        if plan.intent == DocumentIntent.UNSUPPORTED:
            return DocumentOperationResult(
                success=False,
                error="不支持的操作意图",
            )

        if plan.backend_required != BackendType.TEXT_ADAPTER:
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
            lines = await self.read_lines(plan.file_path)
            summary_parts = []

            if plan.intent == DocumentIntent.SUMMARIZE:
                summary_parts.append(f"共 {len(lines)} 行")

            for op in plan.operations:
                if op.action == "read_lines":
                    start = op.target.get("start", 1)
                    end = op.target.get("end", len(lines))
                    selected = lines[max(0, start - 1):min(end, len(lines))]
                    summary_parts.append(f"第{start}-{end}行：{'、'.join(selected[:5])}")

            structure = await self.read_structure(plan.file_path)
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

        if action == "replace_line":
            line = self._extract_line(op)
            if line is None:
                raise ValueError(f"replace_line 缺少 line: {op.target}")
            await self.replace_line(file_path, line, op.value or "")

        elif action == "delete_line":
            line = self._extract_line(op)
            if line is None:
                raise ValueError(f"delete_line 缺少 line: {op.target}")
            await self.delete_line(file_path, line)

        elif action == "delete_lines":
            start = op.target.get("start")
            end = op.target.get("end")
            if start is None or end is None:
                raise ValueError(f"delete_lines 缺少 start 或 end: {op.target}")
            await self.delete_lines(file_path, int(start), int(end))

        elif action == "append_lines":
            lines = op.target.get("lines", [])
            if not lines:
                if not op.value:
                    raise ValueError(f"append_lines 缺少 lines 或 value: {op.target}")
                lines = [op.value]
            if not isinstance(lines, list) or any(not isinstance(line, str) for line in lines):
                raise ValueError(f"append_lines 的 lines 必须是字符串列表: {op.target}")
            await self.append_lines(file_path, lines)

        elif action == "replace_text":
            old = op.target.get("old")
            if not old:
                raise ValueError(f"replace_text 缺少 old: {op.target}")
            count = await self.replace_text(file_path, old, op.value or "")
            if count == 0:
                raise ValueError(f"replace_text 未找到匹配: {old}")

        else:
            raise ValueError(f"不支持的操作动作: {op.action}")

    @staticmethod
    def _extract_line(op: DocumentOperation) -> int | None:
        """从 target 中提取行号，显式判断 key 存在（1-based）。"""
        target = op.target
        for key in ("line", "line_number", "index"):
            if key in target:
                return int(target[key])
        return None

    @staticmethod
    def _build_output_path(file_path: str) -> str:
        """生成唯一副本路径：原文件名_副本_<uuid>.txt，避免连续执行覆盖。"""
        p = Path(file_path)
        short_id = uuid.uuid4().hex[:8]
        return str(p.parent / f"{p.stem}_副本_{short_id}{p.suffix}")


class MockTxtBackend(TxtBackend):
    """TXT 后端的 mock 实现

    使用内存中的行列表模拟文本文件操作，用于测试和开发。
    不依赖任何外部库或 MCP 服务。
    """

    def __init__(self) -> None:
        self._available = True
        # file_path -> list[str]（行文本）
        self._files: dict[str, list[str]] = {}

    @property
    def is_available(self) -> bool:
        return self._available

    def set_available(self, value: bool) -> None:
        """测试用：切换可用状态"""
        self._available = value

    def load_file(self, file_path: str, lines: list[str]) -> None:
        """测试用：加载模拟文件"""
        self._files[file_path] = list(lines)

    def _get_lines(self, file_path: str) -> list[str]:
        """获取文件行列表，未加载则抛 FileNotFoundError"""
        if file_path not in self._files:
            raise FileNotFoundError(f"文件未加载: {file_path}")
        return self._files[file_path]

    async def read_lines(self, file_path: str) -> list[str]:
        if not self.is_available:
            raise BackendUnavailableError("MockTxtBackend 不可用")
        return list(self._get_lines(file_path))

    async def replace_line(self, file_path: str, line: int, new_text: str) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockTxtBackend 不可用")
        lines = self._get_lines(file_path)
        if line < 1 or line > len(lines):
            raise IndexError(f"行号 {line} 超出范围（共 {len(lines)} 行）")
        lines[line - 1] = new_text

    async def delete_line(self, file_path: str, line: int) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockTxtBackend 不可用")
        lines = self._get_lines(file_path)
        if line < 1 or line > len(lines):
            raise IndexError(f"行号 {line} 超出范围（共 {len(lines)} 行）")
        lines.pop(line - 1)

    async def delete_lines(self, file_path: str, start: int, end: int) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockTxtBackend 不可用")
        lines = self._get_lines(file_path)
        if start < 1 or end > len(lines) or start > end:
            raise IndexError(f"行范围 {start}-{end} 超出范围（共 {len(lines)} 行）")
        del lines[start - 1:end]

    async def append_lines(self, file_path: str, lines: list[str]) -> None:
        if not self.is_available:
            raise BackendUnavailableError("MockTxtBackend 不可用")
        self._get_lines(file_path).extend(lines)

    async def replace_text(self, file_path: str, old: str, new: str) -> int:
        if not self.is_available:
            raise BackendUnavailableError("MockTxtBackend 不可用")
        lines = self._get_lines(file_path)
        count = 0
        for i, line in enumerate(lines):
            if old in line:
                lines[i] = line.replace(old, new)
                count += 1
        return count

    async def save_copy(self, file_path: str, output_path: str) -> str:
        if not self.is_available:
            raise BackendUnavailableError("MockTxtBackend 不可用")
        lines = self._get_lines(file_path)
        self._files[output_path] = copy.deepcopy(lines)
        return output_path

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        if not self.is_available:
            raise BackendUnavailableError("MockTxtBackend 不可用")
        lines = self._get_lines(file_path)
        return {
            "type": "txt",
            "line_count": len(lines),
            "preview": "\n".join(lines[:5]),
        }
