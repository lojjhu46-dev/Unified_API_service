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

        if plan.backend_required not in (BackendType.DOCX_MCP, None):
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不匹配后端 {plan.backend_required}",
            )

        if plan.intent == DocumentIntent.UNSUPPORTED:
            return DocumentOperationResult(
                success=False,
                error="不支持的操作意图",
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
                output_file=output_path if Path(output_path).exists() else None,
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

    @property
    def is_available(self) -> bool:
        return self._available

    def set_available(self, value: bool) -> None:
        """测试用：切换可用状态"""
        self._available = value

    def load_document(self, file_path: str, paragraphs: list[str]) -> None:
        """测试用：加载模拟文档"""
        self._documents[file_path] = list(paragraphs)

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        paras = self._get_paras(file_path)
        return {
            "type": "docx",
            "paragraph_count": len(paras),
            "headings": [p for p in paras if p.startswith("#")],
            "tables": [],
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

    async def save_copy(self, file_path: str, output_path: str) -> str:
        if not self.is_available:
            raise BackendUnavailableError("MockDocxBackend 不可用")
        paras = self._documents.get(file_path)
        if paras is None:
            raise FileNotFoundError(f"文件未加载: {file_path}")
        # 保存副本到内存（复制段落列表）
        self._documents[output_path] = copy.deepcopy(paras)
        return output_path
