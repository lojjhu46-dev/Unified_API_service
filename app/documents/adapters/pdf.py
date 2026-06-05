"""PDF 后端接口与 mock 实现

PdfBackend 定义 PDF 文档的只读操作接口。
MockPdfBackend 用于测试和无 MCP 环境时的开发验证。
PDF 永远只读，不支持编辑、删除、生成已编辑副本。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.documents.adapters.base import BackendUnavailableError, DocumentBackend
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperationResult,
    DocumentPlan,
    FileType,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)


class PdfBackend(DocumentBackend):
    """PDF 后端只读接口

    PDF 不支持任何编辑操作。所有编辑请求在 execute() 入口直接拒绝。
    只支持 REVIEW / EXTRACT / SUMMARIZE 意图。
    """

    @property
    def name(self) -> str:
        return "pdf_backend"

    # ---- 子类必须实现的只读原子操作 ----

    @abstractmethod
    async def read_structure(self, file_path: str) -> dict[str, Any]:
        """读取 PDF 文档结构（页数、目录等）"""
        ...

    @abstractmethod
    async def read_page(self, file_path: str, page: int) -> str:
        """读取指定页的文本内容（1-based）"""
        ...

    @abstractmethod
    async def extract_text(self, file_path: str, start: int = 1, end: int | None = None) -> str:
        """提取指定范围的文本（1-based，包含 start 和 end）"""
        ...

    # ---- 统一执行入口 ----

    async def execute(self, plan: DocumentPlan) -> DocumentOperationResult:
        """执行 DocumentPlan。PDF 只支持只读操作。"""
        if not self.is_available:
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不可用",
            )

        if plan.file_type != FileType.PDF:
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不支持 {plan.file_type.value} 文件",
            )

        if plan.intent == DocumentIntent.UNSUPPORTED:
            return DocumentOperationResult(
                success=False,
                error="不支持的操作意图",
            )

        if plan.backend_required != BackendType.PDF_READER:
            return DocumentOperationResult(
                success=False,
                error=f"{self.name} 不匹配后端 {plan.backend_required}",
            )

        # PDF 禁止所有编辑操作
        if plan.intent == DocumentIntent.EDIT:
            return DocumentOperationResult(
                success=False,
                error="PDF 不支持编辑，仅支持 extract / review / summarize",
            )

        # 只读操作
        return await self._handle_readonly(plan)

    async def _handle_readonly(self, plan: DocumentPlan) -> DocumentOperationResult:
        """处理只读操作"""
        try:
            structure = await self.read_structure(plan.file_path)
            summary_parts = []

            if plan.intent == DocumentIntent.SUMMARIZE:
                summary_parts.append(f"PDF 共 {structure.get('page_count', '?')} 页")

            for op in plan.operations:
                if op.action == "extract_text":
                    page_range = op.target.get("page_range")
                    if page_range is not None:
                        if not isinstance(page_range, list) or len(page_range) < 2:
                            raise ValueError(f"extract_text 的 page_range 需要至少 2 个元素: {page_range}")
                        text = await self.extract_text(plan.file_path, page_range[0], page_range[1])
                        preview = text[:200] + ("..." if len(text) > 200 else "")
                        summary_parts.append(f"第{page_range[0]}-{page_range[1]}页：{preview}")
                    else:
                        text = await self.extract_text(plan.file_path)
                        preview = text[:200] + ("..." if len(text) > 200 else "")
                        summary_parts.append(f"全文：{preview}")

                elif op.action == "read_page":
                    page = op.target.get("page")
                    if page is None:
                        raise ValueError(f"read_page 缺少 page: {op.target}")
                    text = await self.read_page(plan.file_path, int(page))
                    preview = text[:200] + ("..." if len(text) > 200 else "")
                    summary_parts.append(f"第{page}页：{preview}")

                else:
                    raise ValueError(f"不支持的操作动作: {op.action}")

            return DocumentOperationResult(
                success=True,
                summary="；".join(summary_parts) if summary_parts else "只读操作完成",
                verification={"structure": structure},
            )
        except Exception as e:
            logger.error(f"只读操作失败: {e}")
            return DocumentOperationResult(success=False, error=str(e))


class MockPdfBackend(PdfBackend):
    """PDF 后端的 mock 实现

    使用内存中的页面列表模拟 PDF 操作，用于测试和开发。
    不依赖任何外部库或 MCP 服务。
    """

    def __init__(self) -> None:
        self._available = True
        # file_path -> list[str]（每页文本）
        self._documents: dict[str, list[str]] = {}

    @property
    def is_available(self) -> bool:
        return self._available

    def set_available(self, value: bool) -> None:
        """测试用：切换可用状态"""
        self._available = value

    def load_document(self, file_path: str, pages: list[str]) -> None:
        """测试用：加载模拟 PDF

        Args:
            file_path: 文件路径
            pages: 每页文本内容列表
        """
        self._documents[file_path] = list(pages)

    def _get_pages(self, file_path: str) -> list[str]:
        """获取页面列表，未加载则抛 FileNotFoundError"""
        if file_path not in self._documents:
            raise FileNotFoundError(f"文件未加载: {file_path}")
        return self._documents[file_path]

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        if not self.is_available:
            raise BackendUnavailableError("MockPdfBackend 不可用")
        pages = self._get_pages(file_path)
        return {
            "type": "pdf",
            "page_count": len(pages),
        }

    async def read_page(self, file_path: str, page: int) -> str:
        if not self.is_available:
            raise BackendUnavailableError("MockPdfBackend 不可用")
        pages = self._get_pages(file_path)
        if page < 1 or page > len(pages):
            raise IndexError(f"页码 {page} 超出范围（共 {len(pages)} 页）")
        return pages[page - 1]

    async def extract_text(self, file_path: str, start: int = 1, end: int | None = None) -> str:
        if not self.is_available:
            raise BackendUnavailableError("MockPdfBackend 不可用")
        pages = self._get_pages(file_path)
        if end is None:
            end = len(pages)
        if start < 1 or end > len(pages) or start > end:
            raise IndexError(f"页码范围 {start}-{end} 超出范围（共 {len(pages)} 页）")
        return "\n\n".join(pages[start - 1:end])
