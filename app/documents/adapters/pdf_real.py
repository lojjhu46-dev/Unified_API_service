"""PDF 真实只读 reader

使用 pypdf 读取页数、元数据、页面文本。
拒绝所有编辑计划，不生成输出文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader

from app.documents.adapters.pdf import PdfBackend
from app.observability.logging import get_logger

logger = get_logger(__name__)


class RealPdfBackend(PdfBackend):
    """PDF 真实只读后端

    使用 pypdf 读取 PDF 文件。只支持 read_structure / read_page / extract_text。
    """

    @property
    def is_available(self) -> bool:
        return True

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        reader = self._get_reader(file_path)
        metadata = reader.metadata
        return {
            "type": "pdf",
            "page_count": len(reader.pages),
            "title": metadata.title if metadata else None,
            "author": metadata.author if metadata else None,
        }

    async def read_page(self, file_path: str, page: int) -> str:
        reader = self._get_reader(file_path)
        if page < 1 or page > len(reader.pages):
            raise IndexError(f"页码 {page} 超出范围（共 {len(reader.pages)} 页）")
        return reader.pages[page - 1].extract_text() or ""

    async def extract_text(self, file_path: str, start: int = 1, end: int | None = None) -> str:
        reader = self._get_reader(file_path)
        if end is None:
            end = len(reader.pages)
        if start < 1 or end > len(reader.pages) or start > end:
            raise IndexError(f"页码范围 {start}-{end} 超出范围（共 {len(reader.pages)} 页）")
        pages = reader.pages[start - 1:end]
        return "\n\n".join(page.extract_text() or "" for page in pages)

    @staticmethod
    def _get_reader(file_path: str) -> PdfReader:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return PdfReader(str(path))
