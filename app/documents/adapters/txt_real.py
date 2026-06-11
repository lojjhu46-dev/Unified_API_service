"""TXT 真实本地 adapter

使用 UTF-8 文本读写。源文档编辑由上层先生成副本，系统生成的编辑副本可原地续编。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from app.documents.adapters.base import BackendUnavailableError
from app.documents.adapters.txt import TxtBackend
from app.observability.logging import get_logger

logger = get_logger(__name__)


class RealTxtBackend(TxtBackend):
    """TXT 真实本地后端

    直接读写文件系统上的 UTF-8 文本文件。
    是否生成副本由 TxtBackend._handle_edit() 根据 DocumentPlan.edit_in_place 决定。
    """

    @property
    def is_available(self) -> bool:
        return True

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        lines = self._read_lines(file_path)
        return {
            "type": "txt",
            "line_count": len(lines),
            "preview": "\n".join(lines[:5]),
        }

    async def read_lines(self, file_path: str) -> list[str]:
        return self._read_lines(file_path)

    async def replace_line(self, file_path: str, line: int, new_text: str) -> None:
        lines = self._read_lines(file_path)
        if line < 1 or line > len(lines):
            raise IndexError(f"行号 {line} 超出范围（共 {len(lines)} 行）")
        lines[line - 1] = new_text
        self._write_lines(file_path, lines)

    async def delete_line(self, file_path: str, line: int) -> None:
        lines = self._read_lines(file_path)
        if line < 1 or line > len(lines):
            raise IndexError(f"行号 {line} 超出范围（共 {len(lines)} 行）")
        lines.pop(line - 1)
        self._write_lines(file_path, lines)

    async def delete_lines(self, file_path: str, start: int, end: int) -> None:
        lines = self._read_lines(file_path)
        if start < 1 or end > len(lines) or start > end:
            raise IndexError(f"行范围 {start}-{end} 超出范围（共 {len(lines)} 行）")
        del lines[start - 1:end]
        self._write_lines(file_path, lines)

    async def append_lines(self, file_path: str, lines: list[str]) -> None:
        existing = self._read_lines(file_path)
        existing.extend(lines)
        self._write_lines(file_path, existing)

    async def replace_text(self, file_path: str, old: str, new: str) -> int:
        lines = self._read_lines(file_path)
        count = 0
        for i, line in enumerate(lines):
            if old in line:
                lines[i] = line.replace(old, new)
                count += 1
        if count > 0:
            self._write_lines(file_path, lines)
        return count

    async def save_copy(self, file_path: str, output_path: str) -> str:
        shutil.copy2(file_path, output_path)
        return output_path

    # ---- 内部方法 ----

    @staticmethod
    def _read_lines(file_path: str) -> list[str]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        text = path.read_text(encoding="utf-8")
        return text.splitlines(keepends=False)

    @staticmethod
    def _write_lines(file_path: str, lines: list[str]) -> None:
        Path(file_path).write_text("\n".join(lines), encoding="utf-8")
