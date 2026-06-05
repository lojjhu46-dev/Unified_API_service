"""文档路径安全校验

限制文档工具只能访问上传目录下的文件，防止 document_file_path 成为任意文件访问入口。
"""

from __future__ import annotations

import os
from pathlib import Path

from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


def get_allowed_dirs() -> list[Path]:
    """获取允许访问的目录列表"""
    dirs = []
    for d in (settings.upload_dir, settings.personal_upload_dir):
        if d:
            try:
                dirs.append(Path(d).resolve())
            except Exception:
                pass
    return dirs


def validate_file_path(file_path: str) -> tuple[Path | None, str | None]:
    """校验文件路径是否在允许目录内。

    Returns:
        (resolved_path, error_message) — 成功时 error_message 为 None。
    """
    if not file_path:
        return None, "缺少 file_path"

    try:
        resolved = Path(file_path).resolve()
    except Exception as e:
        return None, f"路径解析失败: {e}"

    # 检查文件是否存在
    if not resolved.exists():
        return None, f"文件不存在: {file_path}"

    if not resolved.is_file():
        return None, f"不是文件: {file_path}"

    # 检查是否在允许目录内
    allowed = get_allowed_dirs()
    if not allowed:
        return None, "未配置允许的上传目录"

    for allowed_dir in allowed:
        try:
            resolved.relative_to(allowed_dir)
            return resolved, None
        except ValueError:
            continue

    return None, f"文件路径不在允许的目录内: {file_path}"


def validate_output_path(output_path: str, source_path: Path) -> tuple[Path | None, str | None]:
    """校验输出路径是否安全（与源文件同目录且在允许目录内）。

    Returns:
        (resolved_path, error_message)
    """
    try:
        resolved = Path(output_path).resolve()
    except Exception as e:
        return None, f"输出路径解析失败: {e}"

    # 输出文件必须与源文件同目录
    if resolved.parent != source_path.parent:
        return None, "输出文件必须与源文件同目录"

    # 检查是否在允许目录内
    allowed = get_allowed_dirs()
    for allowed_dir in allowed:
        try:
            resolved.relative_to(allowed_dir)
            return resolved, None
        except ValueError:
            continue

    return None, f"输出路径不在允许的目录内: {output_path}"
