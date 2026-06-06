"""MCP 服务路径安全校验

限制文档操作只能访问允许目录下的文件。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def get_allowed_dirs() -> list[Path]:
    """获取允许访问的目录列表"""
    dirs_str = os.environ.get("MCP_ALLOWED_DIR", "./data/uploads,./data/personal_uploads")
    dirs = []
    for d in dirs_str.split(","):
        d = d.strip()
        if d:
            try:
                dirs.append(Path(d).resolve())
            except Exception:
                pass
    return dirs


def validate_file_path(file_path: str) -> tuple[Optional[Path], Optional[str]]:
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


def validate_output_path(output_path: str) -> tuple[Optional[Path], Optional[str]]:
    """校验输出路径是否在允许目录内。

    输出路径不要求文件已存在，但父目录必须存在且在允许目录内。
    """
    if not output_path:
        return None, "缺少 output_path"

    try:
        resolved = Path(output_path).resolve()
    except Exception as e:
        return None, f"路径解析失败: {e}"

    # 检查父目录是否存在
    if not resolved.parent.exists():
        return None, f"输出目录不存在: {resolved.parent}"

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

    return None, f"输出路径不在允许的目录内: {output_path}"
