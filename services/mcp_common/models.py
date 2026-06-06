"""MCP 服务统一响应模型"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class McpResponse(BaseModel):
    """标准 MCP 响应格式：{success: bool, data?: object, error?: string}"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


def ok(data: Any = None) -> dict:
    """返回成功响应"""
    return McpResponse(success=True, data=data).model_dump()


def fail(error: str) -> dict:
    """返回失败响应"""
    return McpResponse(success=False, error=error).model_dump()
