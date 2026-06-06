"""MCP 服务 API Key 认证"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request


def get_api_key() -> Optional[str]:
    """获取配置的 API Key"""
    return os.environ.get("MCP_API_KEY")


def verify_api_key(request: Request) -> None:
    """验证请求中的 API Key

    如果配置了 MCP_API_KEY，则必须在请求头中提供 X-MCP-API-Key。
    如果未配置，则仅允许开发模式（不验证）。
    """
    configured_key = get_api_key()

    # 开发模式：未配置 API Key 时允许访问
    if not configured_key:
        return

    # 生产模式：必须提供正确的 API Key
    provided_key = request.headers.get("X-MCP-API-Key")

    if not provided_key:
        raise HTTPException(
            status_code=401,
            detail="缺少 X-MCP-API-Key 请求头",
        )

    if provided_key != configured_key:
        raise HTTPException(
            status_code=403,
            detail="API Key 无效",
        )
