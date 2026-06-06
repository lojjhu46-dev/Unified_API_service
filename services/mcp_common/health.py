"""MCP 服务健康检查端点"""

from __future__ import annotations

from typing import Callable

from fastapi import FastAPI


def add_health_endpoint(app: FastAPI, service_name: str) -> None:
    """为 FastAPI 应用添加 /health 端点"""

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": service_name}
