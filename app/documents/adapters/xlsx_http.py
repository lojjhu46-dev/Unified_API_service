"""XLSX HTTP MCP 后端

通过 HTTP 调用独立的 XLSX MCP 服务。
服务需能访问同一文件系统路径。
"""

from __future__ import annotations

from typing import Any

import httpx

from app.documents.adapters.base import BackendUnavailableError
from app.documents.adapters.xlsx import XlsxBackend
from app.observability.logging import get_logger

logger = get_logger(__name__)


class HttpXlsxBackend(XlsxBackend):
    """XLSX HTTP MCP 后端

    通过 HTTP JSON 调用独立的 XLSX MCP 服务。
    请求格式：POST {base_url}/{endpoint}，JSON body。
    响应格式：{success: bool, data?: object, error?: string}。
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        return True

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _call(self, endpoint: str, payload: dict) -> dict:
        """调用 MCP 服务端点，返回 data 字段或抛异常。"""
        client = await self._get_client()
        url = f"{self._base_url}{endpoint}"
        try:
            resp = await client.post(url, json=payload)
        except httpx.TimeoutException:
            self._available = False
            raise BackendUnavailableError(f"XLSX MCP 超时: {url}")
        except httpx.RequestError as e:
            self._available = False
            raise BackendUnavailableError(f"XLSX MCP 连接失败: {e}")

        if resp.status_code != 200:
            raise BackendUnavailableError(f"XLSX MCP HTTP {resp.status_code}: {url}")

        try:
            body = resp.json()
        except Exception:
            raise BackendUnavailableError(f"XLSX MCP 响应非 JSON: {resp.text[:200]}")

        if not body.get("success"):
            raise BackendUnavailableError(f"XLSX MCP 错误: {body.get('error', '未知')}")

        self._available = True
        return body.get("data", {})

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        return await self._call("/xlsx/read_structure", {"file_path": file_path})

    async def read_cell(self, file_path: str, sheet: str, cell: str) -> str:
        data = await self._call("/xlsx/read_cell", {
            "file_path": file_path,
            "sheet": sheet,
            "cell": cell,
        })
        return data.get("value", "")

    async def modify_cell(self, file_path: str, sheet: str, cell: str, value: str) -> None:
        await self._call("/xlsx/modify_cell", {
            "file_path": file_path,
            "sheet": sheet,
            "cell": cell,
            "value": value,
        })

    async def append_row(self, file_path: str, sheet: str, values: list[str]) -> None:
        await self._call("/xlsx/append_row", {
            "file_path": file_path,
            "sheet": sheet,
            "values": values,
        })

    async def delete_row(self, file_path: str, sheet: str, row: int) -> None:
        await self._call("/xlsx/delete_row", {
            "file_path": file_path,
            "sheet": sheet,
            "row": row,
        })

    async def save_copy(self, file_path: str, output_path: str) -> str:
        data = await self._call("/xlsx/save_copy", {
            "file_path": file_path,
            "output_path": output_path,
        })
        return data.get("output_path", output_path)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
