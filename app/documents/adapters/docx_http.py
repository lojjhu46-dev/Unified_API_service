"""DOCX HTTP MCP 后端

通过 HTTP 调用独立的 DOCX MCP 服务。
服务需能访问同一文件系统路径。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.documents.adapters.base import BackendUnavailableError
from app.documents.adapters.docx import DocxBackend
from app.observability.logging import get_logger

logger = get_logger(__name__)


class HttpDocxBackend(DocxBackend):
    """DOCX HTTP MCP 后端

    通过 HTTP JSON 调用独立的 DOCX MCP 服务。
    请求格式：POST {base_url}/{endpoint}，JSON body。
    响应格式：{success: bool, data?: object, error?: string}。

    不缓存可用性状态：每次调用独立处理，临时网络抖动不会永久阻断。
    """

    def __init__(self, base_url: str, timeout: float = 30.0, api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def is_available(self) -> bool:
        return True

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _call(self, endpoint: str, payload: dict) -> dict:
        """调用 MCP 服务端点，返回 data 字段或抛异常。"""
        client = await self._get_client()
        url = f"{self._base_url}{endpoint}"
        headers = {}
        if self._api_key:
            headers["X-MCP-API-Key"] = self._api_key
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.post(url, json=payload, headers=headers)
                break
            except (httpx.TimeoutException, httpx.RequestError) as e:
                logger.warning(
                    f"DOCX MCP 请求异常: {type(e).__name__}: {e}",
                    exc_info=True,
                    extra={"endpoint": endpoint, "attempt": attempt, "max_attempts": max_attempts},
                )
                if attempt < max_attempts:
                    await asyncio.sleep(0.3 * attempt)
                    continue
                raise BackendUnavailableError(f"DOCX MCP 连接失败: {e}") from e

        if resp.status_code != 200:
            logger.warning(
                f"DOCX MCP HTTP {resp.status_code}: {resp.text[:200]}",
                extra={"endpoint": endpoint},
            )
            raise BackendUnavailableError(f"DOCX MCP HTTP {resp.status_code}: {url}")

        try:
            body = resp.json()
        except Exception as e:
            logger.warning(
                f"DOCX MCP 响应非 JSON: {type(e).__name__}: {e}",
                exc_info=True,
                extra={"endpoint": endpoint, "body": resp.text[:200]},
            )
            raise BackendUnavailableError(f"DOCX MCP 响应非 JSON: {resp.text[:200]}")

        if not body.get("success"):
            logger.warning(
                f"DOCX MCP 业务失败: {body.get('error', '未知')}",
                extra={"endpoint": endpoint},
            )
            raise BackendUnavailableError(f"DOCX MCP 错误: {body.get('error', '未知')}")

        return body.get("data", {})

    async def read_structure(self, file_path: str) -> dict[str, Any]:
        return await self._call("/docx/read_structure", {"file_path": file_path})

    async def read_paragraph(self, file_path: str, index: int) -> str:
        data = await self._call("/docx/read_paragraph", {
            "file_path": file_path,
            "index": index,
        })
        return data.get("text", "")

    async def replace_paragraph(self, file_path: str, index: int, new_text: str) -> None:
        await self._call("/docx/replace_paragraph", {
            "file_path": file_path,
            "index": index,
            "new_text": new_text,
        })

    async def delete_paragraph(self, file_path: str, index: int) -> None:
        await self._call("/docx/delete_paragraph", {
            "file_path": file_path,
            "index": index,
        })

    async def append_paragraph(self, file_path: str, text: str) -> None:
        await self._call("/docx/append_paragraph", {
            "file_path": file_path,
            "text": text,
        })

    async def read_table_cell(self, file_path: str, table_index: int, row: int, col: int) -> str:
        data = await self._call("/docx/read_table_cell", {
            "file_path": file_path,
            "table_index": table_index,
            "row": row,
            "col": col,
        })
        return data.get("text", "")

    async def replace_table_cell(self, file_path: str, table_index: int, row: int, col: int, new_text: str) -> None:
        await self._call("/docx/replace_table_cell", {
            "file_path": file_path,
            "table_index": table_index,
            "row": row,
            "col": col,
            "new_text": new_text,
        })

    async def clear_table_cell(self, file_path: str, table_index: int, row: int, col: int) -> None:
        await self._call("/docx/clear_table_cell", {
            "file_path": file_path,
            "table_index": table_index,
            "row": row,
            "col": col,
        })

    async def save_copy(self, file_path: str, output_path: str) -> str:
        data = await self._call("/docx/save_copy", {
            "file_path": file_path,
            "output_path": output_path,
        })
        return data.get("output_path", output_path)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
