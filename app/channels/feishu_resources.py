"""Read-only Feishu cloud resource helpers."""

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import settings
from app.llm.gateway import LLMGatewayError, llm_gateway
from app.observability.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_RESOURCE_TYPES = {"docx", "sheets", "bitable"}
RESOURCE_PERMISSION_ERROR = (
    "我识别到了飞书文档链接，但当前应用没有该资源的读取权限。"
    "请将文档授权给机器人/应用，或让管理员开通对应 OpenAPI 只读权限。"
)
RESOURCE_UNSUPPORTED_ERROR = "暂不支持该飞书链接类型；请使用新版 Docx、Sheets 电子表格或 Bitable 多维表格链接。"
URL_RE = re.compile(r"https?://[^\s<>\"]+")


@dataclass(frozen=True)
class FeishuResourceLink:
    resource_type: str
    token: str
    url: str


def extract_feishu_resource_links(text: str, max_count: int | None = None) -> list[FeishuResourceLink]:
    """Extract supported Feishu resource links from text."""
    limit = settings.feishu_link_max_count if max_count is None else max_count
    links: list[FeishuResourceLink] = []
    seen: set[tuple[str, str]] = set()

    for match in URL_RE.finditer(text or ""):
        raw_url = match.group(0).rstrip(").,，。；;]")
        parsed = urlparse(raw_url)
        host = parsed.netloc.lower()
        if not (host.endswith("feishu.cn") or host.endswith("larksuite.com")):
            continue

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            continue

        path_type = parts[0].lower()
        token = parts[1]
        if path_type == "docx":
            resource_type = "docx"
        elif path_type == "sheets":
            resource_type = "sheets"
        elif path_type == "base":
            resource_type = "bitable"
        elif path_type == "docs":
            resource_type = "docs"
        else:
            continue

        key = (resource_type, token)
        if key in seen:
            continue
        seen.add(key)
        links.append(FeishuResourceLink(resource_type=resource_type, token=token, url=raw_url))
        if len(links) >= max(limit, 0):
            break

    return links


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value)


def _format_rows(rows: list[list[Any]], max_rows: int) -> str:
    return "\n".join(
        "\t".join(_stringify_cell(cell) for cell in row)
        for row in rows[:max_rows]
    )


def _format_bitable_records(records: list[dict], max_records: int) -> str:
    lines = []
    for index, record in enumerate(records[:max_records], start=1):
        fields = record.get("fields", record)
        if isinstance(fields, dict):
            field_text = "；".join(f"{key}: {_stringify_cell(value)}" for key, value in fields.items())
        else:
            field_text = _stringify_cell(fields)
        lines.append(f"{index}. {field_text}")
    return "\n".join(lines)


def _extract_docx_raw_content(data: dict) -> str:
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    for key in ("content", "raw_content", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_sheets(data: dict) -> list[dict]:
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    sheets = payload.get("sheets") or payload.get("sheet_infos") or []
    return sheets if isinstance(sheets, list) else []


def _extract_sheet_id(sheet: dict) -> str | None:
    for key in ("sheet_id", "sheetId", "id"):
        value = sheet.get(key)
        if value:
            return str(value)
    return None


def _extract_sheet_title(sheet: dict, fallback: str) -> str:
    for key in ("title", "name"):
        value = sheet.get(key)
        if value:
            return str(value)
    return fallback


def _extract_values(data: dict) -> list[list[Any]]:
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    value_range = payload.get("valueRange") or payload.get("value_range") or payload
    values = value_range.get("values", [])
    return values if isinstance(values, list) else []


def _extract_tables(data: dict) -> list[dict]:
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    items = payload.get("items") or payload.get("tables") or []
    return items if isinstance(items, list) else []


def _extract_table_id(table: dict) -> str | None:
    for key in ("table_id", "tableId", "id"):
        value = table.get(key)
        if value:
            return str(value)
    return None


def _extract_table_name(table: dict, fallback: str) -> str:
    for key in ("name", "title"):
        value = table.get(key)
        if value:
            return str(value)
    return fallback


def _extract_records(data: dict) -> list[dict]:
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    items = payload.get("items") or payload.get("records") or []
    return items if isinstance(items, list) else []


def _is_permission_error(code: int | None, status_code: int) -> bool:
    if status_code in {401, 403}:
        return True
    return code in {99991663, 99991664, 99991668, 99991671, 99991672, 99991673}


async def _summarize_resource_text(text: str, title: str, resource_type: str) -> str:
    if not text.strip():
        return f"{title or '飞书资源'}未读取到可总结的文本内容。"

    prompt = (
        "请基于以下飞书在线资源的只读内容生成简洁中文摘要。"
        "输出包括：1. 内容概述；2. 关键要点；3. 如为表格或多维表格，说明字段/表头和数据主题。"
        "不要编造未出现的信息。\n\n"
        f"资源类型：{resource_type}\n标题：{title}\n\n内容：\n{text}"
    )
    try:
        return await llm_gateway.generate(
            prompt,
            system_prompt="你是严谨的飞书在线资源识别与总结助手。",
            max_tokens=800,
            temperature=0.2,
            allow_mock=False,
        )
    except LLMGatewayError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        selected = "\n".join(lines[:8])
        return f"已读取到资源内容，前 {min(len(lines), 8)} 条要点如下：\n{selected}"


async def summarize_feishu_resource_content(resource: dict) -> dict:
    if not resource.get("success"):
        return resource

    text = resource.get("text") or ""
    sample_text, sample_truncated = _truncate(text, settings.feishu_resource_max_chars)
    summary = await _summarize_resource_text(
        sample_text,
        resource.get("title") or resource.get("token") or "飞书资源",
        resource.get("resource_type") or "feishu",
    )
    warnings = list(resource.get("warnings") or [])
    if sample_truncated:
        warnings.append("内容较长，已截断部分文本用于总结。")

    return {
        **resource,
        "summary": summary,
        "sample_text": sample_text,
        "warnings": warnings,
    }


class FeishuResourceClient:
    """Read Feishu cloud resources with tenant_access_token."""

    def __init__(self, token_provider, base_url: str):
        self._token_provider = token_provider
        self._base_url = base_url.rstrip("/")

    async def read_resource(self, link: FeishuResourceLink) -> dict:
        if link.resource_type == "docs":
            return {
                "success": False,
                "error": RESOURCE_UNSUPPORTED_ERROR,
                "resource_type": link.resource_type,
                "token": link.token,
                "url": link.url,
                "permission_error": False,
            }
        if link.resource_type == "docx":
            return await self._read_docx(link)
        if link.resource_type == "sheets":
            return await self._read_sheets(link)
        if link.resource_type == "bitable":
            return await self._read_bitable(link)
        return {
            "success": False,
            "error": RESOURCE_UNSUPPORTED_ERROR,
            "resource_type": link.resource_type,
            "token": link.token,
            "url": link.url,
            "permission_error": False,
        }

    async def _headers(self) -> dict | None:
        token = await self._token_provider()
        if not token:
            return None
        return {"Authorization": f"Bearer {token}"}

    async def _get_json(self, client: httpx.AsyncClient, path: str, params: dict | None = None) -> dict:
        headers = await self._headers()
        if not headers:
            return {"success": False, "error": "飞书 tenant_access_token 获取失败。", "permission_error": True}

        try:
            resp = await client.get(f"{self._base_url}{path}", headers=headers, params=params)
            data = resp.json()
        except Exception as e:
            logger.warning("读取飞书在线资源异常: %s", e, exc_info=True)
            return {"success": False, "error": f"读取飞书在线资源失败: {e}", "permission_error": False}

        code = data.get("code")
        if resp.status_code == 200 and code == 0:
            return {"success": True, "data": data}
        if _is_permission_error(code, resp.status_code):
            return {"success": False, "error": RESOURCE_PERMISSION_ERROR, "permission_error": True, "code": code}
        msg = data.get("msg") or data.get("message") or resp.text[:300]
        return {"success": False, "error": f"读取飞书在线资源失败: {msg}", "permission_error": False, "code": code}

    async def _read_docx(self, link: FeishuResourceLink) -> dict:
        async with httpx.AsyncClient(timeout=15) as client:
            result = await self._get_json(client, f"/docx/v1/documents/{link.token}/raw_content")
        if not result.get("success"):
            return {**result, "resource_type": "docx", "token": link.token, "url": link.url}

        text = _extract_docx_raw_content(result["data"])
        title = f"Docx {link.token}"
        return {
            "success": True,
            "resource_type": "docx",
            "token": link.token,
            "url": link.url,
            "title": title,
            "outline": ["raw_content"],
            "text": text,
            "warnings": [],
        }

    async def _read_sheets(self, link: FeishuResourceLink) -> dict:
        outline: list[str] = []
        parts: list[str] = []
        warnings: list[str] = []
        async with httpx.AsyncClient(timeout=20) as client:
            sheets_result = await self._get_json(
                client,
                f"/sheets/v3/spreadsheets/{link.token}/sheets/query",
            )
            if not sheets_result.get("success"):
                return {**sheets_result, "resource_type": "sheets", "token": link.token, "url": link.url}

            sheets = _extract_sheets(sheets_result["data"])
            for index, sheet in enumerate(sheets, start=1):
                sheet_id = _extract_sheet_id(sheet)
                title = _extract_sheet_title(sheet, f"Sheet{index}")
                outline.append(title)
                if not sheet_id:
                    continue
                range_name = f"{sheet_id}!A1:Z{settings.feishu_sheet_sample_rows}"
                values_result = await self._get_json(
                    client,
                    f"/sheets/v2/spreadsheets/{link.token}/values/{range_name}",
                )
                if not values_result.get("success"):
                    warnings.append(f"{title} 样例读取失败: {values_result.get('error')}")
                    continue
                values = _extract_values(values_result["data"])
                sample = _format_rows(values, settings.feishu_sheet_sample_rows)
                if sample:
                    parts.append(f"[{title}]\n{sample}")

        return {
            "success": True,
            "resource_type": "sheets",
            "token": link.token,
            "url": link.url,
            "title": f"Sheets {link.token}",
            "outline": outline,
            "text": "\n\n".join(parts),
            "warnings": warnings,
        }

    async def _read_bitable(self, link: FeishuResourceLink) -> dict:
        outline: list[str] = []
        parts: list[str] = []
        warnings: list[str] = []
        async with httpx.AsyncClient(timeout=20) as client:
            tables_result = await self._get_json(client, f"/bitable/v1/apps/{link.token}/tables")
            if not tables_result.get("success"):
                return {**tables_result, "resource_type": "bitable", "token": link.token, "url": link.url}

            tables = _extract_tables(tables_result["data"])
            for index, table in enumerate(tables, start=1):
                table_id = _extract_table_id(table)
                name = _extract_table_name(table, f"Table{index}")
                outline.append(name)
                if not table_id:
                    continue
                records_result = await self._get_json(
                    client,
                    f"/bitable/v1/apps/{link.token}/tables/{table_id}/records",
                    params={"page_size": settings.feishu_bitable_sample_records},
                )
                if not records_result.get("success"):
                    warnings.append(f"{name} 样例读取失败: {records_result.get('error')}")
                    continue
                records = _extract_records(records_result["data"])
                sample = _format_bitable_records(records, settings.feishu_bitable_sample_records)
                if sample:
                    parts.append(f"[{name}]\n{sample}")

        return {
            "success": True,
            "resource_type": "bitable",
            "token": link.token,
            "url": link.url,
            "title": f"Bitable {link.token}",
            "outline": outline,
            "text": "\n\n".join(parts),
            "warnings": warnings,
        }
