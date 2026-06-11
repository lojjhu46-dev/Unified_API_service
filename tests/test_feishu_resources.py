from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.channels.feishu_resources import (
    FeishuResourceClient,
    FeishuResourceLink,
    extract_feishu_resource_links,
)


def _response(payload: dict, status_code: int = 200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = str(payload)
    return response


def test_extract_feishu_resource_links_supports_docx_sheets_and_bitable():
    text = (
        "文档 https://abc.feishu.cn/docx/docx_token "
        "表格 https://abc.feishu.cn/sheets/sheet_token "
        "多维表 https://abc.feishu.cn/base/base_token"
    )

    links = extract_feishu_resource_links(text)

    assert [(link.resource_type, link.token) for link in links] == [
        ("docx", "docx_token"),
        ("sheets", "sheet_token"),
        ("bitable", "base_token"),
    ]


def test_extract_feishu_resource_links_ignores_non_feishu_and_limits_count():
    text = (
        "https://example.com/docx/a "
        "https://abc.feishu.cn/docx/one "
        "https://abc.feishu.cn/docx/two "
        "https://abc.feishu.cn/docx/three"
    )

    links = extract_feishu_resource_links(text, max_count=2)

    assert [link.token for link in links] == ["one", "two"]


def test_extract_feishu_resource_links_marks_legacy_docs_as_docs():
    links = extract_feishu_resource_links("https://abc.feishu.cn/docs/legacy_token")

    assert len(links) == 1
    assert links[0].resource_type == "docs"


@pytest.mark.asyncio
async def test_read_docx_raw_content_success():
    client = FeishuResourceClient(AsyncMock(return_value="tenant_token"), "https://open.feishu.cn/open-apis")
    mock_get = AsyncMock(return_value=_response({"code": 0, "data": {"content": "正文内容"}}))

    with patch("httpx.AsyncClient.get", new=mock_get):
        result = await client.read_resource(
            FeishuResourceLink("docx", "docx_token", "https://abc.feishu.cn/docx/docx_token")
        )

    assert result["success"] is True
    assert result["resource_type"] == "docx"
    assert result["text"] == "正文内容"
    assert mock_get.await_args.args[0].endswith("/docx/v1/documents/docx_token/raw_content")


@pytest.mark.asyncio
async def test_read_sheets_queries_sheet_list_and_sample_values():
    client = FeishuResourceClient(AsyncMock(return_value="tenant_token"), "https://open.feishu.cn/open-apis")
    mock_get = AsyncMock(side_effect=[
        _response({"code": 0, "data": {"sheets": [{"sheet_id": "sid1", "title": "Sheet1"}]}}),
        _response({"code": 0, "data": {"valueRange": {"values": [["姓名", "分数"], ["张三", 90]]}}}),
    ])

    with patch("httpx.AsyncClient.get", new=mock_get):
        result = await client.read_resource(
            FeishuResourceLink("sheets", "sheet_token", "https://abc.feishu.cn/sheets/sheet_token")
        )

    assert result["success"] is True
    assert result["outline"] == ["Sheet1"]
    assert "张三" in result["text"]
    assert mock_get.await_args_list[0].args[0].endswith("/sheets/v3/spreadsheets/sheet_token/sheets/query")
    assert mock_get.await_args_list[1].args[0].endswith("/sheets/v2/spreadsheets/sheet_token/values/sid1!A1:Z30")


@pytest.mark.asyncio
async def test_read_bitable_tables_and_records():
    client = FeishuResourceClient(AsyncMock(return_value="tenant_token"), "https://open.feishu.cn/open-apis")
    mock_get = AsyncMock(side_effect=[
        _response({"code": 0, "data": {"items": [{"table_id": "tbl1", "name": "需求表"}]}}),
        _response({"code": 0, "data": {"items": [{"fields": {"标题": "任务A", "状态": "进行中"}}]}}),
    ])

    with patch("httpx.AsyncClient.get", new=mock_get):
        result = await client.read_resource(
            FeishuResourceLink("bitable", "base_token", "https://abc.feishu.cn/base/base_token")
        )

    assert result["success"] is True
    assert result["outline"] == ["需求表"]
    assert "任务A" in result["text"]
    assert mock_get.await_args_list[1].kwargs["params"] == {"page_size": 50}


@pytest.mark.asyncio
async def test_read_resource_permission_error_is_structured():
    client = FeishuResourceClient(AsyncMock(return_value="tenant_token"), "https://open.feishu.cn/open-apis")

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=_response({"code": 99991671, "msg": "forbidden"}, 403))):
        result = await client.read_resource(
            FeishuResourceLink("docx", "docx_token", "https://abc.feishu.cn/docx/docx_token")
        )

    assert result["success"] is False
    assert result["permission_error"] is True
    assert "没有该资源的读取权限" in result["error"]


@pytest.mark.asyncio
async def test_read_resource_retries_transient_timeout():
    client = FeishuResourceClient(AsyncMock(return_value="tenant_token"), "https://open.feishu.cn/open-apis")
    get = AsyncMock(side_effect=[
        httpx.ConnectTimeout("timeout"),
        _response({"code": 0, "data": {"content": "正文内容"}}),
    ])

    with patch("httpx.AsyncClient.get", new=get), patch("asyncio.sleep", new=AsyncMock()) as sleep:
        result = await client.read_resource(
            FeishuResourceLink("docx", "docx_token", "https://abc.feishu.cn/docx/docx_token")
        )

    assert result["success"] is True
    assert result["text"] == "正文内容"
    assert get.await_count == 2
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_resource_non_json_error_is_structured():
    client = FeishuResourceClient(AsyncMock(return_value="tenant_token"), "https://open.feishu.cn/open-apis")
    response = MagicMock()
    response.status_code = 502
    response.text = "bad gateway"
    response.json.side_effect = ValueError("not json")

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=response)):
        result = await client.read_resource(
            FeishuResourceLink("docx", "docx_token", "https://abc.feishu.cn/docx/docx_token")
        )

    assert result["success"] is False
    assert result["permission_error"] is False
    assert "非 JSON 响应" in result["error"]
