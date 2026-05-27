"""工具模块测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.tools.calculator import calculate, extract_math_expression, safe_eval
from app.tools.search import web_search, format_search_results
from app.tools.registry import tool_registry


class TestCalculator:
    """计算器测试"""

    def test_basic_arithmetic(self):
        result = calculate("2 + 3")
        assert result["success"] is True
        assert result["result"] == "5"

    def test_complex_expression(self):
        result = calculate("(100 - 32) * 5 / 9")
        assert result["success"] is True
        assert float(result["result"]) == pytest.approx(37.7778, rel=0.01)

    def test_temperature_conversion(self):
        result = calculate("(100 - 32) * 5 / 9")
        assert result["success"] is True

    def test_empty_expression(self):
        result = calculate("")
        assert result["success"] is False
        assert "空" in result["error"]

    def test_invalid_expression(self):
        result = calculate("import os")
        assert result["success"] is False

    def test_power_expression(self):
        result = calculate("2 ^ 10")
        assert result["success"] is True
        assert result["result"] == "1024"

    def test_comparison(self):
        result = calculate("10 > 5")
        assert result["success"] is True
        assert result["result"] == "True"

    def test_extract_expression_from_natural_language(self):
        assert extract_math_expression("计算 2+3 等于多少") == "2+3"
        assert extract_math_expression("请帮我算 8 乘以 7") == "8 * 7"


class TestWebSearch:
    """联网搜索测试"""

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        with patch("app.tools.search.settings") as mock_settings:
            mock_settings.serper_api_key = None
            result = await web_search("测试")
            assert result["success"] is False
            assert "未配置" in result["error"]

    @pytest.mark.asyncio
    async def test_search_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {"title": "测试标题", "link": "https://example.com", "snippet": "测试摘要"},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("app.tools.search.settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            mock_settings.serper_api_key = "test_key"
            mock_settings.serper_url = "https://google.serper.dev/search"
            mock_settings.search_timeout = 10
            mock_settings.max_retries = 0

            result = await web_search("测试")
            assert result["success"] is True
            assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_search_429(self):
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("app.tools.search.settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            mock_settings.serper_api_key = "test_key"
            mock_settings.serper_url = "https://google.serper.dev/search"
            mock_settings.search_timeout = 10
            mock_settings.max_retries = 0

            result = await web_search("测试")
            assert result["success"] is False
            assert "429" in result["error"]

    @pytest.mark.asyncio
    async def test_search_429_retry_uses_async_sleep(self):
        first_response = MagicMock()
        first_response.status_code = 429
        second_response = MagicMock()
        second_response.status_code = 200
        second_response.json.return_value = {
            "organic": [
                {"title": "测试标题", "link": "https://example.com", "snippet": "测试摘要"},
            ],
        }
        second_response.raise_for_status = MagicMock()

        with patch("app.tools.search.settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=[first_response, second_response])), \
             patch("app.tools.search.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            mock_settings.serper_api_key = "test_key"
            mock_settings.serper_url = "https://google.serper.dev/search"
            mock_settings.search_timeout = 10
            mock_settings.max_retries = 1

            result = await web_search("测试")

        assert result["success"] is True
        mock_sleep.assert_awaited_once_with(1)

    @pytest.mark.asyncio
    async def test_search_401(self):
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("app.tools.search.settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            mock_settings.serper_api_key = "invalid_key"
            mock_settings.serper_url = "https://google.serper.dev/search"
            mock_settings.search_timeout = 10
            mock_settings.max_retries = 0

            result = await web_search("测试")
            assert result["success"] is False
            assert "401" in result["error"]


def test_format_search_results():
    results = [
        {"title": "标题1", "snippet": "摘要1", "url": "https://example1.com"},
        {"title": "标题2", "snippet": "摘要2", "url": ""},
    ]
    text = format_search_results(results)
    assert "标题1" in text
    assert "摘要1" in text
    assert "来源: https://example1.com" in text


class TestToolRegistry:
    """工具注册表测试"""

    @pytest.mark.asyncio
    async def test_calculator_tool(self):
        trace = await tool_registry.execute("calculator", {"expression": "2 + 3"})
        assert trace.status == "success"
        assert "5" in trace.output_preview

    @pytest.mark.asyncio
    async def test_execute_with_result_returns_trace_and_result(self):
        execution = await tool_registry.execute_with_result("calculator", {"expression": "2 + 3"})
        assert execution.trace.status == "success"
        assert execution.result["success"] is True
        assert execution.result["result"] == "5"

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        trace = await tool_registry.execute("unknown_tool", {})
        assert trace.status == "error"
        assert "未知工具" in trace.error_message

    def test_get_available_tools(self):
        tools = tool_registry.get_available_tools()
        assert "calculator" in tools
        assert "web_search" in tools
