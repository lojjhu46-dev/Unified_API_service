"""LLM Gateway测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from openai import AuthenticationError, APITimeoutError
from app.llm.gateway import LLMGateway, LLMGatewayError
from app.llm.prompts import build_rag_prompt
from app.config import settings


@pytest.fixture
def mock_settings():
    """模拟配置"""
    with patch("app.llm.gateway.settings") as mock:
        yield mock


@pytest.mark.asyncio
async def test_mock_generate():
    gateway = LLMGateway()
    gateway.provider = "mock"
    result = await gateway.generate("测试问题")
    assert "[MOCK回答]" in result
    assert "测试问题" in result


@pytest.mark.asyncio
async def test_deepseek_no_api_key(mock_settings):
    mock_settings.llm_provider = "deepseek"
    mock_settings.deepseek_api_key = None
    mock_settings.deepseek_base_url = "https://api.deepseek.com"
    mock_settings.llm_timeout_seconds = 30

    gateway = LLMGateway()
    gateway.provider = "deepseek"
    gateway._client = None

    with pytest.raises(LLMGatewayError, match="未配置DEEPSEEK_API_KEY"):
        await gateway.generate("测试问题")


@pytest.mark.asyncio
async def test_deepseek_authentication_error(mock_settings):
    mock_settings.llm_provider = "deepseek"
    mock_settings.deepseek_api_key = "invalid_key"
    mock_settings.deepseek_base_url = "https://api.deepseek.com"
    mock_settings.deepseek_model = "deepseek-chat"
    mock_settings.llm_timeout_seconds = 30

    gateway = LLMGateway()
    gateway.provider = "deepseek"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=AuthenticationError("Invalid API key", response=MagicMock(status_code=401), body=None)
    )
    gateway._client = mock_client

    with pytest.raises(LLMGatewayError, match="API Key无效或已过期"):
        await gateway.generate("测试问题")


@pytest.mark.asyncio
async def test_deepseek_timeout_error(mock_settings):
    mock_settings.llm_provider = "deepseek"
    mock_settings.deepseek_api_key = "test_key"
    mock_settings.deepseek_base_url = "https://api.deepseek.com"
    mock_settings.deepseek_model = "deepseek-chat"
    mock_settings.llm_timeout_seconds = 30

    gateway = LLMGateway()
    gateway.provider = "deepseek"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=APITimeoutError(request=MagicMock())
    )
    gateway._client = mock_client

    with pytest.raises(LLMGatewayError, match="请求超时"):
        await gateway.generate("测试问题")


@pytest.mark.asyncio
async def test_deepseek_empty_response(mock_settings):
    mock_settings.llm_provider = "deepseek"
    mock_settings.deepseek_api_key = "test_key"
    mock_settings.deepseek_base_url = "https://api.deepseek.com"
    mock_settings.deepseek_model = "deepseek-chat"
    mock_settings.llm_timeout_seconds = 30

    gateway = LLMGateway()
    gateway.provider = "deepseek"

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=None))]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    gateway._client = mock_client

    result = await gateway.generate("测试问题")
    assert result == "抱歉，无法生成回答。"


@pytest.mark.asyncio
async def test_deepseek_success(mock_settings):
    mock_settings.llm_provider = "deepseek"
    mock_settings.deepseek_api_key = "test_key"
    mock_settings.deepseek_base_url = "https://api.deepseek.com"
    mock_settings.deepseek_model = "deepseek-chat"
    mock_settings.llm_timeout_seconds = 30

    gateway = LLMGateway()
    gateway.provider = "deepseek"

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="测试回答"))]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    gateway._client = mock_client

    result = await gateway.generate("测试问题")
    assert result == "测试回答"


@pytest.mark.asyncio
async def test_unsupported_provider():
    gateway = LLMGateway()
    gateway.provider = "unsupported"

    with pytest.raises(LLMGatewayError, match="不支持的LLM提供商"):
        await gateway.generate("测试问题")


def test_provider_reads_current_settings(mock_settings):
    mock_settings.llm_provider = "mock"
    gateway = LLMGateway()
    assert gateway.provider == "mock"

    mock_settings.llm_provider = "deepseek"
    assert gateway.provider == "deepseek"


def test_global_search_prompt_has_separator():
    system_prompt, _user_prompt = build_rag_prompt("列出所有产品", "片段", is_global=True)
    assert "准确。\n\n本轮是全局汇总任务" in system_prompt
