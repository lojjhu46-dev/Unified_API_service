"""会话记忆测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.memory.store import InMemoryStore, RedisStore, get_memory_store, memory_store
from app.memory.rewrite import rewrite_question, _history_to_text


@pytest.fixture
def memory():
    return InMemoryStore()


@pytest.mark.asyncio
async def test_create_session(memory):
    session_id = await memory.create_session()
    assert session_id is not None
    assert len(session_id) == 12


@pytest.mark.asyncio
async def test_append_and_get_history(memory):
    session_id = await memory.create_session()
    await memory.append_turn(session_id, "问题1", "回答1")
    await memory.append_turn(session_id, "问题2", "回答2")

    history = await memory.get_history(session_id)
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "问题1"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "回答1"


@pytest.mark.asyncio
async def test_max_messages(memory):
    session_id = await memory.create_session()
    for i in range(10):
        await memory.append_turn(session_id, f"问题{i}", f"回答{i}")

    history = await memory.get_history(session_id, max_messages=4)
    assert len(history) == 4


@pytest.mark.asyncio
async def test_clear_session(memory):
    session_id = await memory.create_session()
    await memory.append_turn(session_id, "问题", "回答")

    await memory.clear_session(session_id)
    history = await memory.get_history(session_id)
    assert len(history) == 0


@pytest.mark.asyncio
async def test_session_exists(memory):
    session_id = await memory.create_session()
    assert await memory.session_exists(session_id) is True
    assert await memory.session_exists("nonexistent") is False


def test_history_to_text():
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    text = _history_to_text(history)
    assert "用户: 你好" in text
    assert "助手: 你好！" in text


def test_history_to_text_empty():
    text = _history_to_text([])
    assert "无历史对话" in text


@pytest.mark.asyncio
async def test_rewrite_no_history():
    result = await rewrite_question("什么是机器学习", [])
    assert result == "什么是机器学习"


@pytest.mark.asyncio
async def test_rewrite_with_history():
    with patch("app.memory.rewrite.llm_gateway") as mock_llm:
        mock_llm.provider = "deepseek"
        mock_llm.generate = AsyncMock(return_value="iPhone 15的价格是多少")
        result = await rewrite_question(
            "它的价格呢",
            [
                {"role": "user", "content": "iPhone 15有什么特点"},
                {"role": "assistant", "content": "iPhone 15特点..."},
            ],
        )
        assert "iPhone 15" in result
        assert "价格" in result


@pytest.mark.asyncio
async def test_rewrite_mock_provider_returns_original_question():
    with patch("app.memory.rewrite.llm_gateway") as mock_llm:
        mock_llm.provider = "mock"
        mock_llm.generate = AsyncMock(return_value="[MOCK回答] 污染文本")
        result = await rewrite_question(
            "它的价格呢",
            [{"role": "user", "content": "iPhone 15有什么特点"}],
        )

    assert result == "它的价格呢"
    mock_llm.generate.assert_not_called()


def test_get_memory_store_defaults_to_in_memory():
    assert isinstance(get_memory_store(), InMemoryStore)


def test_redis_store_requires_explicit_async_enablement():
    with pytest.raises(RuntimeError):
        RedisStore()
