"""会话记忆测试"""

import pytest
from unittest.mock import AsyncMock, patch
from app.memory.store import FallbackMemoryStore, InMemoryStore, RedisStore, get_memory_store, memory_store
from app.memory.rewrite import rewrite_question, _history_to_text
from app.redis_client import create_redis_client, reset_redis_client


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
    assert isinstance(get_memory_store(), FallbackMemoryStore)


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.values = {}
        self.lists = {}
        self.expirations = {}

    def _maybe_fail(self):
        if self.fail:
            raise RuntimeError("redis down")

    async def set(self, key, value, ex=None):
        self._maybe_fail()
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex

    async def rpush(self, key, *values):
        self._maybe_fail()
        self.lists.setdefault(key, []).extend(values)

    async def ltrim(self, key, start, stop):
        self._maybe_fail()
        values = self.lists.get(key, [])
        self.lists[key] = values[start:] if stop == -1 else values[start:stop + 1]

    async def expire(self, key, seconds):
        self._maybe_fail()
        self.expirations[key] = seconds

    async def lrange(self, key, start, stop):
        self._maybe_fail()
        values = self.lists.get(key, [])
        return values[start:] if stop == -1 else values[start:stop + 1]

    async def delete(self, *keys):
        self._maybe_fail()
        for key in keys:
            self.values.pop(key, None)
            self.lists.pop(key, None)

    async def exists(self, key):
        self._maybe_fail()
        return int(key in self.values or key in self.lists)

    async def ping(self):
        self._maybe_fail()
        return True


@pytest.mark.asyncio
async def test_redis_store_append_get_clear_and_ttl():
    redis = FakeRedis()
    store = RedisStore(redis)
    session_id = await store.create_session()

    await store.append_turn(session_id, "问题1", "回答1")
    await store.append_turn(session_id, "问题2", "回答2")

    history = await store.get_history(session_id)
    assert [item["content"] for item in history] == ["问题1", "回答1", "问题2", "回答2"]
    session_key = store._session_key(session_id)
    meta_key = store._session_meta_key(session_id)
    assert session_key in redis.expirations
    assert meta_key in redis.expirations
    assert await store.session_exists(session_id) is True

    await store.clear_session(session_id)
    assert await store.get_history(session_id) == []
    assert await store.session_exists(session_id) is False


@pytest.mark.asyncio
async def test_redis_store_trims_to_memory_max_messages():
    redis = FakeRedis()
    store = RedisStore(redis)
    session_id = await store.create_session()

    for index in range(12):
        await store.append_turn(session_id, f"问题{index}", f"回答{index}")

    history = await store.get_history(session_id, max_messages=30)
    assert len(history) == 20
    assert history[0]["content"] == "问题2"


@pytest.mark.asyncio
async def test_redis_store_can_read_existing_fake_redis_after_rebuild():
    redis = FakeRedis()
    first = RedisStore(redis)
    session_id = await first.create_session()
    await first.append_turn(session_id, "问题", "回答")

    second = RedisStore(redis)

    assert await second.get_history(session_id) == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]


@pytest.mark.asyncio
async def test_fallback_memory_store_uses_memory_when_redis_fails():
    store = FallbackMemoryStore(RedisStore(FakeRedis(fail=True)), InMemoryStore())

    session_id = await store.create_session()
    await store.append_turn(session_id, "问题", "回答")

    assert await store.get_history(session_id) == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]
    health = await store.health()
    assert health["backend"] == "memory"
    assert health["degraded"] is True


@pytest.mark.asyncio
async def test_fallback_memory_store_retries_and_recovers_after_cooldown():
    redis = FakeRedis(fail=True)
    store = FallbackMemoryStore(RedisStore(redis), InMemoryStore())

    with patch("app.redis_client.settings.redis_retry_cooldown_seconds", 0):
        await store.append_turn("session_1", "fallback question", "fallback answer")
        failed_health = await store.health()
        assert failed_health["backend"] == "memory"
        assert failed_health["next_retry_at"] is not None

        redis.fail = False
        await store.append_turn("session_1", "redis question", "redis answer")
        recovered_health = await store.health()

    assert recovered_health["backend"] == "redis"
    history = await store.get_history("session_1")
    assert history[-2]["content"] == "redis question"
    assert history[-1]["content"] == "redis answer"


def test_create_redis_client_reuses_shared_client_and_keeps_configured_connect_timeout():
    reset_redis_client()
    with patch("redis.asyncio.from_url") as mock_from_url, \
         patch("app.redis_client.settings.redis_socket_timeout", 7), \
         patch("app.redis_client.settings.redis_connect_timeout", 7), \
         patch("app.redis_client.settings.redis_health_check_interval", 15):
        mock_from_url.return_value = object()
        first = create_redis_client()
        second = create_redis_client()

    try:
        assert first is second
        assert mock_from_url.call_count == 1
        kwargs = mock_from_url.call_args.kwargs
        assert kwargs["socket_timeout"] == 7
        assert kwargs["socket_connect_timeout"] == 7
        assert kwargs["retry_on_timeout"] is True
        assert kwargs["health_check_interval"] == 15
        assert kwargs["socket_keepalive"] is True
    finally:
        reset_redis_client()
