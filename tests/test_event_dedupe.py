"""Feishu event dedupe store tests."""

import pytest

from app.channels.event_dedupe import (
    FallbackEventDedupeStore,
    InMemoryEventDedupeStore,
    RedisEventDedupeStore,
)


@pytest.mark.asyncio
async def test_in_memory_event_dedupe_first_seen_then_duplicate():
    store = InMemoryEventDedupeStore()

    assert await store.mark_seen("message_1", ttl_seconds=60) is True
    assert await store.mark_seen("message_1", ttl_seconds=60) is False


@pytest.mark.asyncio
async def test_in_memory_event_dedupe_expired_key_can_be_seen_again():
    store = InMemoryEventDedupeStore()

    assert await store.mark_seen("message_1", ttl_seconds=-1) is True
    assert await store.mark_seen("message_1", ttl_seconds=60) is True


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.calls = []

    async def set(self, key, value, nx=False, ex=None):
        self.calls.append({"key": key, "value": value, "nx": nx, "ex": ex})
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True


@pytest.mark.asyncio
async def test_redis_event_dedupe_uses_set_nx_ex_and_prefixed_key():
    redis = FakeRedis()
    store = RedisEventDedupeStore(redis)

    assert await store.mark_seen("message_1", ttl_seconds=30) is True
    assert await store.mark_seen("message_1", ttl_seconds=30) is False

    first_call = redis.calls[0]
    assert first_call["key"].endswith(":feishu_dedupe:message_1")
    assert first_call["value"] == "1"
    assert first_call["nx"] is True
    assert first_call["ex"] == 30


class FailingRedisStore:
    async def mark_seen(self, dedupe_key: str, ttl_seconds: int) -> bool:
        raise RuntimeError("redis down")


@pytest.mark.asyncio
async def test_fallback_event_dedupe_uses_memory_when_redis_fails():
    fallback = InMemoryEventDedupeStore()
    store = FallbackEventDedupeStore(FailingRedisStore(), fallback)

    assert await store.mark_seen("message_1", ttl_seconds=60) is True
    assert await store.mark_seen("message_1", ttl_seconds=60) is False


class RecoveringRedisStore:
    def __init__(self):
        self.fail = True
        self.seen = set()

    async def mark_seen(self, dedupe_key: str, ttl_seconds: int) -> bool:
        if self.fail:
            raise RuntimeError("redis down")
        if dedupe_key in self.seen:
            return False
        self.seen.add(dedupe_key)
        return True

    async def health(self) -> dict:
        if self.fail:
            raise RuntimeError("redis down")
        return {"backend": "redis", "fallback_active": False, "degraded": False}


@pytest.mark.asyncio
async def test_fallback_event_dedupe_retries_and_recovers_after_cooldown():
    primary = RecoveringRedisStore()
    store = FallbackEventDedupeStore(primary, InMemoryEventDedupeStore())

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("app.redis_client.settings.redis_retry_cooldown_seconds", 0)
        assert await store.mark_seen("message_1", ttl_seconds=60) is True
        failed_health = await store.health()
        assert failed_health["backend"] == "memory"
        assert failed_health["next_retry_at"] is not None

        primary.fail = False
        assert await store.mark_seen("message_2", ttl_seconds=60) is True
        recovered_health = await store.health()

    assert recovered_health["backend"] == "redis"
    assert recovered_health["degraded"] is False
