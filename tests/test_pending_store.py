"""Pending Feishu file store tests."""

import json
import time

import pytest

from app.channels.pending_store import FallbackPendingFileStore, InMemoryPendingFileStore, RedisPendingFileStore


@pytest.mark.asyncio
async def test_in_memory_pending_store_create_get_delete():
    store = InMemoryPendingFileStore()
    pending = {"open_id": "ou_owner", "expires_at": time.time() + 60}

    await store.create("pending_1", pending, ttl_seconds=60)

    assert await store.get("pending_1") == pending
    await store.delete("pending_1")
    assert await store.get("pending_1") is None


@pytest.mark.asyncio
async def test_in_memory_pending_store_removes_expired_pending():
    backing = {}
    store = InMemoryPendingFileStore(backing)
    await store.create("pending_1", {"expires_at": 0}, ttl_seconds=60)

    assert await store.get("pending_1") is None
    assert "pending_1" not in backing


@pytest.mark.asyncio
async def test_in_memory_pending_store_consumes_once():
    store = InMemoryPendingFileStore()
    pending = {"open_id": "ou_owner", "expires_at": time.time() + 60}
    await store.create("pending_1", pending, ttl_seconds=60)

    assert await store.consume("pending_1") == pending
    assert await store.consume("pending_1") is None


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    async def set(self, key, value, ex):
        self.values[key] = value
        self.expirations[key] = ex

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)

    async def eval(self, _script, _numkeys, key):
        return self.values.pop(key, None)

    async def ping(self):
        return True


@pytest.mark.asyncio
async def test_redis_pending_store_uses_ttl_and_json_payload():
    redis = FakeRedis()
    store = RedisPendingFileStore(redis)
    pending = {"open_id": "ou_owner", "expires_at": time.time() + 60}

    await store.create("pending_1", pending, ttl_seconds=30)

    key = next(iter(redis.values))
    assert key.endswith(":feishu_pending:pending_1")
    assert redis.expirations[key] == 30
    assert json.loads(redis.values[key]) == pending


@pytest.mark.asyncio
async def test_redis_pending_store_consumes_once():
    redis = FakeRedis()
    store = RedisPendingFileStore(redis)
    pending = {"open_id": "ou_owner", "expires_at": time.time() + 60}
    await store.create("pending_1", pending, ttl_seconds=30)

    assert await store.consume("pending_1") == pending
    assert await store.consume("pending_1") is None


class RecoveringPendingStore:
    def __init__(self):
        self.fail = True
        self.values = {}

    def _maybe_fail(self):
        if self.fail:
            raise RuntimeError("redis down")

    async def create(self, pending_id: str, pending: dict, ttl_seconds: int) -> None:
        self._maybe_fail()
        self.values[pending_id] = dict(pending)

    async def get(self, pending_id: str) -> dict | None:
        self._maybe_fail()
        return self.values.get(pending_id)

    async def delete(self, pending_id: str) -> None:
        self._maybe_fail()
        self.values.pop(pending_id, None)

    async def consume(self, pending_id: str) -> dict | None:
        self._maybe_fail()
        return self.values.pop(pending_id, None)

    async def health(self) -> dict:
        self._maybe_fail()
        return {"backend": "redis", "fallback_active": False, "degraded": False}


@pytest.mark.asyncio
async def test_fallback_pending_store_retries_and_recovers_after_cooldown():
    primary = RecoveringPendingStore()
    fallback = InMemoryPendingFileStore()
    store = FallbackPendingFileStore(primary, fallback)
    pending = {"open_id": "ou_owner", "expires_at": time.time() + 60}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("app.redis_client.settings.redis_retry_cooldown_seconds", 0)
        await store.create("pending_1", pending, ttl_seconds=60)
        failed_health = await store.health()
        assert failed_health["backend"] == "memory"
        assert failed_health["next_retry_at"] is not None

        primary.fail = False
        await store.create("pending_2", pending, ttl_seconds=60)
        recovered_health = await store.health()

    assert recovered_health["backend"] == "redis"
    assert await store.get("pending_2") == pending
