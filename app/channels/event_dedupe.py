"""Feishu event dedupe storage."""

import time
from typing import Protocol

from app.observability.logging import get_logger
from app.redis_client import RedisFallbackState, create_redis_client, redis_key

logger = get_logger(__name__)


class EventDedupeStore(Protocol):
    async def mark_seen(self, dedupe_key: str, ttl_seconds: int) -> bool:
        ...

    async def health(self) -> dict:
        ...


class InMemoryEventDedupeStore:
    def __init__(
        self,
        storage: dict[str, float] | None = None,
        *,
        degraded: bool = False,
        fallback_reason: str | None = None,
    ):
        self._storage = storage if storage is not None else {}
        self._degraded = degraded
        self._fallback_reason = fallback_reason

    async def mark_seen(self, dedupe_key: str, ttl_seconds: int) -> bool:
        now = time.time()
        expired_keys = [
            key for key, expires_at in self._storage.items()
            if expires_at <= now
        ]
        for key in expired_keys:
            self._storage.pop(key, None)

        if dedupe_key in self._storage:
            return False

        self._storage[dedupe_key] = now + ttl_seconds
        return True

    async def health(self) -> dict:
        return {
            "backend": "memory",
            "fallback_active": self._degraded,
            "degraded": self._degraded,
            "fallback_reason": self._fallback_reason,
            "local_key_count": len(self._storage),
        }


class RedisEventDedupeStore:
    def __init__(self, redis_client):
        self._redis = redis_client

    def _key(self, dedupe_key: str) -> str:
        return redis_key("feishu_dedupe", dedupe_key)

    async def mark_seen(self, dedupe_key: str, ttl_seconds: int) -> bool:
        result = await self._redis.set(
            self._key(dedupe_key),
            "1",
            nx=True,
            ex=max(int(ttl_seconds), 1),
        )
        return bool(result)

    async def health(self) -> dict:
        ping = getattr(self._redis, "ping", None)
        if ping:
            await ping()
        return {
            "backend": "redis",
            "fallback_active": False,
            "degraded": False,
        }


class FallbackEventDedupeStore:
    def __init__(self, primary: EventDedupeStore, fallback: InMemoryEventDedupeStore):
        self._primary = primary
        self._fallback = fallback
        self._state = RedisFallbackState(
            "feishu_event_dedupe",
            "Redis Feishu event dedupe unavailable; falling back to in-memory store",
            "Redis Feishu event dedupe recovered; using Redis backend",
        )

    def _record_primary_failure(self, error: Exception) -> None:
        self._state.record_failure(
            error,
            dedupe_backend="memory",
            dedupe_degraded=True,
        )

    async def mark_seen(self, dedupe_key: str, ttl_seconds: int) -> bool:
        fallback_first_seen = await self._fallback.mark_seen(dedupe_key, ttl_seconds)
        if not self._state.should_try_primary():
            return fallback_first_seen
        try:
            primary_first_seen = await self._primary.mark_seen(dedupe_key, ttl_seconds)
            self._state.record_success()
            return primary_first_seen and fallback_first_seen
        except Exception as e:
            self._record_primary_failure(e)
            return fallback_first_seen

    async def health(self) -> dict:
        if self._state.should_try_primary():
            try:
                status = await self._primary.health()
                self._state.record_success()
                return self._state.redis_health(status)
            except Exception as e:
                self._record_primary_failure(e)

        fallback_status = await self._fallback.health()
        return self._state.fallback_health(fallback_status)


def get_event_dedupe_store(storage: dict[str, float] | None = None) -> EventDedupeStore:
    fallback = InMemoryEventDedupeStore(storage)
    try:
        client = create_redis_client()
        return FallbackEventDedupeStore(RedisEventDedupeStore(client), fallback)
    except Exception as e:
        logger.warning(
            "Redis Feishu event dedupe cannot be initialized; using in-memory store",
            extra={
                "component": "feishu_event_dedupe",
                "storage_backend": "memory",
                "dedupe_backend": "memory",
                "dedupe_degraded": True,
                "storage_degraded": True,
                "fallback_reason": "redis_init_failed",
                "error": str(e),
            },
        )
        return InMemoryEventDedupeStore(
            storage,
            degraded=True,
            fallback_reason="redis_init_failed",
        )
