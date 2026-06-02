"""Feishu tenant access token cache storage."""

import time
from typing import Protocol

from app.observability.logging import get_logger
from app.redis_client import RedisFallbackState, create_redis_client, redis_key

logger = get_logger(__name__)


class FeishuTokenCache(Protocol):
    async def get(self) -> str | None:
        ...

    async def set(self, token: str, ttl_seconds: int) -> None:
        ...

    async def clear(self) -> None:
        ...

    async def health(self) -> dict:
        ...


class InMemoryFeishuTokenCache:
    def __init__(self):
        self._token: str | None = None
        self._expires_at: float = 0

    async def get(self) -> str | None:
        if self._token and time.time() < self._expires_at:
            return self._token
        self._token = None
        self._expires_at = 0
        return None

    async def set(self, token: str, ttl_seconds: int) -> None:
        self._token = token
        self._expires_at = time.time() + max(int(ttl_seconds), 1)

    async def clear(self) -> None:
        self._token = None
        self._expires_at = 0

    async def health(self) -> dict:
        return {
            "backend": "memory",
            "fallback_active": False,
            "degraded": False,
            "has_token": bool(await self.get()),
        }


class RedisFeishuTokenCache:
    def __init__(self, redis_client):
        self._redis = redis_client

    def _key(self) -> str:
        return redis_key("feishu_token", "tenant_access_token")

    async def get(self) -> str | None:
        return await self._redis.get(self._key())

    async def set(self, token: str, ttl_seconds: int) -> None:
        await self._redis.set(self._key(), token, ex=max(int(ttl_seconds), 1))

    async def clear(self) -> None:
        await self._redis.delete(self._key())

    async def health(self) -> dict:
        ping = getattr(self._redis, "ping", None)
        if ping:
            await ping()
        return {
            "backend": "redis",
            "fallback_active": False,
            "degraded": False,
        }


class FallbackFeishuTokenCache:
    def __init__(self, primary: FeishuTokenCache, fallback: InMemoryFeishuTokenCache):
        self._primary = primary
        self._fallback = fallback
        self._state = RedisFallbackState(
            "feishu_token_cache",
            "Redis Feishu token cache unavailable; falling back to in-memory cache",
            "Redis Feishu token cache recovered; using Redis backend",
        )

    def _record_primary_failure(self, error: Exception) -> None:
        self._state.record_failure(error)

    async def _try_primary(self, method_name: str, *args):
        if not self._state.should_try_primary():
            return None
        try:
            method = getattr(self._primary, method_name)
            result = await method(*args)
            self._state.record_success()
            return result
        except Exception as e:
            self._record_primary_failure(e)
            return None

    async def get(self) -> str | None:
        token = await self._try_primary("get")
        return token or await self._fallback.get()

    async def set(self, token: str, ttl_seconds: int) -> None:
        await self._try_primary("set", token, ttl_seconds)
        await self._fallback.set(token, ttl_seconds)

    async def clear(self) -> None:
        await self._try_primary("clear")
        await self._fallback.clear()

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


def get_feishu_token_cache() -> FeishuTokenCache:
    fallback = InMemoryFeishuTokenCache()
    try:
        return FallbackFeishuTokenCache(RedisFeishuTokenCache(create_redis_client()), fallback)
    except Exception as e:
        logger.warning(
            "Redis Feishu token cache cannot be initialized; using in-memory cache",
            extra={
                "component": "feishu_token_cache",
                "storage_backend": "memory",
                "storage_degraded": True,
                "fallback_reason": "redis_init_failed",
                "error": str(e),
            },
        )
        return fallback
