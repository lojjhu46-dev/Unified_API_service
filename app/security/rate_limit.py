"""Lightweight fixed-window rate limiting."""

import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from app.config import settings
from app.observability.logging import get_logger
from app.redis_client import RedisFallbackState, create_redis_client
from app.security.auth import AuthContext

logger = get_logger(__name__)


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after: int = 60


class InMemoryRateLimitBackend:
    def __init__(self):
        self._counters: dict[str, tuple[int, float]] = {}

    async def increment(self, key: str, window_seconds: int) -> int:
        now = time.time()
        count, expires_at = self._counters.get(key, (0, now + window_seconds))
        if expires_at <= now:
            count = 0
            expires_at = now + window_seconds
        count += 1
        self._counters[key] = (count, expires_at)
        return count

    def clear(self) -> None:
        self._counters.clear()

    async def health(self) -> dict:
        return {
            "backend": "memory",
            "fallback_active": False,
            "degraded": False,
            "local_key_count": len(self._counters),
        }


class RedisRateLimitBackend:
    def __init__(self, redis_client):
        self._redis = redis_client

    async def increment(self, key: str, window_seconds: int) -> int:
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, window_seconds)
        return int(count)

    async def health(self) -> dict:
        ping = getattr(self._redis, "ping", None)
        if ping:
            await ping()
        return {
            "backend": "redis",
            "fallback_active": False,
            "degraded": False,
        }


class FallbackRateLimitBackend:
    def __init__(self, primary: RedisRateLimitBackend, fallback: InMemoryRateLimitBackend):
        self._primary = primary
        self._fallback = fallback
        self._state = RedisFallbackState(
            "rate_limiter",
            "Redis rate limit backend unavailable; falling back to in-memory limiter",
            "Redis rate limit backend recovered; using Redis backend",
        )

    def _record_primary_failure(self, error: Exception) -> None:
        self._state.record_failure(error)

    async def increment(self, key: str, window_seconds: int) -> int:
        if self._state.should_try_primary():
            try:
                count = await self._primary.increment(key, window_seconds)
                self._state.record_success()
                return count
            except Exception as e:
                self._record_primary_failure(e)
        return await self._fallback.increment(key, window_seconds)

    def clear(self) -> None:
        self._fallback.clear()

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


class RateLimiter:
    def __init__(self, backend):
        self._backend = backend

    def _bucket_key(self, scope: str, identity: str, window_name: str, window_seconds: int) -> str:
        bucket = int(time.time() // window_seconds)
        return f"{settings.redis_key_prefix}:rate:{scope}:{identity}:{window_name}:{bucket}"

    async def check(self, scope: str, identity: str) -> RateLimitResult:
        minute_key = self._bucket_key(scope, identity, "minute", 60)
        minute_count = await self._backend.increment(minute_key, 60)
        if minute_count > settings.rate_limit_per_minute:
            return RateLimitResult(allowed=False, retry_after=60)

        hour_key = self._bucket_key(scope, identity, "hour", 3600)
        hour_count = await self._backend.increment(hour_key, 3600)
        if hour_count > settings.rate_limit_per_hour:
            return RateLimitResult(allowed=False, retry_after=3600)

        return RateLimitResult(allowed=True)

    def clear(self) -> None:
        clear = getattr(self._backend, "clear", None)
        if clear:
            clear()

    async def health(self) -> dict:
        health = getattr(self._backend, "health", None)
        if not health:
            return {"backend": "unknown", "degraded": True}
        return await health()


def _get_client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def identity_from_auth(request: Request, auth_context: AuthContext | None) -> str:
    if auth_context and auth_context.user_id:
        return f"user:{auth_context.user_id}"
    return f"ip:{_get_client_ip(request)}"


def identity_from_feishu_event(request: Request, event_data: dict | None, action_data: dict | None = None) -> str:
    data = event_data or action_data or {}
    if data.get("open_id"):
        return f"feishu_user:{data['open_id']}"
    if data.get("chat_id"):
        return f"feishu_chat:{data['chat_id']}"
    return f"ip:{_get_client_ip(request)}"


async def enforce_rate_limit(scope: str, identity: str) -> None:
    result = await rate_limiter.check(scope, identity)
    if result.allowed:
        return
    raise HTTPException(
        status_code=429,
        detail="请求过于频繁，请稍后重试",
        headers={"Retry-After": str(result.retry_after)},
    )


def get_rate_limiter() -> RateLimiter:
    fallback = InMemoryRateLimitBackend()
    try:
        client = create_redis_client()
        return RateLimiter(FallbackRateLimitBackend(RedisRateLimitBackend(client), fallback))
    except Exception as e:
        logger.warning(
            "Redis rate limit backend cannot be initialized; using in-memory limiter",
            extra={
                "component": "rate_limiter",
                "storage_backend": "memory",
                "storage_degraded": True,
                "fallback_reason": "redis_init_failed",
                "error": str(e),
            },
        )
        return RateLimiter(fallback)


rate_limiter = get_rate_limiter()
