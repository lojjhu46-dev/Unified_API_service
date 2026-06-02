"""Shared Redis client helpers."""

import time
from urllib.parse import urlsplit, urlunsplit

from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

_shared_redis_client = None


def _redact_redis_url(url: str) -> str:
    parsed = urlsplit(url)
    if "@" not in parsed.netloc:
        return url
    host = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, f"***@{host}", parsed.path, parsed.query, parsed.fragment))


def _build_redis_client():
    import redis.asyncio as redis

    socket_timeout = max(float(settings.redis_socket_timeout), 0.1)
    connect_timeout = max(float(settings.redis_connect_timeout), 0.1)
    health_check_interval = max(int(settings.redis_health_check_interval), 0)
    return redis.from_url(
        settings.redis_url,
        socket_timeout=socket_timeout,
        socket_connect_timeout=connect_timeout,
        retry_on_timeout=True,
        health_check_interval=health_check_interval,
        socket_keepalive=True,
        decode_responses=True,
    )


def create_redis_client(shared: bool = True):
    global _shared_redis_client
    if not shared:
        return _build_redis_client()
    if _shared_redis_client is None:
        _shared_redis_client = _build_redis_client()
    return _shared_redis_client


def reset_redis_client() -> None:
    global _shared_redis_client
    _shared_redis_client = None


async def log_redis_startup_health() -> None:
    client = create_redis_client()
    start = time.perf_counter()
    try:
        await client.ping()
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Redis startup ping succeeded",
            extra={
                "component": "redis",
                "redis_url": _redact_redis_url(settings.redis_url),
                "latency_ms": latency_ms,
                "socket_timeout": settings.redis_socket_timeout,
                "connect_timeout": settings.redis_connect_timeout,
            },
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "Redis startup ping failed; Redis-backed components may use memory fallback",
            extra={
                "component": "redis",
                "redis_url": _redact_redis_url(settings.redis_url),
                "latency_ms": latency_ms,
                "socket_timeout": settings.redis_socket_timeout,
                "connect_timeout": settings.redis_connect_timeout,
                "fallback_reason": "redis_startup_ping_failed",
                "error": str(e),
            },
        )


class RedisFallbackState:
    def __init__(self, component: str, warning_message: str, recovery_message: str):
        self.component = component
        self.warning_message = warning_message
        self.recovery_message = recovery_message
        self.primary_available = True
        self.last_error: str | None = None
        self.last_fallback_at: float | None = None
        self.next_retry_at: float | None = None

    def should_try_primary(self) -> bool:
        if self.primary_available:
            return True
        if self.next_retry_at is None:
            return True
        return time.time() >= self.next_retry_at

    def record_failure(self, error: Exception, **extra) -> None:
        now = time.time()
        self.primary_available = False
        self.last_error = str(error)
        self.last_fallback_at = now
        self.next_retry_at = now + max(float(settings.redis_retry_cooldown_seconds), 0)
        logger.warning(
            self.warning_message,
            extra={
                "component": self.component,
                "storage_backend": "memory",
                "storage_degraded": True,
                "fallback_reason": "redis_error",
                "error": str(error),
                "next_retry_at": self.next_retry_at,
                **extra,
            },
        )

    def record_success(self) -> None:
        if not self.primary_available:
            logger.info(
                self.recovery_message,
                extra={
                    "component": self.component,
                    "storage_backend": "redis",
                    "storage_degraded": False,
                    "last_fallback_at": self.last_fallback_at,
                },
            )
        self.primary_available = True
        self.last_error = None
        self.next_retry_at = None

    def redis_health(self, status: dict) -> dict:
        return {
            **status,
            "primary_backend": "redis",
            "fallback_backend": "memory",
            "fallback_active": False,
            "degraded": False,
            "last_error": None,
            "last_fallback_at": self.last_fallback_at,
            "next_retry_at": None,
        }

    def fallback_health(self, fallback_status: dict) -> dict:
        return {
            **fallback_status,
            "backend": "memory",
            "primary_backend": "redis",
            "fallback_backend": "memory",
            "fallback_active": True,
            "degraded": True,
            "fallback_reason": "redis_unavailable",
            "last_error": self.last_error,
            "last_fallback_at": self.last_fallback_at,
            "next_retry_at": self.next_retry_at,
        }


def redis_key(domain: str, *parts: object) -> str:
    suffix = ":".join(str(part) for part in parts if part is not None)
    if suffix:
        return f"{settings.redis_key_prefix}:{domain}:{suffix}"
    return f"{settings.redis_key_prefix}:{domain}"
