"""Pending Feishu file confirmation storage."""

import json
import time
from typing import Protocol

from app.observability.logging import get_logger
from app.redis_client import RedisFallbackState, create_redis_client, redis_key

logger = get_logger(__name__)

pending_feishu_files: dict[str, dict] = {}


class PendingFileStore(Protocol):
    async def create(self, pending_id: str, pending: dict, ttl_seconds: int) -> None:
        ...

    async def get(self, pending_id: str) -> dict | None:
        ...

    async def delete(self, pending_id: str) -> None:
        ...

    async def consume(self, pending_id: str) -> dict | None:
        ...

    async def health(self) -> dict:
        ...


class InMemoryPendingFileStore:
    def __init__(self, storage: dict[str, dict] | None = None):
        self._storage = storage if storage is not None else {}

    async def create(self, pending_id: str, pending: dict, ttl_seconds: int) -> None:
        self._storage[pending_id] = dict(pending)

    async def get(self, pending_id: str) -> dict | None:
        pending = self._storage.get(pending_id)
        if not pending:
            return None
        if pending.get("expires_at", 0) < time.time():
            self._storage.pop(pending_id, None)
            return None
        return dict(pending)

    async def delete(self, pending_id: str) -> None:
        self._storage.pop(pending_id, None)

    async def consume(self, pending_id: str) -> dict | None:
        pending = await self.get(pending_id)
        if not pending:
            return None
        self._storage.pop(pending_id, None)
        return pending

    async def health(self) -> dict:
        return {
            "backend": "memory",
            "fallback_active": False,
            "degraded": False,
            "local_key_count": len(self._storage),
        }


class RedisPendingFileStore:
    def __init__(self, redis_client):
        self._redis = redis_client

    def _key(self, pending_id: str) -> str:
        return redis_key("feishu_pending", pending_id)

    async def create(self, pending_id: str, pending: dict, ttl_seconds: int) -> None:
        payload = json.dumps(pending, ensure_ascii=False)
        await self._redis.set(self._key(pending_id), payload, ex=max(int(ttl_seconds), 1))

    async def get(self, pending_id: str) -> dict | None:
        payload = await self._redis.get(self._key(pending_id))
        if not payload:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        pending = json.loads(payload)
        if pending.get("expires_at", 0) < time.time():
            await self.delete(pending_id)
            return None
        return pending

    async def delete(self, pending_id: str) -> None:
        await self._redis.delete(self._key(pending_id))

    async def consume(self, pending_id: str) -> dict | None:
        script = """
        local value = redis.call('GET', KEYS[1])
        if not value then
            return nil
        end
        redis.call('DEL', KEYS[1])
        return value
        """
        payload = await self._redis.eval(script, 1, self._key(pending_id))
        if not payload:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        pending = json.loads(payload)
        if pending.get("expires_at", 0) < time.time():
            return None
        return pending

    async def health(self) -> dict:
        ping = getattr(self._redis, "ping", None)
        if ping:
            await ping()
        return {
            "backend": "redis",
            "fallback_active": False,
            "degraded": False,
        }


class FallbackPendingFileStore:
    def __init__(self, primary: PendingFileStore, fallback: InMemoryPendingFileStore):
        self._primary = primary
        self._fallback = fallback
        self._state = RedisFallbackState(
            "feishu_pending_store",
            "Redis pending file store unavailable; falling back to in-memory store",
            "Redis pending file store recovered; using Redis backend",
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

    async def create(self, pending_id: str, pending: dict, ttl_seconds: int) -> None:
        await self._try_primary("create", pending_id, pending, ttl_seconds)
        await self._fallback.create(pending_id, pending, ttl_seconds)

    async def get(self, pending_id: str) -> dict | None:
        pending = await self._try_primary("get", pending_id)
        return pending or await self._fallback.get(pending_id)

    async def delete(self, pending_id: str) -> None:
        await self._try_primary("delete", pending_id)
        await self._fallback.delete(pending_id)

    async def consume(self, pending_id: str) -> dict | None:
        pending = await self._try_primary("consume", pending_id)
        fallback_pending = await self._fallback.consume(pending_id)
        return pending or fallback_pending

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


def get_pending_file_store() -> PendingFileStore:
    fallback = InMemoryPendingFileStore(pending_feishu_files)
    try:
        client = create_redis_client()
        return FallbackPendingFileStore(RedisPendingFileStore(client), fallback)
    except Exception as e:
        logger.warning(
            "Redis pending file store cannot be initialized; using in-memory store",
            extra={
                "component": "feishu_pending_store",
                "storage_backend": "memory",
                "storage_degraded": True,
                "fallback_reason": "redis_init_failed",
                "error": str(e),
            },
        )
        return fallback
