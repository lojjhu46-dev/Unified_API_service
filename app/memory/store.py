"""会话记忆存储"""

import json
import uuid
from typing import List
from app.config import settings
from app.observability.logging import get_logger
from app.redis_client import RedisFallbackState, create_redis_client, redis_key

logger = get_logger(__name__)


class MemoryStore:
    """会话记忆存储基类"""

    async def get_history(self, session_id: str, max_messages: int = 6) -> List[dict]:
        raise NotImplementedError

    async def append_turn(self, session_id: str, question: str, answer: str) -> None:
        raise NotImplementedError

    async def create_session(self) -> str:
        raise NotImplementedError

    async def clear_session(self, session_id: str) -> None:
        raise NotImplementedError

    async def session_exists(self, session_id: str) -> bool:
        raise NotImplementedError

    async def health(self) -> dict:
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """内存会话存储"""

    def __init__(self):
        self._sessions: dict[str, List[dict]] = {}

    async def get_history(self, session_id: str, max_messages: int = 6) -> List[dict]:
        messages = self._sessions.get(session_id, [])
        return messages[-max_messages:]

    async def append_turn(self, session_id: str, question: str, answer: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": "user", "content": question})
        self._sessions[session_id].append({"role": "assistant", "content": answer})

    async def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:12]
        self._sessions[session_id] = []
        return session_id

    async def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def health(self) -> dict:
        return {
            "backend": "memory",
            "fallback_active": False,
            "degraded": False,
            "session_count": len(self._sessions),
        }


class RedisStore(MemoryStore):
    """Redis会话存储"""

    def __init__(self, redis_client=None):
        self._redis = redis_client or create_redis_client()

    def _session_key(self, session_id: str) -> str:
        return redis_key("session", session_id)

    def _session_meta_key(self, session_id: str) -> str:
        return redis_key("session_meta", session_id)

    async def _refresh_ttl(self, session_id: str) -> None:
        ttl = max(int(settings.memory_session_ttl_seconds), 1)
        await self._redis.expire(self._session_key(session_id), ttl)
        await self._redis.expire(self._session_meta_key(session_id), ttl)

    async def get_history(self, session_id: str, max_messages: int = 6) -> List[dict]:
        key = self._session_key(session_id)
        messages = await self._redis.lrange(key, -max_messages, -1)
        return [json.loads(m) for m in messages]

    async def append_turn(self, session_id: str, question: str, answer: str) -> None:
        key = self._session_key(session_id)
        ttl = max(int(settings.memory_session_ttl_seconds), 1)
        await self._redis.set(self._session_meta_key(session_id), "1", ex=ttl)
        await self._redis.rpush(
            key,
            json.dumps({"role": "user", "content": question}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False),
        )
        await self._redis.ltrim(key, -settings.memory_max_messages, -1)
        await self._refresh_ttl(session_id)

    async def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:12]
        await self._redis.set(
            self._session_meta_key(session_id),
            "1",
            ex=max(int(settings.memory_session_ttl_seconds), 1),
        )
        return session_id

    async def clear_session(self, session_id: str) -> None:
        await self._redis.delete(self._session_key(session_id), self._session_meta_key(session_id))

    async def session_exists(self, session_id: str) -> bool:
        return bool(
            await self._redis.exists(self._session_key(session_id))
            or await self._redis.exists(self._session_meta_key(session_id))
        )

    async def health(self) -> dict:
        ping = getattr(self._redis, "ping", None)
        if ping:
            await ping()
        return {
            "backend": "redis",
            "fallback_active": False,
            "degraded": False,
            "ttl_seconds": settings.memory_session_ttl_seconds,
        }


class FallbackMemoryStore(MemoryStore):
    """Redis优先、内存兜底的会话存储。"""

    def __init__(self, primary: MemoryStore, fallback: InMemoryStore):
        self._primary = primary
        self._fallback = fallback
        self._state = RedisFallbackState(
            "memory_store",
            "Redis memory store unavailable; falling back to in-memory store",
            "Redis memory store recovered; using Redis backend",
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

    async def get_history(self, session_id: str, max_messages: int = 6) -> List[dict]:
        if self._state.should_try_primary():
            try:
                history = await self._primary.get_history(session_id, max_messages)
                self._state.record_success()
                fallback_history = await self._fallback.get_history(session_id, max_messages)
                return history or fallback_history
            except Exception as e:
                self._record_primary_failure(e)
        return await self._fallback.get_history(session_id, max_messages)

    async def append_turn(self, session_id: str, question: str, answer: str) -> None:
        await self._try_primary("append_turn", session_id, question, answer)
        await self._fallback.append_turn(session_id, question, answer)

    async def create_session(self) -> str:
        if self._state.should_try_primary():
            try:
                session_id = await self._primary.create_session()
                self._state.record_success()
                if session_id not in self._fallback._sessions:
                    self._fallback._sessions[session_id] = []
                return session_id
            except Exception as e:
                self._record_primary_failure(e)
        return await self._fallback.create_session()

    async def clear_session(self, session_id: str) -> None:
        await self._try_primary("clear_session", session_id)
        await self._fallback.clear_session(session_id)

    async def session_exists(self, session_id: str) -> bool:
        if self._state.should_try_primary():
            try:
                exists = await self._primary.session_exists(session_id)
                self._state.record_success()
                return exists or await self._fallback.session_exists(session_id)
            except Exception as e:
                self._record_primary_failure(e)
        return await self._fallback.session_exists(session_id)

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


def get_memory_store() -> MemoryStore:
    """获取会话存储实例"""
    fallback = InMemoryStore()
    try:
        return FallbackMemoryStore(RedisStore(), fallback)
    except Exception as e:
        logger.warning(
            "Redis memory store cannot be initialized; using in-memory store",
            extra={
                "component": "memory_store",
                "storage_backend": "memory",
                "storage_degraded": True,
                "fallback_reason": "redis_init_failed",
                "error": str(e),
            },
        )
        return fallback


memory_store = get_memory_store()
