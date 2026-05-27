"""会话记忆存储"""

import uuid
from typing import List
from app.observability.logging import get_logger

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


class RedisStore(MemoryStore):
    """Redis会话存储"""

    def __init__(self):
        raise RuntimeError("RedisStore 需要异步 ping 验证，当前阶段默认使用 InMemoryStore")

    def _session_key(self, session_id: str) -> str:
        return f"{settings.redis_key_prefix}:session:{session_id}"

    async def get_history(self, session_id: str, max_messages: int = 6) -> List[dict]:
        if not self._redis:
            return []
        try:
            key = self._session_key(session_id)
            messages = await self._redis.lrange(key, -max_messages, -1)
            return [json.loads(m) for m in messages]
        except Exception as e:
            logger.error(f"获取历史失败: {e}")
            return []

    async def append_turn(self, session_id: str, question: str, answer: str) -> None:
        if not self._redis:
            return
        try:
            key = self._session_key(session_id)
            await self._redis.rpush(
                key,
                json.dumps({"role": "user", "content": question}, ensure_ascii=False),
                json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False),
            )
            await self._redis.ltrim(key, -settings.memory_max_messages, -1)
        except Exception as e:
            logger.error(f"写入历史失败: {e}")

    async def create_session(self) -> str:
        return str(uuid.uuid4())[:12]

    async def clear_session(self, session_id: str) -> None:
        if not self._redis:
            return
        try:
            await self._redis.delete(self._session_key(session_id))
        except Exception as e:
            logger.error(f"清除会话失败: {e}")

    async def session_exists(self, session_id: str) -> bool:
        if not self._redis:
            return False
        try:
            return await self._redis.exists(self._session_key(session_id)) > 0
        except Exception:
            return False


def get_memory_store() -> MemoryStore:
    """获取会话存储实例"""
    # 阶段四优先保证默认开发/测试环境的会话记忆可靠可用。
    # Redis 生产化启用需要异步 ping 验证，避免假连接导致历史静默丢失。
    return InMemoryStore()


memory_store = get_memory_store()
