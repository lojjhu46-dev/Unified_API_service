"""Rate limiter Redis fallback tests."""

import pytest

from app.security.rate_limit import FallbackRateLimitBackend, InMemoryRateLimitBackend, RedisRateLimitBackend


class RecoveringRedis:
    def __init__(self):
        self.fail = True
        self.values = {}

    def _maybe_fail(self):
        if self.fail:
            raise RuntimeError("redis down")

    async def incr(self, key):
        self._maybe_fail()
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key, window_seconds):
        self._maybe_fail()

    async def ping(self):
        self._maybe_fail()
        return True


@pytest.mark.asyncio
async def test_fallback_rate_limiter_retries_and_recovers_after_cooldown():
    redis = RecoveringRedis()
    backend = FallbackRateLimitBackend(RedisRateLimitBackend(redis), InMemoryRateLimitBackend())

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("app.redis_client.settings.redis_retry_cooldown_seconds", 0)
        assert await backend.increment("key", 60) == 1
        failed_health = await backend.health()
        assert failed_health["backend"] == "memory"
        assert failed_health["next_retry_at"] is not None

        redis.fail = False
        assert await backend.increment("key", 60) == 1
        recovered_health = await backend.health()

    assert recovered_health["backend"] == "redis"
    assert recovered_health["degraded"] is False
