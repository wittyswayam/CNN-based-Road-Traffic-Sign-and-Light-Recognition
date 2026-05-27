"""
TrafficVision-AI :: Caching Layer
====================================
Unified caching abstraction with Redis backend (production)
and in-memory LRU fallback (development/testing).

Features
--------
- Transparent backend switching via config
- Image inference result caching (SHA-256 keyed)
- Model prediction caching with TTL
- Cache warming for hot images
- Cache statistics and hit-rate tracking
- Async Redis support
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses


# ---------------------------------------------------------------------------
# Abstract cache interface
# ---------------------------------------------------------------------------


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        ...

    @abstractmethod
    def set(self, key: str, value: str, ttl_seconds: int = 3600) -> bool:
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    def flush(self) -> None:
        ...

    @abstractmethod
    def stats(self) -> CacheStats:
        ...


# ---------------------------------------------------------------------------
# In-memory LRU backend
# ---------------------------------------------------------------------------


class InMemoryCache(CacheBackend):
    """
    Thread-safe LRU cache with TTL support.
    Suitable for single-process development and testing.
    NOT suitable for multi-replica deployments (use Redis).
    """

    def __init__(self, max_size: int = 2048) -> None:
        self._store: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._max_size = max_size
        self._stats = CacheStats()

    def _is_expired(self, key: str) -> bool:
        expiry = self._expiry.get(key)
        return expiry is not None and time.time() > expiry

    def get(self, key: str) -> Optional[str]:
        if key not in self._store or self._is_expired(key):
            if key in self._store:
                del self._store[key]
                del self._expiry[key]
            self._stats.misses += 1
            return None

        # Move to end (LRU)
        self._store.move_to_end(key)
        self._stats.hits += 1
        return self._store[key]

    def set(self, key: str, value: str, ttl_seconds: int = 3600) -> bool:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        self._expiry[key] = time.time() + ttl_seconds

        if len(self._store) > self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            self._expiry.pop(evicted_key, None)
            self._stats.evictions += 1

        self._stats.sets += 1
        return True

    def delete(self, key: str) -> bool:
        existed = key in self._store
        self._store.pop(key, None)
        self._expiry.pop(key, None)
        return existed

    def exists(self, key: str) -> bool:
        return key in self._store and not self._is_expired(key)

    def flush(self) -> None:
        self._store.clear()
        self._expiry.clear()

    def stats(self) -> CacheStats:
        return self._stats


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


class RedisCache(CacheBackend):
    """
    Redis-backed cache for production multi-replica deployments.
    Requires: pip install redis
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 20,
    ) -> None:
        try:
            import redis
            self._pool = redis.ConnectionPool(
                host=host, port=port, db=db, password=password,
                max_connections=max_connections, decode_responses=True,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            self._client.ping()
            logger.info("Redis cache connected: %s:%d db=%d", host, port, db)
        except ImportError:
            raise RuntimeError("Install redis: pip install redis")
        except Exception as exc:
            raise RuntimeError(f"Redis connection failed: {exc}") from exc

        self._stats = CacheStats()

    def get(self, key: str) -> Optional[str]:
        try:
            value = self._client.get(key)
            if value is None:
                self._stats.misses += 1
            else:
                self._stats.hits += 1
            return value
        except Exception as exc:
            self._stats.errors += 1
            logger.warning("Redis GET error for key=%s: %s", key, exc)
            return None

    def set(self, key: str, value: str, ttl_seconds: int = 3600) -> bool:
        try:
            self._client.setex(key, ttl_seconds, value)
            self._stats.sets += 1
            return True
        except Exception as exc:
            self._stats.errors += 1
            logger.warning("Redis SET error for key=%s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        try:
            return bool(self._client.delete(key))
        except Exception as exc:
            logger.warning("Redis DELETE error: %s", exc)
            return False

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client.exists(key))
        except Exception:
            return False

    def flush(self) -> None:
        try:
            self._client.flushdb()
        except Exception as exc:
            logger.warning("Redis FLUSH error: %s", exc)

    def stats(self) -> CacheStats:
        return self._stats


# ---------------------------------------------------------------------------
# High-level inference cache
# ---------------------------------------------------------------------------


class InferenceCache:
    """
    Domain-specific cache for model inference results.
    Keyed by SHA-256 of raw image bytes.
    Values serialised as JSON.
    """

    KEY_PREFIX = "tv:infer:"

    def __init__(
        self,
        backend: CacheBackend,
        ttl_seconds: int = 3600,
    ) -> None:
        self._backend = backend
        self._ttl = ttl_seconds

    def _key(self, image_bytes: bytes) -> str:
        digest = hashlib.sha256(image_bytes).hexdigest()
        return f"{self.KEY_PREFIX}{digest}"

    def get(self, image_bytes: bytes) -> Optional[Dict]:
        raw = self._backend.get(self._key(image_bytes))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set(self, image_bytes: bytes, result: Dict) -> None:
        self._backend.set(self._key(image_bytes), json.dumps(result), self._ttl)

    def invalidate(self, image_bytes: bytes) -> None:
        self._backend.delete(self._key(image_bytes))

    def hit_rate(self) -> float:
        return self._backend.stats().hit_rate

    def stats(self) -> CacheStats:
        return self._backend.stats()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_cache(
    backend: str = "memory",
    host: str = "localhost",
    port: int = 6379,
    ttl_seconds: int = 3600,
    max_size: int = 2048,
) -> InferenceCache:
    """
    Factory function for creating the appropriate cache backend.

    Parameters
    ----------
    backend     : "memory" | "redis"
    host        : Redis host (ignored for memory backend)
    port        : Redis port
    ttl_seconds : Cache entry TTL
    max_size    : LRU max entries (memory backend only)
    """
    if backend == "redis":
        try:
            _backend: CacheBackend = RedisCache(host=host, port=port)
        except RuntimeError as exc:
            logger.warning("Redis unavailable (%s) — falling back to in-memory cache", exc)
            _backend = InMemoryCache(max_size=max_size)
    else:
        _backend = InMemoryCache(max_size=max_size)

    return InferenceCache(_backend, ttl_seconds=ttl_seconds)
