from __future__ import annotations

from hashlib import sha256
import re
from typing import Protocol


class RedisLike(Protocol):
    def scan_iter(self, *, match: str, count: int): ...
    def delete(self, *keys: str) -> int: ...


def _valid_segment(value: str, *, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError(f"invalid {name}")
    return value


def cache_key_for_retrieval(query: str, index_version: str) -> str:
    version = _valid_segment(index_version, name="index version")
    if not query or query != query.strip():
        raise ValueError("invalid query")
    digest = sha256(query.encode("utf-8")).hexdigest()[:32]
    return f"retrieval:{version}:{digest}"


def invalidate_index_cache(redis: RedisLike, index_version: str) -> int:
    version = _valid_segment(index_version, name="index version")
    deleted = 0
    batch: list[str] = []
    for key in redis.scan_iter(match=f"retrieval:{version}:*", count=100):
        batch.append(key)
        if len(batch) == 100:
            deleted += redis.delete(*batch)
            batch.clear()
    if batch:
        deleted += redis.delete(*batch)
    return deleted
