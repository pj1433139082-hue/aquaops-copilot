from __future__ import annotations

import pytest

from aquaops.cache.service import cache_key_for_retrieval, invalidate_index_cache


class _FakeRedis:
    def __init__(self, keys: list[str]) -> None:
        self.keys = set(keys)
        self.deleted: list[str] = []
        self.delete_batches: list[int] = []

    def scan_iter(self, *, match: str, count: int):
        assert count == 100
        prefix = match.removesuffix("*")
        return iter(sorted(key for key in self.keys if key.startswith(prefix)))

    def delete(self, *keys: str) -> int:
        self.delete_batches.append(len(keys))
        self.deleted.extend(keys)
        self.keys.difference_update(keys)
        return len(keys)


def test_retrieval_cache_key_changes_when_index_version_changes() -> None:
    assert cache_key_for_retrieval("nh3-n", "index-v1") != cache_key_for_retrieval(
        "nh3-n", "index-v2"
    )


def test_retrieval_cache_key_does_not_contain_raw_query() -> None:
    key = cache_key_for_retrieval("sensitive synthetic query", "index-v1")
    assert "sensitive" not in key
    assert key.startswith("retrieval:index-v1:")


def test_invalidation_deletes_only_the_requested_index_version() -> None:
    target = cache_key_for_retrieval("a", "index-v1")
    retained = cache_key_for_retrieval("a", "index-v2")
    redis = _FakeRedis([target, retained, "unrelated:key"])

    deleted = invalidate_index_cache(redis, "index-v1")

    assert deleted == 1
    assert redis.deleted == [target]
    assert retained in redis.keys
    assert "unrelated:key" in redis.keys


def test_invalidation_deletes_in_bounded_batches() -> None:
    keys = [
        cache_key_for_retrieval(f"query-{index}", "index-v1") for index in range(250)
    ]
    redis = _FakeRedis(keys)

    assert invalidate_index_cache(redis, "index-v1") == 250
    assert len(redis.deleted) == 250
    assert redis.delete_batches == [100, 100, 50]


@pytest.mark.parametrize("version", ["bad\\version", "bad*version", "版本一", "a" * 81])
def test_cache_version_rejects_non_ascii_or_glob_input(version: str) -> None:
    with pytest.raises(ValueError, match="invalid index version"):
        cache_key_for_retrieval("query", version)
