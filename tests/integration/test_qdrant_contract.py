"""Opt-in, local-only Qdrant contract coverage; this test never starts Docker."""

import os
from collections.abc import Mapping
from types import SimpleNamespace
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from aquaops.config import Settings
from aquaops.rag.public_corpus import PublicKnowledgeDocument, chunk_public_document
from aquaops.rag.store import (
    PUBLIC_COLLECTION,
    PublicStoreUnavailableError,
    QdrantPublicKnowledgeStore,
)


def _contract_writes_enabled(env: Mapping[str, str]) -> bool:
    return (
        env.get("AQUAOPS_RUN_QDRANT_CONTRACT") == "1"
        and env.get("AQUAOPS_QDRANT_CONTRACT_ALLOW_WRITE") == "1"
    )


def _is_local_qdrant_url(url: str) -> bool:
    return urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"}


def _contract_collection_name(run_uuid: UUID) -> str:
    return f"aquaops_contract_{run_uuid.hex}"


def _run_contract(client, collection_name: str, topic: str, chunk) -> None:
    store = None
    primary_error: BaseException | None = None
    cleanup_error: Exception | None = None
    try:
        store = QdrantPublicKnowledgeStore._for_contract_testing(
            client, collection_name
        )
        store.initialize()
        store.upsert([chunk], [[0.25] * 1024], topic=topic, effective_date="2026-07-19")
        results = store.search_dense([0.25] * 1024, limit=1, topic=topic)

        assert results
        assert results[0].data_class == "public"
        assert results[0].access_policy == "public_read"
        assert results[0].chunk_id == chunk.chunk_id
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            if store is not None and store._collection_created:
                try:
                    client.delete_collection(collection_name=collection_name)
                except Exception as error:
                    if primary_error is None:
                        cleanup_error = error
        finally:
            try:
                client.close()
            except Exception as error:
                if primary_error is None and cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error


@pytest.mark.parametrize(
    ("env", "enabled"),
    [
        ({}, False),
        ({"AQUAOPS_RUN_QDRANT_CONTRACT": "1"}, False),
        ({"AQUAOPS_QDRANT_CONTRACT_ALLOW_WRITE": "1"}, False),
        (
            {
                "AQUAOPS_RUN_QDRANT_CONTRACT": "1",
                "AQUAOPS_QDRANT_CONTRACT_ALLOW_WRITE": "1",
            },
            True,
        ),
    ],
)
def test_contract_writes_enabled_requires_both_flags(env, enabled) -> None:
    assert _contract_writes_enabled(env) is enabled


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("http://localhost:6333", True),
        ("http://127.0.0.1:6333", True),
        ("http://qdrant:6333", False),
        ("https://example.test", False),
        ("not-a-url", False),
    ],
)
def test_local_qdrant_url_guard(url, allowed) -> None:
    assert _is_local_qdrant_url(url) is allowed


def test_contract_collection_builder_and_store_factory_share_the_exact_contract() -> (
    None
):
    collection_name = _contract_collection_name(
        UUID("12345678-1234-5678-9abc-def012345678")
    )

    store = QdrantPublicKnowledgeStore._for_contract_testing(
        SimpleNamespace(), collection_name
    )

    assert collection_name == "aquaops_contract_12345678123456789abcdef012345678"
    assert store.collection_name == collection_name


def test_contract_create_failure_closes_without_deleting_a_collection() -> None:
    class Client:
        def __init__(self):
            self.deleted = []
            self.closed = False

        def collection_exists(self, name):
            return False

        def create_collection(self, **kwargs):
            raise RuntimeError("create failed")

        def close(self):
            self.closed = True

        def delete_collection(self, collection_name):
            self.deleted.append(collection_name)

    client = Client()

    with pytest.raises(PublicStoreUnavailableError):
        _run_contract(client, f"aquaops_contract_{'d' * 32}", "topic", public_chunk())

    assert client.deleted == []
    assert client.closed is True


def test_contract_assertion_failure_deletes_temporary_collection_and_closes() -> None:
    class Client:
        def __init__(self):
            self.calls = []
            self.closed = False

        def collection_exists(self, name):
            self.calls.append(("exists", name))
            return False

        def create_collection(self, **kwargs):
            self.calls.append(("create", kwargs["collection_name"]))
            return True

        def create_payload_index(self, **kwargs):
            self.calls.append(("index", kwargs["collection_name"]))

        def upsert(self, **kwargs):
            self.calls.append(("upsert", kwargs["collection_name"]))

        def query_points(self, **kwargs):
            self.calls.append(("query", kwargs["collection_name"]))
            return SimpleNamespace(points=[])

        def delete_collection(self, collection_name):
            self.calls.append(("delete", collection_name))

        def close(self):
            self.closed = True

    client = Client()
    collection_name = f"aquaops_contract_{'a' * 32}"

    with pytest.raises(AssertionError):
        _run_contract(client, collection_name, "topic", public_chunk())

    assert ("delete", collection_name) in client.calls
    assert client.closed is True
    assert all(name != PUBLIC_COLLECTION for _, name in client.calls)


def test_contract_index_failure_after_collection_creation_deletes_and_closes() -> None:
    class Client:
        def __init__(self):
            self.created = False
            self.deleted = []
            self.closed = False
            self.indexes = 0

        def collection_exists(self, name):
            return self.created

        def create_collection(self, **kwargs):
            self.created = True
            return True

        def create_payload_index(self, **kwargs):
            self.indexes += 1
            if self.indexes == 3:
                raise RuntimeError("index failed")

        def delete_collection(self, collection_name):
            self.deleted.append(collection_name)

        def close(self):
            self.closed = True

    client = Client()
    collection_name = f"aquaops_contract_{'b' * 32}"

    with pytest.raises(PublicStoreUnavailableError):
        _run_contract(client, collection_name, "topic", public_chunk())

    assert client.deleted == [collection_name]
    assert client.closed is True


def test_contract_cleanup_failure_does_not_override_the_primary_assertion() -> None:
    class Client:
        def collection_exists(self, name):
            return False

        def create_collection(self, **kwargs):
            return True

        def create_payload_index(self, **kwargs):
            return None

        def upsert(self, **kwargs):
            return None

        def query_points(self, **kwargs):
            return SimpleNamespace(points=[])

        def delete_collection(self, collection_name):
            raise RuntimeError("cleanup failed")

        def close(self):
            return None

    with pytest.raises(AssertionError):
        _run_contract(Client(), f"aquaops_contract_{'e' * 32}", "topic", public_chunk())


def test_contract_successful_run_still_closes_when_delete_fails() -> None:
    class Client:
        def __init__(self):
            self.point = None
            self.closed = False

        def collection_exists(self, name):
            return False

        def create_collection(self, **kwargs):
            return True

        def create_payload_index(self, **kwargs):
            return None

        def upsert(self, **kwargs):
            self.point = kwargs["points"][0]

        def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    models.ScoredPoint(
                        id=self.point.id,
                        version=1,
                        score=0.9,
                        payload=self.point.payload,
                    )
                ]
            )

        def delete_collection(self, collection_name):
            raise RuntimeError("delete failed")

        def close(self):
            self.closed = True

    client = Client()

    with pytest.raises(RuntimeError, match="delete failed"):
        _run_contract(client, f"aquaops_contract_{'c' * 32}", "topic", public_chunk())

    assert client.closed is True


@pytest.mark.skipif(
    not _contract_writes_enabled(os.environ),
    reason=(
        "set AQUAOPS_RUN_QDRANT_CONTRACT=1 and "
        "AQUAOPS_QDRANT_CONTRACT_ALLOW_WRITE=1 for an already-running local Qdrant service"
    ),
)
def test_public_qdrant_contract_uses_an_isolated_temporary_collection() -> None:
    settings = Settings()
    if not _is_local_qdrant_url(settings.qdrant_url):
        pytest.skip(
            "the opt-in Qdrant contract test only permits localhost or 127.0.0.1"
        )
    run_uuid = uuid4()
    collection_name = _contract_collection_name(run_uuid)
    topic = f"contract-{run_uuid.hex}"
    chunk = chunk_public_document(
        PublicKnowledgeDocument(
            source_id=f"contract-synthetic-{run_uuid}",
            title="Contract synthetic guidance",
            source_url="https://example.test/contract-public-guidance",
            source_version="2026-07",
            license_name="CC BY 4.0",
            text="# Sampling\n\nUse synthetic public sample guidance.",
        )
    )[0]

    _run_contract(QdrantClient(settings.qdrant_url), collection_name, topic, chunk)


def public_chunk():
    return chunk_public_document(
        PublicKnowledgeDocument(
            source_id="contract-test-synthetic",
            title="Contract test guidance",
            source_url="https://example.test/contract-test",
            source_version="2026-07",
            license_name="CC BY 4.0",
            text="# Sampling\n\nUse synthetic public sample guidance.",
        )
    )[0]
