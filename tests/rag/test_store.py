from dataclasses import replace
from math import inf, nan
from types import SimpleNamespace
from uuid import UUID, uuid4, uuid5

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

import aquaops.rag.store as store_module
from aquaops.config import Settings
from aquaops.rag.public_corpus import (
    PublicKnowledgeChunk,
    PublicKnowledgeDocument,
    chunk_public_document,
)
from aquaops.rag.store import (
    PUBLIC_COLLECTION,
    PublicStoreError,
    QdrantPublicKnowledgeStore,
)


def public_chunk() -> PublicKnowledgeChunk:
    return PublicKnowledgeChunk(
        chunk_id="a" * 64,
        qdrant_point_id=str(
            uuid5(UUID("ce3be424-4f91-55a3-8ed0-dd56dc95c39a"), "a" * 64)
        ),
        source_id="synthetic-guidance",
        source_url="https://example.test/public-guidance",
        source_version="2026-07",
        parent_title="Synthetic public guidance",
        section_path=("Water quality", "Sampling"),
        text="Use clean synthetic sample containers.",
    )


def vector() -> list[float]:
    return [0.25] * 1024


def test_only_the_fixed_public_collection_is_allowed() -> None:
    store = QdrantPublicKnowledgeStore(QdrantClient(":memory:"))

    assert store.collection_name == PUBLIC_COLLECTION
    with pytest.raises(ValueError):
        QdrantPublicKnowledgeStore(QdrantClient(":memory:"), collection_name="private")


def test_contract_factory_only_allows_the_strict_temporary_collection_prefix() -> None:
    client = QdrantClient(":memory:")
    collection_name = f"aquaops_contract_{'a' * 32}"

    contract_store = QdrantPublicKnowledgeStore._for_contract_testing(
        client, collection_name
    )

    assert contract_store.collection_name == collection_name
    for invalid_name in (
        "water_public_knowledge_v1",
        "aquaops_contract_abc123",
        f"aquaops_contract_{'A' * 32}",
    ):
        with pytest.raises(ValueError):
            QdrantPublicKnowledgeStore._for_contract_testing(client, invalid_name)
    with pytest.raises(TypeError):
        QdrantPublicKnowledgeStore(
            client, collection_name=collection_name, _allow_contract_collection=True
        )


def test_normal_constructor_initializes_all_store_state() -> None:
    client = QdrantClient(":memory:")

    store = QdrantPublicKnowledgeStore(client)

    assert store._client is client
    assert store.collection_name == PUBLIC_COLLECTION
    assert store._initialized is False
    assert store._collection_created is False


def test_contract_factory_uses_the_normal_constructor_and_external_tokens_cannot_bypass_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_name = f"aquaops_contract_{'b' * 32}"
    client = QdrantClient(":memory:")
    original_init = QdrantPublicKnowledgeStore.__init__
    constructor_calls = []

    def tracking_init(self, *args, **kwargs):
        constructor_calls.append((args, kwargs))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(QdrantPublicKnowledgeStore, "__init__", tracking_init)

    store = QdrantPublicKnowledgeStore._for_contract_testing(client, collection_name)

    assert store.collection_name == collection_name
    assert len(constructor_calls) == 1
    for external_token in (object(), True):
        with pytest.raises(ValueError):
            QdrantPublicKnowledgeStore(
                client,
                collection_name=collection_name,
                _contract_sentinel=external_token,
            )


def test_settings_rejects_environment_collection_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AQUAOPS_PUBLIC_QDRANT_COLLECTION", "private")

    with pytest.raises(ValueError):
        Settings()


def test_store_exposes_distinct_validation_and_unavailability_errors() -> None:
    assert issubclass(store_module.PublicStoreValidationError, PublicStoreError)
    assert issubclass(store_module.PublicStoreUnavailableError, PublicStoreError)
    assert not issubclass(store_module.PublicStoreUnavailableError, ValueError)


def test_initialize_only_ensures_a_collection_once_per_store_instance() -> None:
    class Client:
        def __init__(self):
            self.creates = 0
            self.indexes = []

        def collection_exists(self, name):
            return False

        def create_collection(self, **kwargs):
            self.creates += 1

        def create_payload_index(self, **kwargs):
            self.indexes.append(kwargs["field_name"])

    client = Client()
    store = QdrantPublicKnowledgeStore(client)

    store.initialize()
    store.initialize()

    assert client.creates == 1
    assert client.indexes == [
        "data_class",
        "access_policy",
        "source_id",
        "source_version",
        "topic",
        "effective_date",
    ]


def test_initialize_recovers_from_collection_creation_race_by_validating_existing_schema() -> (
    None
):
    class ConflictError(Exception):
        status_code = 409

    class RacingClient:
        def __init__(self):
            self.gets = 0
            self.indexes = []

        def collection_exists(self, name):
            return False

        def create_collection(self, **kwargs):
            raise ConflictError("opaque conflict")

        def get_collection(self, name):
            self.gets += 1
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(
                            size=1024, distance=models.Distance.COSINE
                        )
                    )
                ),
                payload_schema={},
            )

        def create_payload_index(self, **kwargs):
            self.indexes.append(kwargs["field_name"])

    client = RacingClient()

    QdrantPublicKnowledgeStore(client).initialize()

    assert client.gets == 1
    assert len(client.indexes) == 6


def test_initialize_rejects_non_conflict_collection_creation_errors_as_unavailable() -> (
    None
):
    class ServerError(Exception):
        status_code = 500

    class BrokenClient:
        def collection_exists(self, name):
            return False

        def create_collection(self, **kwargs):
            raise ServerError("not a conflict")

    with pytest.raises(store_module.PublicStoreUnavailableError):
        QdrantPublicKnowledgeStore(BrokenClient()).initialize()


def test_existing_collection_configuration_mismatch_is_not_a_caller_validation_error() -> (
    None
):
    class Client:
        def collection_exists(self, name):
            return True

        def get_collection(self, name):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=3, distance=models.Distance.DOT)
                    )
                ),
                payload_schema={},
            )

    with pytest.raises(store_module.PublicStoreUnavailableError):
        QdrantPublicKnowledgeStore(Client()).initialize()


def test_upsert_requires_initialize_before_any_client_call() -> None:
    class FailingClient:
        def collection_exists(self, name):
            raise AssertionError("client must not be called")

        def upsert(self, **kwargs):
            raise AssertionError("client must not be called")

    with pytest.raises(PublicStoreError, match="initialized"):
        QdrantPublicKnowledgeStore(FailingClient()).upsert([public_chunk()], [vector()])


def test_ensure_and_upsert_create_only_public_collection_in_memory() -> None:
    client = QdrantClient(":memory:")
    store = QdrantPublicKnowledgeStore(client)

    store.initialize()
    store.upsert([public_chunk()], [vector()])

    info = client.get_collection(PUBLIC_COLLECTION)
    assert info.config.params.vectors.size == 1024
    assert info.config.params.vectors.distance == models.Distance.COSINE
    stored = client.retrieve(
        PUBLIC_COLLECTION, [public_chunk().qdrant_point_id], with_payload=True
    )
    assert stored[0].id == public_chunk().qdrant_point_id
    assert stored[0].payload["data_class"] == "public"
    assert stored[0].payload["access_policy"] == "public_read"


def test_upsert_accepts_a_valid_public_corpus_chunk_without_headings() -> None:
    chunk = chunk_public_document(
        PublicKnowledgeDocument(
            source_id="synthetic-no-heading",
            title="Synthetic guidance",
            source_url="https://example.test/no-heading",
            source_version="2026-07",
            license_name="CC BY 4.0",
            text="Public synthetic guidance without a heading.",
        )
    )[0]
    assert chunk.section_path == ()
    client = QdrantClient(":memory:")
    store = QdrantPublicKnowledgeStore(client)

    store.initialize()
    store.upsert([chunk], [vector()])

    assert client.retrieve(PUBLIC_COLLECTION, [chunk.qdrant_point_id])


def test_existing_collection_with_wrong_dense_vector_is_rejected_before_index_calls() -> (
    None
):
    class ExistingClient:
        def collection_exists(self, name):
            return True

        def get_collection(self, name):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=3, distance=models.Distance.DOT)
                    )
                )
            )

        def create_payload_index(self, **kwargs):
            raise AssertionError(
                "indexes must not be changed after incompatible collection validation"
            )

    with pytest.raises(
        store_module.PublicStoreUnavailableError, match="vector configuration"
    ):
        QdrantPublicKnowledgeStore(ExistingClient()).ensure_public_collection()


def test_ensure_creates_all_required_keyword_indexes_for_an_existing_collection() -> (
    None
):
    class CapturingClient:
        def __init__(self):
            self.indexes = []

        def collection_exists(self, name):
            return True

        def get_collection(self, name):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(
                            size=1024, distance=models.Distance.COSINE
                        )
                    )
                ),
                payload_schema={},
            )

        def create_payload_index(self, **kwargs):
            self.indexes.append((kwargs["field_name"], kwargs["field_schema"]))

    client = CapturingClient()
    QdrantPublicKnowledgeStore(client).ensure_public_collection()

    assert client.indexes == [
        ("data_class", models.PayloadSchemaType.KEYWORD),
        ("access_policy", models.PayloadSchemaType.KEYWORD),
        ("source_id", models.PayloadSchemaType.KEYWORD),
        ("source_version", models.PayloadSchemaType.KEYWORD),
        ("topic", models.PayloadSchemaType.KEYWORD),
        ("effective_date", models.PayloadSchemaType.KEYWORD),
    ]


@pytest.mark.parametrize(
    "chunk",
    [
        lambda: replace(public_chunk(), data_class="private"),
        lambda: replace(public_chunk(), access_policy="restricted"),
        lambda: replace(public_chunk(), qdrant_point_id=str(uuid4())),
        lambda: replace(public_chunk(), qdrant_point_id="not-a-uuid"),
        lambda: replace(
            public_chunk(), qdrant_point_id=str(uuid5(UUID(int=0), "other"))
        ),
        lambda: replace(public_chunk(), source_id=""),
        lambda: replace(public_chunk(), section_path=("valid", 2)),
    ],
)
def test_upsert_rejects_non_public_or_invalid_point_before_client_call(chunk) -> None:
    class FailingClient:
        def upsert(self, *args, **kwargs):
            raise AssertionError("client must not be called")

    store = QdrantPublicKnowledgeStore(FailingClient())
    store._initialized = True

    with pytest.raises(ValueError):
        store.upsert([chunk()], [vector()])


def test_upsert_rejects_non_public_chunk_lookalike_before_client_call() -> None:
    class FailingClient:
        def upsert(self, *args, **kwargs):
            raise AssertionError("client must not be called")

    store = QdrantPublicKnowledgeStore(FailingClient())
    store._initialized = True

    with pytest.raises(ValueError, match="PublicKnowledgeChunk"):
        store.upsert([SimpleNamespace(**public_chunk().__dict__)], [vector()])


def test_search_applies_non_removable_public_filters_and_optional_topic() -> None:
    class CapturingClient:
        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {"points": []})()

    client = CapturingClient()
    store = QdrantPublicKnowledgeStore(client)

    assert store.search_dense(vector(), limit=2, topic="sampling") == []
    assert client.kwargs["collection_name"] == PUBLIC_COLLECTION
    conditions = client.kwargs["query_filter"].must
    assert [(item.key, item.match.value) for item in conditions] == [
        ("data_class", "public"),
        ("access_policy", "public_read"),
        ("topic", "sampling"),
    ]


@pytest.mark.parametrize("topic", ["", "  ", 3, ["sampling"]])
def test_search_rejects_invalid_topic_before_client_call(topic) -> None:
    class FailingClient:
        def query_points(self, *args, **kwargs):
            raise AssertionError("client must not be called")

    with pytest.raises(PublicStoreError):
        QdrantPublicKnowledgeStore(FailingClient()).search_dense(
            vector(), limit=1, topic=topic
        )


@pytest.mark.parametrize(
    "topic, effective_date",
    [("", "2026-07-01"), ("sampling", " "), (3, "2026-07-01"), ("sampling", [])],
)
def test_upsert_rejects_invalid_public_metadata_before_client_call(
    topic, effective_date
) -> None:
    class InitializedStore(QdrantPublicKnowledgeStore):
        def __init__(self):
            self._initialized = True
            self._client = SimpleNamespace(
                upsert=lambda **kwargs: (_ for _ in ()).throw(
                    AssertionError("client called")
                )
            )
            self.collection_name = PUBLIC_COLLECTION

    with pytest.raises(PublicStoreError):
        InitializedStore().upsert(
            [public_chunk()], [vector()], topic=topic, effective_date=effective_date
        )


def test_client_transport_failure_is_wrapped_as_unavailable() -> None:
    class BrokenClient:
        def query_points(self, **kwargs):
            raise RuntimeError("transport down")

    with pytest.raises(store_module.PublicStoreUnavailableError):
        QdrantPublicKnowledgeStore(BrokenClient()).search_dense(vector(), limit=1)


def test_malformed_client_response_is_wrapped_as_unavailable() -> None:
    class MalformedClient:
        def query_points(self, **kwargs):
            return object()

    with pytest.raises(store_module.PublicStoreUnavailableError):
        QdrantPublicKnowledgeStore(MalformedClient()).search_dense(vector(), limit=1)


@pytest.mark.parametrize(
    ("invalid_vector", "limit"),
    [
        ([0.1] * 1023, 1),
        ([nan] * 1024, 1),
        ([inf] * 1024, 1),
        ([1] * 1024, 1),
        ([True] * 1024, 1),
        ("x" * 1024, 1),
        (vector(), 0),
        (vector(), 41),
        (vector(), 1.5),
        (vector(), True),
    ],
)
def test_search_rejects_invalid_vector_or_limit_before_client_call(
    invalid_vector, limit
) -> None:
    class FailingClient:
        def query_points(self, *args, **kwargs):
            raise AssertionError("client must not be called")

    with pytest.raises(ValueError):
        QdrantPublicKnowledgeStore(FailingClient()).search_dense(invalid_vector, limit)


def test_search_rejects_non_public_payload_returned_by_client() -> None:
    class MaliciousClient:
        def query_points(self, **kwargs):
            point = models.ScoredPoint(
                id=public_chunk().qdrant_point_id,
                version=1,
                score=0.9,
                payload={
                    "chunk_id": public_chunk().chunk_id,
                    "source_id": "synthetic-guidance",
                    "source_url": "https://example.test/public-guidance",
                    "source_version": "2026-07",
                    "parent_title": "Synthetic public guidance",
                    "section_path": ["Water quality"],
                    "text": "should not escape",
                    "data_class": "private",
                    "access_policy": "restricted",
                    "topic": "sampling",
                    "effective_date": "2026-07-01",
                },
            )
            return type("Response", (), {"points": [point]})()

    with pytest.raises(ValueError, match="non-public"):
        QdrantPublicKnowledgeStore(MaliciousClient()).search_dense(vector(), limit=1)


@pytest.mark.parametrize(
    "score", [nan, inf, "not-a-score", pytest.param(10**10000, id="overflow-int")]
)
def test_search_rejects_malformed_score_from_client(score) -> None:
    class MalformedClient:
        def query_points(self, **kwargs):
            point = SimpleNamespace(
                id=public_chunk().qdrant_point_id, score=score, payload=public_payload()
            )
            return SimpleNamespace(points=[point])

    with pytest.raises(PublicStoreError):
        QdrantPublicKnowledgeStore(MalformedClient()).search_dense(vector(), limit=1)


def test_search_rejects_malformed_payload_shape_from_client() -> None:
    class MalformedClient:
        def query_points(self, **kwargs):
            payload = public_payload()
            payload["section_path"] = "not-a-sequence"
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id=public_chunk().qdrant_point_id, score=0.9, payload=payload
                    )
                ]
            )

    with pytest.raises(ValueError):
        QdrantPublicKnowledgeStore(MalformedClient()).search_dense(vector(), limit=1)


def public_payload() -> dict[str, object]:
    chunk = public_chunk()
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "source_url": chunk.source_url,
        "source_version": chunk.source_version,
        "parent_title": chunk.parent_title,
        "section_path": list(chunk.section_path),
        "text": chunk.text,
        "data_class": "public",
        "access_policy": "public_read",
        "topic": "sampling",
        "effective_date": "2026-07-01",
    }
