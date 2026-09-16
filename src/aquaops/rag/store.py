"""A Qdrant store that accepts and returns only public knowledge evidence."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import re
from typing import Protocol, TypeVar
from uuid import UUID, uuid5

from qdrant_client.http import models

from aquaops.rag.public_corpus import PublicKnowledgeChunk


PUBLIC_COLLECTION = "water_public_knowledge_v1"
_CONTRACT_COLLECTION_PREFIX = "aquaops_contract_"
_CONTRACT_COLLECTION = re.compile(r"^aquaops_contract_[0-9a-f]{32}$")
_CONTRACT_COLLECTION_SENTINEL = object()
_VECTOR_DIMENSION = 1024
_PUBLIC_DATA_CLASS = "public"
_PUBLIC_ACCESS_POLICY = "public_read"
_QDRANT_NAMESPACE = UUID("ce3be424-4f91-55a3-8ed0-dd56dc95c39a")
_PAYLOAD_INDEXES = (
    "data_class",
    "access_policy",
    "source_id",
    "source_version",
    "topic",
    "effective_date",
)
_T = TypeVar("_T")


class QdrantPublicClient(Protocol):
    """The narrow Qdrant client surface used by the public-only store."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(self, **kwargs: object) -> bool: ...

    def get_collection(self, collection_name: str) -> models.CollectionInfo: ...

    def create_payload_index(self, **kwargs: object) -> models.UpdateResult: ...

    def upsert(self, **kwargs: object) -> models.UpdateResult: ...

    def query_points(self, **kwargs: object) -> models.QueryResponse: ...


@dataclass(frozen=True)
class StoredChunk:
    """A public knowledge result together with its dense-search score."""

    chunk_id: str
    qdrant_point_id: str
    source_id: str
    source_url: str
    source_version: str
    parent_title: str
    section_path: tuple[str, ...]
    text: str
    data_class: str
    access_policy: str
    score: float


class PublicStoreError(Exception):
    """Base error for a rejected public-store operation."""


class PublicStoreValidationError(PublicStoreError, ValueError):
    """Caller input or a returned payload violates the public-only contract."""


class PublicStoreUnavailableError(PublicStoreError):
    """The Qdrant transport or its top-level response API is unavailable."""


class QdrantPublicKnowledgeStore:
    """Constrained Qdrant interface for the one public-read collection."""

    def __init__(
        self,
        client: QdrantPublicClient,
        collection_name: str = PUBLIC_COLLECTION,
        *,
        _contract_sentinel: object | None = None,
    ) -> None:
        if collection_name != PUBLIC_COLLECTION and not (
            _contract_sentinel is _CONTRACT_COLLECTION_SENTINEL
            and self._is_contract_collection(collection_name)
        ):
            raise PublicStoreValidationError(
                "only the fixed public collection is allowed"
            )
        self._client = client
        self.collection_name = collection_name
        self._initialized = False
        self._collection_created = False

    @classmethod
    def _for_contract_testing(
        cls, client: QdrantPublicClient, collection_name: str
    ) -> "QdrantPublicKnowledgeStore":
        """Create a store for an isolated, UUID-derived integration-test collection."""
        if not cls._is_contract_collection(collection_name):
            raise PublicStoreValidationError(
                "contract collections must use the aquaops_contract_ prefix"
            )
        return cls(
            client, collection_name, _contract_sentinel=_CONTRACT_COLLECTION_SENTINEL
        )

    def initialize(self) -> None:
        """Ensure the public collection once for this store instance."""
        if self._initialized:
            return
        self.ensure_public_collection()
        self._initialized = True

    def ensure_public_collection(self) -> None:
        """Create or validate the one public collection and its six keyword indexes."""
        exists = self._call_client(
            "check collection",
            lambda: self._client.collection_exists(self.collection_name),
        )
        if exists:
            self._validate_existing_collection()
        else:
            try:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=_VECTOR_DIMENSION, distance=models.Distance.COSINE
                    ),
                    metadata={
                        "data_class": _PUBLIC_DATA_CLASS,
                        "access_policy": _PUBLIC_ACCESS_POLICY,
                        "topic": "public classification label supplied to upsert (default: general)",
                        "effective_date": "public source date supplied to upsert (default: unspecified)",
                    },
                )
                self._collection_created = True
            except Exception as error:
                if not self._is_already_exists(error):
                    raise PublicStoreUnavailableError(
                        "Qdrant could not create the public collection"
                    ) from error
                self._validate_existing_collection()
        for field_name in _PAYLOAD_INDEXES:
            self._call_client(
                f"create payload index {field_name}",
                lambda field_name=field_name: self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                ),
            )

    def upsert(
        self,
        chunks: Sequence[PublicKnowledgeChunk],
        vectors: Sequence[Sequence[float]],
        *,
        topic: str = "general",
        effective_date: str = "unspecified",
    ) -> None:
        """Store public chunks after ``initialize`` has verified the collection."""
        self._require_initialized()
        if not isinstance(chunks, (list, tuple)) or not isinstance(
            vectors, (list, tuple)
        ):
            raise PublicStoreValidationError(
                "chunks and vectors must be lists or tuples"
            )
        if len(chunks) != len(vectors):
            raise PublicStoreValidationError(
                "chunks and vectors must have equal length"
            )
        topic = self._validate_metadata("topic", topic)
        effective_date = self._validate_metadata("effective_date", effective_date)
        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._validate_chunk(chunk)
            self._validate_vector(vector)
            try:
                points.append(
                    models.PointStruct(
                        id=chunk.qdrant_point_id,
                        vector=list(vector),
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "source_id": chunk.source_id,
                            "source_url": chunk.source_url,
                            "source_version": chunk.source_version,
                            "parent_title": chunk.parent_title,
                            "section_path": list(chunk.section_path),
                            "text": chunk.text,
                            "data_class": _PUBLIC_DATA_CLASS,
                            "access_policy": _PUBLIC_ACCESS_POLICY,
                            "topic": topic,
                            "effective_date": effective_date,
                        },
                    )
                )
            except Exception as error:
                raise PublicStoreValidationError(
                    "public chunk could not be encoded for Qdrant"
                ) from error
        if points:
            self._call_client(
                "upsert public chunks",
                lambda: self._client.upsert(
                    collection_name=self.collection_name, points=points, wait=True
                ),
            )

    def search_dense(
        self, vector: Sequence[float], limit: int, topic: str | None = None
    ) -> list[StoredChunk]:
        """Search with non-removable public-read filters."""
        self._validate_vector(vector)
        if type(limit) is not int or not 1 <= limit <= 40:
            raise PublicStoreValidationError(
                "limit must be an integer between 1 and 40"
            )
        if topic is not None:
            topic = self._validate_metadata("topic", topic)
        conditions = [
            models.FieldCondition(
                key="data_class", match=models.MatchValue(value=_PUBLIC_DATA_CLASS)
            ),
            models.FieldCondition(
                key="access_policy",
                match=models.MatchValue(value=_PUBLIC_ACCESS_POLICY),
            ),
        ]
        if topic is not None:
            conditions.append(
                models.FieldCondition(key="topic", match=models.MatchValue(value=topic))
            )
        response = self._call_client(
            "search public chunks",
            lambda: self._client.query_points(
                collection_name=self.collection_name,
                query=list(vector),
                query_filter=models.Filter(must=conditions),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ),
        )
        try:
            points = response.points  # type: ignore[attr-defined]
        except AttributeError as error:
            raise PublicStoreUnavailableError(
                "Qdrant returned an invalid search response"
            ) from error
        if not isinstance(points, (list, tuple)):
            raise PublicStoreUnavailableError(
                "Qdrant returned an invalid search response"
            )
        return [self._stored_chunk(point) for point in points]

    def _validate_existing_collection(self) -> None:
        collection = self._call_client(
            "read public collection",
            lambda: self._client.get_collection(self.collection_name),
        )
        try:
            vectors = collection.config.params.vectors
        except AttributeError as error:
            raise PublicStoreUnavailableError(
                "Qdrant returned an invalid collection response"
            ) from error
        if (
            isinstance(vectors, Mapping)
            or vectors.size != _VECTOR_DIMENSION
            or vectors.distance != models.Distance.COSINE
        ):
            raise PublicStoreUnavailableError(
                "public collection has incompatible dense vector configuration"
            )
        try:
            payload_schema = collection.payload_schema
        except AttributeError:
            payload_schema = {}
        if not isinstance(payload_schema, Mapping):
            raise PublicStoreUnavailableError(
                "Qdrant returned an invalid payload-schema response"
            )
        for field_name in _PAYLOAD_INDEXES:
            index = payload_schema.get(field_name)
            if index is None:
                continue
            try:
                data_type = index.data_type
            except AttributeError as error:
                raise PublicStoreUnavailableError(
                    "Qdrant returned an invalid payload index response"
                ) from error
            if data_type != models.PayloadSchemaType.KEYWORD:
                raise PublicStoreUnavailableError(
                    f"public collection index {field_name} is not keyword"
                )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise PublicStoreValidationError(
                "public store must be initialized before upsert"
            )

    @staticmethod
    def _validate_metadata(name: str, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise PublicStoreValidationError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _validate_chunk(chunk: PublicKnowledgeChunk) -> None:
        if not isinstance(chunk, PublicKnowledgeChunk):
            raise PublicStoreValidationError(
                "chunks must be PublicKnowledgeChunk instances"
            )
        if (
            chunk.data_class != _PUBLIC_DATA_CLASS
            or chunk.access_policy != _PUBLIC_ACCESS_POLICY
        ):
            raise PublicStoreValidationError(
                "only public, public_read chunks may be stored"
            )
        for field_name in (
            "source_id",
            "source_url",
            "source_version",
            "parent_title",
            "text",
        ):
            value = getattr(chunk, field_name)
            if not isinstance(value, str) or not value.strip():
                raise PublicStoreValidationError(
                    f"{field_name} must be a non-empty string"
                )
        if (
            not isinstance(chunk.chunk_id, str)
            or len(chunk.chunk_id) != 64
            or any(char not in "0123456789abcdef" for char in chunk.chunk_id)
        ):
            raise PublicStoreValidationError(
                "chunk_id must be a 64-character lowercase hexadecimal evidence id"
            )
        if not isinstance(chunk.qdrant_point_id, str):
            raise PublicStoreValidationError("qdrant_point_id must be a UUID v5")
        if not isinstance(chunk.section_path, (list, tuple)) or not all(
            isinstance(part, str) and part.strip() for part in chunk.section_path
        ):
            raise PublicStoreValidationError(
                "section_path must be a sequence of non-empty strings"
            )
        try:
            point_id = UUID(chunk.qdrant_point_id)
        except (TypeError, ValueError) as error:
            raise PublicStoreValidationError(
                "qdrant_point_id must be a UUID v5"
            ) from error
        if point_id.version != 5 or point_id != uuid5(
            _QDRANT_NAMESPACE, chunk.chunk_id
        ):
            raise PublicStoreValidationError(
                "qdrant_point_id must match the UUID v5 derived from chunk_id"
            )

    @staticmethod
    def _validate_vector(vector: Sequence[float]) -> None:
        if (
            not isinstance(vector, (list, tuple))
            or len(vector) != _VECTOR_DIMENSION
            or not all(type(value) is float and isfinite(value) for value in vector)
        ):
            raise PublicStoreValidationError(
                "dense vectors must contain 1024 finite float values"
            )

    @staticmethod
    def _stored_chunk(point: models.ScoredPoint) -> StoredChunk:
        try:
            payload = point.payload
            if not isinstance(payload, dict):
                raise PublicStoreValidationError(
                    "invalid public payload returned by Qdrant"
                )
            if (
                payload.get("data_class") != _PUBLIC_DATA_CLASS
                or payload.get("access_policy") != _PUBLIC_ACCESS_POLICY
            ):
                raise PublicStoreValidationError(
                    "non-public payload returned by Qdrant"
                )
            score = float(point.score)
            if not isfinite(score):
                raise PublicStoreValidationError("Qdrant returned a non-finite score")
            section_path = payload["section_path"]
            if not isinstance(section_path, (list, tuple)):
                raise PublicStoreValidationError(
                    "invalid public payload returned by Qdrant"
                )
            QdrantPublicKnowledgeStore._validate_chunk(
                PublicKnowledgeChunk(
                    chunk_id=payload["chunk_id"],
                    qdrant_point_id=str(point.id),
                    source_id=payload["source_id"],
                    source_url=payload["source_url"],
                    source_version=payload["source_version"],
                    parent_title=payload["parent_title"],
                    section_path=tuple(section_path),
                    text=payload["text"],
                )
            )
        except PublicStoreValidationError:
            raise
        except (
            AttributeError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ) as error:
            raise PublicStoreValidationError(
                "invalid public payload returned by Qdrant"
            ) from error
        return StoredChunk(
            chunk_id=payload["chunk_id"],
            qdrant_point_id=str(point.id),
            source_id=payload["source_id"],
            source_url=payload["source_url"],
            source_version=payload["source_version"],
            parent_title=payload["parent_title"],
            section_path=tuple(section_path),
            text=payload["text"],
            data_class=_PUBLIC_DATA_CLASS,
            access_policy=_PUBLIC_ACCESS_POLICY,
            score=score,
        )

    def _call_client(self, action: str, operation: Callable[[], _T]) -> _T:
        try:
            return operation()
        except PublicStoreError:
            raise
        except Exception as error:
            raise PublicStoreUnavailableError(f"Qdrant could not {action}") from error

    @staticmethod
    def _is_already_exists(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            return status_code == 409
        return "collection already exists" in str(error).lower()

    @staticmethod
    def _is_contract_collection(collection_name: str) -> bool:
        return (
            isinstance(collection_name, str)
            and _CONTRACT_COLLECTION.fullmatch(collection_name) is not None
        )
