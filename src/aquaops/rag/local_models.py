"""Lazy, pinned local model adapters for the verified public demo corpus."""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite
import re
from threading import Lock
from typing import Protocol

from rank_bm25 import BM25Plus

from aquaops.rag.store import StoredChunk


BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_RERANKER_MODEL_ID = "BAAI/bge-reranker-base"
BGE_RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
PUBLIC_RERANKER_BACKENDS = frozenset({"torch", "openvino"})
PUBLIC_EMBEDDING_DIMENSION = 1024
_MAX_MODEL_BATCH = 256
_MAX_QUERY_CACHE = 256
_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u3400-\u9fff]+")


class PublicModelContractError(ValueError):
    """A local model or lexical result violated the fixed public contract."""


class EmbeddingBackend(Protocol):
    def encode(self, sentences: list[str], **kwargs: object) -> object: ...


class RerankerBackend(Protocol):
    def predict(self, pairs: list[list[str]], **kwargs: object) -> object: ...


@dataclass(frozen=True)
class PublicRerankerBackendStatus:
    """Safe aggregate startup state; it never carries paths or exception details."""

    requested_backend: str
    active_backend: str
    fallback_used: bool
    status_code: str

    def __post_init__(self) -> None:
        valid_states = {
            ("torch", "torch", False, "torch_selected"),
            ("torch", "torch", False, "injected_reranker"),
            ("openvino", "openvino", False, "openvino_ready"),
            (
                "openvino",
                "torch",
                True,
                "openvino_unavailable_fallback_torch",
            ),
        }
        if (
            type(self.requested_backend) is not str
            or type(self.active_backend) is not str
            or type(self.fallback_used) is not bool
            or type(self.status_code) is not str
            or (
                self.requested_backend,
                self.active_backend,
                self.fallback_used,
                self.status_code,
            )
            not in valid_states
        ):
            raise PublicModelContractError("public reranker backend status is invalid")


@dataclass(frozen=True)
class PublicModelSpec:
    embedding_model_id: str
    embedding_revision: str
    reranker_model_id: str
    reranker_revision: str
    embedding_dimension: int
    trust_remote_code: bool
    local_files_only: bool

    def __post_init__(self) -> None:
        if (
            self.embedding_model_id != BGE_M3_MODEL_ID
            or self.embedding_revision != BGE_M3_REVISION
            or self.reranker_model_id != BGE_RERANKER_MODEL_ID
            or self.reranker_revision != BGE_RERANKER_REVISION
            or self.embedding_dimension != PUBLIC_EMBEDDING_DIMENSION
            or self.trust_remote_code is not False
            or type(self.local_files_only) is not bool
        ):
            raise PublicModelContractError("public demo model specification is fixed")

    @classmethod
    def default(cls, *, local_files_only: bool = False) -> "PublicModelSpec":
        return cls(
            embedding_model_id=BGE_M3_MODEL_ID,
            embedding_revision=BGE_M3_REVISION,
            reranker_model_id=BGE_RERANKER_MODEL_ID,
            reranker_revision=BGE_RERANKER_REVISION,
            embedding_dimension=PUBLIC_EMBEDDING_DIMENSION,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )


class LazyBgeM3Encoder:
    """Load the pinned BGE-M3 model only on first explicit encode."""

    def __init__(
        self,
        *,
        spec: PublicModelSpec | None = None,
        loader: Callable[[], EmbeddingBackend] | None = None,
    ) -> None:
        self.spec = spec or PublicModelSpec.default()
        self._loader = loader or self._load_backend
        self._backend: EmbeddingBackend | None = None
        self._query_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._query_lock = Lock()

    def encode_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return self._encode(texts)

    def encode_query(self, query: str) -> list[float]:
        if type(query) is not str:
            raise PublicModelContractError("public embedding input is invalid")
        with self._query_lock:
            cached = self._query_cache.get(query)
            if cached is not None:
                self._query_cache.move_to_end(query)
                return list(cached)
            vector = self._encode((query,))[0]
            self._query_cache[query] = vector
            if len(self._query_cache) > _MAX_QUERY_CACHE:
                self._query_cache.popitem(last=False)
            return list(vector)

    def _encode(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if (
            type(texts) is not tuple
            or not 1 <= len(texts) <= _MAX_MODEL_BATCH
            or any(
                type(text) is not str or not text.strip() or len(text) > 20_000
                for text in texts
            )
        ):
            raise PublicModelContractError("public embedding input is invalid")
        try:
            output = self._model().encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            rows = output.tolist() if hasattr(output, "tolist") else output
        except PublicModelContractError:
            raise
        except Exception as error:
            raise PublicModelContractError(
                "public embedding model is unavailable"
            ) from error
        if type(rows) is not list or len(rows) != len(texts):
            raise PublicModelContractError("public embedding output is invalid")
        vectors: list[tuple[float, ...]] = []
        for row in rows:
            if (
                type(row) is not list
                or len(row) != self.spec.embedding_dimension
                or any(type(value) is not float or not isfinite(value) for value in row)
            ):
                raise PublicModelContractError("public embedding output is invalid")
            vectors.append(tuple(row))
        return tuple(vectors)

    def _model(self) -> EmbeddingBackend:
        if self._backend is None:
            try:
                self._backend = self._loader()
            except Exception as error:
                raise PublicModelContractError(
                    "public embedding model is unavailable"
                ) from error
        return self._backend

    def _load_backend(self) -> EmbeddingBackend:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.spec.embedding_model_id,
            revision=self.spec.embedding_revision,
            trust_remote_code=self.spec.trust_remote_code,
            local_files_only=self.spec.local_files_only,
        )


class LazyBgeReranker:
    """Load the pinned cross encoder lazily and only reorder supplied candidates."""

    def __init__(
        self,
        *,
        spec: PublicModelSpec | None = None,
        loader: Callable[[], RerankerBackend] | None = None,
    ) -> None:
        self.spec = spec or PublicModelSpec.default()
        self._loader = loader or self._load_backend
        self._backend: RerankerBackend | None = None

    def rerank(self, query: str, candidates: list[StoredChunk]) -> list[StoredChunk]:
        if (
            type(query) is not str
            or not query.strip()
            or len(query) > 512
            or type(candidates) is not list
            or not 1 <= len(candidates) <= 20
            or any(type(candidate) is not StoredChunk for candidate in candidates)
        ):
            raise PublicModelContractError("public reranker input is invalid")
        snapshots = tuple(candidates)
        try:
            output = self._model().predict(
                [[query, candidate.text] for candidate in snapshots],
                show_progress_bar=False,
            )
            scores = output.tolist() if hasattr(output, "tolist") else output
        except PublicModelContractError:
            raise
        except Exception as error:
            raise PublicModelContractError("public reranker is unavailable") from error
        if (
            type(scores) is not list
            or len(scores) != len(snapshots)
            or any(
                type(score) not in (int, float) or not isfinite(float(score))
                for score in scores
            )
        ):
            raise PublicModelContractError("public reranker output is invalid")
        order = sorted(
            range(len(snapshots)), key=lambda index: (-float(scores[index]), index)
        )
        return [snapshots[index] for index in order]

    def _model(self) -> RerankerBackend:
        if self._backend is None:
            try:
                self._backend = self._loader()
            except Exception as error:
                raise PublicModelContractError(
                    "public reranker is unavailable"
                ) from error
        return self._backend

    def _load_backend(self) -> RerankerBackend:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            self.spec.reranker_model_id,
            revision=self.spec.reranker_revision,
            trust_remote_code=self.spec.trust_remote_code,
            local_files_only=self.spec.local_files_only,
        )


def select_public_reranker(
    *,
    requested_backend: str = "torch",
    spec: PublicModelSpec | None = None,
    torch_loader: Callable[[], RerankerBackend] | None = None,
    openvino_loader: Callable[[], RerankerBackend] | None = None,
) -> tuple[LazyBgeReranker, PublicRerankerBackendStatus]:
    """Select one fixed public reranker and safely fall back during startup."""
    if type(requested_backend) is not str or requested_backend not in (
        "torch",
        "openvino",
    ):
        raise PublicModelContractError("public reranker backend is fixed")
    model_spec = spec or PublicModelSpec.default()
    if requested_backend == "torch":
        reranker = LazyBgeReranker(spec=model_spec, loader=torch_loader)
        return reranker, PublicRerankerBackendStatus(
            requested_backend="torch",
            active_backend="torch",
            fallback_used=False,
            status_code="torch_selected",
        )
    try:
        backend = (openvino_loader or _load_verified_openvino_backend)()
        _probe_reranker_backend(backend)
    except Exception:
        fallback = LazyBgeReranker(spec=model_spec, loader=torch_loader)
        return fallback, PublicRerankerBackendStatus(
            requested_backend="openvino",
            active_backend="torch",
            fallback_used=True,
            status_code="openvino_unavailable_fallback_torch",
        )
    return (
        LazyBgeReranker(spec=model_spec, loader=lambda: backend),
        PublicRerankerBackendStatus(
            requested_backend="openvino",
            active_backend="openvino",
            fallback_used=False,
            status_code="openvino_ready",
        ),
    )


def _probe_reranker_backend(backend: RerankerBackend) -> None:
    output = backend.predict(
        [["公开后端自检", "公开水务运维知识"]],
        show_progress_bar=False,
    )
    scores = output.tolist() if hasattr(output, "tolist") else output
    if (
        type(scores) is not list
        or len(scores) != 1
        or type(scores[0]) not in (int, float)
        or not isfinite(float(scores[0]))
    ):
        raise PublicModelContractError("public reranker startup probe failed")


def _load_verified_openvino_backend() -> RerankerBackend:
    from aquaops.rag.openvino_reranker import load_verified_openvino_reranker

    return load_verified_openvino_reranker()


class PublicBm25Retriever:
    """In-memory BM25 over the exact immutable verified public chunk set."""

    def __init__(self, chunks: tuple[StoredChunk, ...]) -> None:
        if (
            type(chunks) is not tuple
            or not chunks
            or len(chunks) > 10_000
            or any(type(chunk) is not StoredChunk for chunk in chunks)
            or any(
                chunk.data_class != "public" or chunk.access_policy != "public_read"
                for chunk in chunks
            )
            or len({chunk.chunk_id for chunk in chunks}) != len(chunks)
        ):
            raise PublicModelContractError("public BM25 corpus is invalid")
        self._chunks = chunks
        self._index = BM25Plus([_lexical_tokens(chunk.text) for chunk in chunks])

    def search(self, query: str, limit: int) -> list[StoredChunk]:
        if (
            type(query) is not str
            or not query.strip()
            or len(query) > 512
            or type(limit) is not int
            or not 1 <= limit <= 40
        ):
            raise PublicModelContractError("public BM25 request is invalid")
        try:
            raw_scores = self._index.get_scores(_lexical_tokens(query))
            scores = (
                raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
            )
        except Exception as error:
            raise PublicModelContractError(
                "public BM25 search is unavailable"
            ) from error
        if type(scores) is not list or len(scores) != len(self._chunks):
            raise PublicModelContractError("public BM25 output is invalid")
        ranked: list[tuple[int, float]] = []
        for index, score in enumerate(scores):
            if type(score) not in (int, float) or not isfinite(float(score)):
                raise PublicModelContractError("public BM25 output is invalid")
            ranked.append((index, float(score)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return [
            replace(self._chunks[index], score=score) for index, score in ranked[:limit]
        ]


def _lexical_tokens(text: str) -> list[str]:
    tokens = [word.casefold() for word in _WORD.findall(text)]
    for sequence in _CJK.findall(text):
        characters = list(sequence)
        tokens.extend(characters)
        tokens.extend(
            "".join(characters[index : index + 2])
            for index in range(len(characters) - 1)
        )
    return tokens or [text.casefold()]
