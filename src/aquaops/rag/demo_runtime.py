"""Composition root for the real, public-only AquaOps RAG demo runtime."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Mapping
from typing import Protocol

from qdrant_client import QdrantClient

from aquaops.rag.demo_corpus import (
    DEMO_CORPUS_SHA256,
    load_verified_demo_chunks,
)
from aquaops.rag.hybrid import (
    HybridPublicRetriever,
    PublicEvidence,
    RetrievalResult,
)
from aquaops.rag.local_models import (
    LazyBgeM3Encoder,
    PublicBm25Retriever,
    PublicModelSpec,
    PublicRerankerBackendStatus,
    select_public_reranker,
)
from aquaops.rag.public_corpus import PublicKnowledgeChunk
from aquaops.rag.store import QdrantPublicKnowledgeStore, StoredChunk


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_DEMO_QDRANT_PATH = _PROJECT_ROOT / "artifacts" / "public-demo" / "qdrant"
PUBLIC_DEMO_VARIANT_NAMES = (
    "bm25",
    "dense",
    "hybrid_rrf",
    "hybrid_rerank",
)
PUBLIC_DEMO_CANDIDATE_LIMIT = 24
PUBLIC_DEMO_RERANK_LIMIT = 8
_PRIVATE_QUERY_MARKERS = (
    "未公开",
    "内部生产",
    "生产工单",
    "负责人姓名",
    "本地水厂",
    "原始数据文件",
    "监测数据库",
    "内部监测",
    "企业内部",
    "接口地址",
    "内部令牌",
    "生产系统",
    "修改泵频率",
)


class PublicQueryRefused(ValueError):
    """The query asks for non-public data or a non-read-only action."""


class PublicEncoder(Protocol):
    def encode_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[tuple[float, ...], ...]: ...

    def encode_query(self, query: str) -> list[float]: ...


class PublicReranker(Protocol):
    def rerank(
        self, query: str, candidates: list[StoredChunk]
    ) -> list[StoredChunk]: ...


@dataclass(frozen=True)
class PublicDemoRuntime:
    retriever: HybridPublicRetriever
    variants: Mapping[str, object]
    chunk_count: int
    corpus_sha256: str
    model_spec: PublicModelSpec
    reranker_backend_status: PublicRerankerBackendStatus = field(
        default_factory=lambda: PublicRerankerBackendStatus(
            requested_backend="torch",
            active_backend="torch",
            fallback_used=False,
            status_code="torch_selected",
        )
    )
    _closer: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def close(self) -> None:
        """Release only resources owned by this runtime factory."""
        if self._closer is not None:
            self._closer()


class _DenseRetriever:
    def __init__(
        self,
        encoder: PublicEncoder,
        store: QdrantPublicKnowledgeStore,
    ) -> None:
        self._encoder = encoder
        self._store = store

    def search(self, query: str, limit: int) -> list[StoredChunk]:
        return self._store.search_dense(self._encoder.encode_query(query), limit)


class _IdentityReranker:
    def rerank(self, query: str, candidates: list[StoredChunk]) -> list[StoredChunk]:
        return list(candidates)


class _SingleProviderRetriever:
    def __init__(self, provider: object) -> None:
        self._provider = provider

    def retrieve(self, query: str, answer_limit: int = 6) -> RetrievalResult:
        if type(answer_limit) is not int or not 1 <= answer_limit <= 6:
            raise ValueError("answer_limit must be between 1 and 6")
        candidates = self._provider.search(query, 40)
        return RetrievalResult(
            evidence=tuple(
                PublicEvidence(
                    chunk_id=chunk.chunk_id,
                    source_url=chunk.source_url,
                    source_version=chunk.source_version,
                    text=chunk.text,
                    score=float(chunk.score),
                )
                for chunk in candidates[:answer_limit]
            ),
            fused_candidate_ids=tuple(chunk.chunk_id for chunk in candidates),
        )


class _SafePublicRetriever:
    def __init__(
        self,
        retriever: object,
        *,
        fixed_hybrid_window: bool = False,
    ) -> None:
        self._retriever = retriever
        self._fixed_hybrid_window = fixed_hybrid_window

    def retrieve(self, query: str, answer_limit: int = 6) -> RetrievalResult:
        if not is_public_demo_query_allowed(query):
            raise PublicQueryRefused("public demo query was refused")
        if self._fixed_hybrid_window:
            return self._retriever.retrieve(
                query,
                candidate_limit=PUBLIC_DEMO_CANDIDATE_LIMIT,
                rerank_limit=PUBLIC_DEMO_RERANK_LIMIT,
                answer_limit=answer_limit,
            )
        return self._retriever.retrieve(query, answer_limit=answer_limit)


def build_public_demo_runtime(
    *,
    client: QdrantClient | None = None,
    encoder: PublicEncoder | None = None,
    reranker: PublicReranker | None = None,
    local_files_only: bool = False,
    reranker_backend: str = "torch",
) -> PublicDemoRuntime:
    """Index the one verified public corpus and return the injected hybrid runtime."""
    if type(local_files_only) is not bool:
        raise TypeError("local_files_only must be a boolean")
    model_spec = PublicModelSpec.default(local_files_only=local_files_only)
    public_encoder = encoder or LazyBgeM3Encoder(spec=model_spec)
    if reranker is None:
        public_reranker, reranker_status = select_public_reranker(
            requested_backend=reranker_backend,
            spec=model_spec,
        )
    else:
        if reranker_backend != "torch":
            raise TypeError("an injected reranker cannot select a backend")
        public_reranker = reranker
        reranker_status = PublicRerankerBackendStatus(
            requested_backend="torch",
            active_backend="torch",
            fallback_used=False,
            status_code="injected_reranker",
        )
    owns_client = client is None
    public_client = client or _open_fixed_local_client()
    chunks = load_verified_demo_chunks()
    vectors = public_encoder.encode_documents(tuple(chunk.text for chunk in chunks))
    store = QdrantPublicKnowledgeStore(public_client)
    store.initialize()
    store.upsert(chunks, vectors, topic="water-operations", effective_date="2026-08-11")
    lexical_chunks = tuple(_to_stored_chunk(chunk) for chunk in chunks)
    lexical = PublicBm25Retriever(lexical_chunks)
    dense = _DenseRetriever(public_encoder, store)
    hybrid_rrf = HybridPublicRetriever(
        lexical_retriever=lexical,
        dense_retriever=dense,
        reranker=_IdentityReranker(),
    )
    hybrid_rerank = HybridPublicRetriever(
        lexical_retriever=PublicBm25Retriever(lexical_chunks),
        dense_retriever=dense,
        reranker=public_reranker,
    )
    variants = MappingProxyType(
        {
            "bm25": _SafePublicRetriever(_SingleProviderRetriever(lexical)),
            "dense": _SafePublicRetriever(_SingleProviderRetriever(dense)),
            "hybrid_rrf": _SafePublicRetriever(hybrid_rrf, fixed_hybrid_window=True),
            "hybrid_rerank": _SafePublicRetriever(
                hybrid_rerank, fixed_hybrid_window=True
            ),
        }
    )
    return PublicDemoRuntime(
        retriever=variants["hybrid_rerank"],  # type: ignore[arg-type]
        variants=variants,
        chunk_count=len(chunks),
        corpus_sha256=DEMO_CORPUS_SHA256,
        model_spec=model_spec,
        reranker_backend_status=reranker_status,
        _closer=_one_shot_client_closer(public_client) if owns_client else None,
    )


def _open_fixed_local_client() -> QdrantClient:
    PUBLIC_DEMO_QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(PUBLIC_DEMO_QDRANT_PATH))


def _one_shot_client_closer(client: QdrantClient) -> Callable[[], None]:
    lock = Lock()
    closed = False

    def close() -> None:
        nonlocal closed
        with lock:
            if not closed:
                client.close()
                closed = True

    return close


def is_public_demo_query_allowed(query: object) -> bool:
    return (
        type(query) is str
        and bool(query.strip())
        and len(query) <= 512
        and not any(marker in query for marker in _PRIVATE_QUERY_MARKERS)
    )


def _to_stored_chunk(chunk: PublicKnowledgeChunk) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk.chunk_id,
        qdrant_point_id=chunk.qdrant_point_id,
        source_id=chunk.source_id,
        source_url=chunk.source_url,
        source_version=chunk.source_version,
        parent_title=chunk.parent_title,
        section_path=tuple(chunk.section_path),
        text=chunk.text,
        data_class=chunk.data_class,
        access_policy=chunk.access_policy,
        score=0.0,
    )
