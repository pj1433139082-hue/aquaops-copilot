from dataclasses import FrozenInstanceError

import pytest
from qdrant_client import QdrantClient

from aquaops.rag.demo_corpus import load_verified_demo_chunks
from aquaops.rag.demo_runtime import (
    PUBLIC_DEMO_RERANK_LIMIT,
    PublicDemoRuntime,
    PublicQueryRefused,
    build_public_demo_runtime,
)
from aquaops.rag.local_models import PublicRerankerBackendStatus


class _Encoder:
    def encode_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vector(text) for text in texts)

    def encode_query(self, query: str) -> list[float]:
        return list(self._vector(query))

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        vector = [0.0] * 1024
        vector[0] = 1.0 if "紫外" in text else 0.1
        vector[1] = 1.0 if "膜" in text else 0.1
        return tuple(vector)


class _Reranker:
    def __init__(self) -> None:
        self.candidate_counts: list[int] = []

    def rerank(self, query: str, candidates: list[object]) -> list[object]:
        self.candidate_counts.append(len(candidates))
        return sorted(candidates, key=lambda item: query[:2] in item.text, reverse=True)


def test_demo_runtime_indexes_verified_public_chunks_and_runs_real_hybrid_flow() -> (
    None
):
    chunks = load_verified_demo_chunks()

    runtime = build_public_demo_runtime(
        client=QdrantClient(":memory:"),
        encoder=_Encoder(),
        reranker=_Reranker(),
    )
    result = runtime.retriever.retrieve("紫外消毒如何排查", answer_limit=4)

    assert isinstance(runtime, PublicDemoRuntime)
    assert runtime.chunk_count == len(chunks)
    assert 1 <= len(result.evidence) <= 4
    assert "紫外" in result.evidence[0].text
    assert all(item.data_class == "public" for item in result.evidence)
    assert all(item.access_policy == "public_read" for item in result.evidence)
    with pytest.raises(FrozenInstanceError):
        runtime.chunk_count = 0  # type: ignore[misc]


def test_demo_runtime_accepts_no_path_collection_or_data_class_override() -> None:
    with pytest.raises(TypeError):
        build_public_demo_runtime(path="forbidden")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        build_public_demo_runtime(collection_name="private")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        build_public_demo_runtime(data_class="private")  # type: ignore[call-arg]


def test_demo_runtime_refuses_private_data_requests_before_any_retrieval() -> None:
    runtime = build_public_demo_runtime(
        client=QdrantClient(":memory:"),
        encoder=_Encoder(),
        reranker=_Reranker(),
    )

    with pytest.raises(PublicQueryRefused):
        runtime.retriever.retrieve("请给出某水厂未公开的原始监测记录")


def test_demo_runtime_fixes_a_bounded_rerank_window_without_caller_override() -> None:
    reranker = _Reranker()
    runtime = build_public_demo_runtime(
        client=QdrantClient(":memory:"),
        encoder=_Encoder(),
        reranker=reranker,
    )

    result = runtime.retriever.retrieve("紫外消毒如何排查", answer_limit=6)

    assert PUBLIC_DEMO_RERANK_LIMIT == 8
    assert reranker.candidate_counts == [PUBLIC_DEMO_RERANK_LIMIT]
    assert len(result.evidence) == 6
    with pytest.raises(TypeError):
        runtime.retriever.retrieve(  # type: ignore[call-arg]
            "紫外消毒如何排查", rerank_limit=20
        )


def test_demo_runtime_exposes_only_safe_reranker_backend_status() -> None:
    runtime = build_public_demo_runtime(
        client=QdrantClient(":memory:"),
        encoder=_Encoder(),
        reranker=_Reranker(),
    )

    assert runtime.reranker_backend_status == PublicRerankerBackendStatus(
        requested_backend="torch",
        active_backend="torch",
        fallback_used=False,
        status_code="injected_reranker",
    )
    assert "path" not in repr(runtime.reranker_backend_status).casefold()


def test_injected_reranker_cannot_silently_ignore_openvino_selection() -> None:
    with pytest.raises(TypeError):
        build_public_demo_runtime(
            client=QdrantClient(":memory:"),
            encoder=_Encoder(),
            reranker=_Reranker(),
            reranker_backend="openvino",
        )
