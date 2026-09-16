from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from math import nan
from uuid import UUID, uuid5

import pytest

from aquaops.rag.local_models import (
    BGE_M3_MODEL_ID,
    BGE_RERANKER_MODEL_ID,
    PUBLIC_EMBEDDING_DIMENSION,
    LazyBgeM3Encoder,
    LazyBgeReranker,
    PublicBm25Retriever,
    PublicModelContractError,
    PublicModelSpec,
    PublicRerankerBackendStatus,
    select_public_reranker,
)
from aquaops.rag.store import StoredChunk


_NAMESPACE = UUID("ce3be424-4f91-55a3-8ed0-dd56dc95c39a")


def _stored(label: str, text: str, score: float = 0.0) -> StoredChunk:
    chunk_id = sha256(label.encode()).hexdigest()
    return StoredChunk(
        chunk_id=chunk_id,
        qdrant_point_id=str(uuid5(_NAMESPACE, chunk_id)),
        source_id="public-demo",
        source_url="https://epa.gov/public-demo",
        source_version="v1",
        parent_title="Public demo",
        section_path=("Public", label),
        text=text,
        data_class="public",
        access_policy="public_read",
        score=score,
    )


class _EmbeddingModel:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append((sentences, kwargs))
        return self.vectors[: len(sentences)]


class _CrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[list[str]], dict[str, object]]] = []

    def predict(self, pairs: list[list[str]], **kwargs: object) -> list[float]:
        self.calls.append((pairs, kwargs))
        return self.scores[: len(pairs)]


def test_public_model_spec_is_fixed_and_immutable() -> None:
    spec = PublicModelSpec.default()

    assert spec.embedding_model_id == BGE_M3_MODEL_ID
    assert spec.reranker_model_id == BGE_RERANKER_MODEL_ID
    assert spec.embedding_dimension == 1024
    assert spec.trust_remote_code is False
    with pytest.raises(FrozenInstanceError):
        spec.embedding_model_id = "changed"  # type: ignore[misc]


def test_bge_encoder_loads_lazily_and_requires_normalized_finite_1024_vectors() -> None:
    vector = [0.0] * PUBLIC_EMBEDDING_DIMENSION
    vector[0] = 1.0
    backend = _EmbeddingModel([vector, vector])
    loads: list[str] = []
    encoder = LazyBgeM3Encoder(loader=lambda: loads.append("loaded") or backend)

    assert loads == []
    encoded = encoder.encode_documents(("公开膜污染知识", "公开紫外消毒知识"))

    assert loads == ["loaded"]
    assert encoded == (tuple(vector), tuple(vector))
    assert backend.calls[0][1] == {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }


def test_bge_encoder_reuses_a_validated_query_vector_without_sharing_mutation() -> None:
    vector = [0.0] * PUBLIC_EMBEDDING_DIMENSION
    vector[0] = 1.0
    backend = _EmbeddingModel([vector])
    encoder = LazyBgeM3Encoder(loader=lambda: backend)

    first = encoder.encode_query("公开紫外消毒查询")
    first[0] = 0.0
    second = encoder.encode_query("公开紫外消毒查询")

    assert len(backend.calls) == 1
    assert second[0] == 1.0


@pytest.mark.parametrize(
    "vector",
    [
        [0.0] * (PUBLIC_EMBEDDING_DIMENSION - 1),
        [nan] * PUBLIC_EMBEDDING_DIMENSION,
        [1] * PUBLIC_EMBEDDING_DIMENSION,
    ],
)
def test_bge_encoder_rejects_malformed_backend_vectors(vector: list[float]) -> None:
    encoder = LazyBgeM3Encoder(loader=lambda: _EmbeddingModel([vector]))

    with pytest.raises(PublicModelContractError):
        encoder.encode_query("公开查询")


def test_reranker_only_reorders_supplied_immutable_candidates() -> None:
    first = _stored("first", "紫外灯管与透过率")
    second = _stored("second", "膜污染与跨膜压差")
    backend = _CrossEncoder([0.1, 0.9])
    loads: list[str] = []
    reranker = LazyBgeReranker(loader=lambda: loads.append("loaded") or backend)

    reranked = reranker.rerank("膜污染怎么排查", [first, second])

    assert loads == ["loaded"]
    assert reranked == [second, first]
    assert backend.calls[0][0] == [
        ["膜污染怎么排查", first.text],
        ["膜污染怎么排查", second.text],
    ]
    assert reranked[0].score == second.score


def test_reranker_rejects_nonfinite_or_wrong_length_scores() -> None:
    candidates = [_stored("first", "公开文本"), _stored("second", "另一公开文本")]

    with pytest.raises(PublicModelContractError):
        LazyBgeReranker(loader=lambda: _CrossEncoder([0.1])).rerank("查询", candidates)
    with pytest.raises(PublicModelContractError):
        LazyBgeReranker(loader=lambda: _CrossEncoder([0.1, nan])).rerank(
            "查询", candidates
        )


def test_torch_reranker_is_default_and_remains_lazy() -> None:
    backend = _CrossEncoder([0.9])
    loads: list[str] = []

    reranker, status = select_public_reranker(
        requested_backend="torch",
        torch_loader=lambda: loads.append("torch") or backend,
    )

    assert status == PublicRerankerBackendStatus(
        requested_backend="torch",
        active_backend="torch",
        fallback_used=False,
        status_code="torch_selected",
    )
    assert loads == []
    reranker.rerank("公开查询", [_stored("one", "公开文本")])
    assert loads == ["torch"]


def test_openvino_reranker_runs_startup_probe_and_becomes_active() -> None:
    backend = _CrossEncoder([0.7, 0.2])

    reranker, status = select_public_reranker(
        requested_backend="openvino",
        openvino_loader=lambda: backend,
        torch_loader=lambda: (_ for _ in ()).throw(AssertionError("no fallback")),
    )

    assert status == PublicRerankerBackendStatus(
        requested_backend="openvino",
        active_backend="openvino",
        fallback_used=False,
        status_code="openvino_ready",
    )
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == [["公开后端自检", "公开水务运维知识"]]
    reranker.rerank("公开查询", [_stored("one", "公开文本")])
    assert len(backend.calls) == 2


@pytest.mark.parametrize("failure", [RuntimeError("missing"), ValueError("tampered")])
def test_openvino_startup_failure_falls_back_to_lazy_torch_without_details(
    failure: Exception,
) -> None:
    torch_backend = _CrossEncoder([0.4])
    torch_loads: list[str] = []

    def _failed_openvino():
        raise failure

    reranker, status = select_public_reranker(
        requested_backend="openvino",
        openvino_loader=_failed_openvino,
        torch_loader=lambda: torch_loads.append("torch") or torch_backend,
    )

    assert status == PublicRerankerBackendStatus(
        requested_backend="openvino",
        active_backend="torch",
        fallback_used=True,
        status_code="openvino_unavailable_fallback_torch",
    )
    assert failure.args[0] not in repr(status)
    assert torch_loads == []
    reranker.rerank("公开查询", [_stored("one", "公开文本")])
    assert torch_loads == ["torch"]


def test_reranker_backend_rejects_unknown_or_non_string_values() -> None:
    for value in ("onnx", "OPENVINO", "", None, 1):
        with pytest.raises(PublicModelContractError):
            select_public_reranker(requested_backend=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values",
    [
        ("torch", "openvino", False, "torch_selected"),
        ("openvino", "torch", False, "openvino_unavailable_fallback_torch"),
        ("openvino", "openvino", True, "openvino_ready"),
        ("torch", "torch", True, "injected_reranker"),
    ],
)
def test_reranker_backend_status_rejects_inconsistent_states(values) -> None:
    with pytest.raises(PublicModelContractError):
        PublicRerankerBackendStatus(*values)


def test_bm25_retriever_returns_public_candidates_without_changing_identity() -> None:
    uv = _stored("uv", "紫外 消毒 灯管 透过率 清洗")
    membrane = _stored("membrane", "膜 污染 跨膜压差 通量 清洗")
    retriever = PublicBm25Retriever((uv, membrane))

    result = retriever.search("膜污染和跨膜压差", limit=2)

    assert result[0].chunk_id == membrane.chunk_id
    assert result[0].text == membrane.text
    assert result[0].data_class == "public"
    assert result[0].access_policy == "public_read"
    assert result[0] != replace(membrane, score=nan)
