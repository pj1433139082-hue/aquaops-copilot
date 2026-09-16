from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from math import nan
from uuid import UUID, uuid4, uuid5

import pytest

from aquaops.rag.hybrid import (
    HybridPublicRetriever,
    HybridRetrievalError,
    HybridRetrievalValidationError,
    PublicEvidence,
    RetrievalUnavailable,
)
from aquaops.rag.retrieve import fuse_public_rrf
from aquaops.rag.store import StoredChunk


_QDRANT_NAMESPACE = UUID("ce3be424-4f91-55a3-8ed0-dd56dc95c39a")


def _chunk_id(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _chunk(label: str, score: float = 1.0) -> StoredChunk:
    chunk_id = _chunk_id(label)
    return StoredChunk(
        chunk_id=chunk_id,
        qdrant_point_id=str(uuid5(_QDRANT_NAMESPACE, chunk_id)),
        source_id="synthetic-source",
        source_url=f"https://example.invalid/{label}",
        source_version="v1",
        parent_title="Synthetic public title",
        section_path=("Synthetic", "Section"),
        text=f"synthetic public evidence {label}",
        data_class="public",
        access_policy="public_read",
        score=score,
    )


class _Recorder:
    def __init__(self, results: list[StoredChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[StoredChunk]:
        self.calls.append((query, limit))
        return self.results[:limit]


class _Reranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[StoredChunk]]] = []

    def rerank(self, query: str, candidates: list[StoredChunk]) -> list[StoredChunk]:
        self.calls.append((query, candidates))
        return list(reversed(candidates))


def _retriever(
    lexical: _Recorder | None = None,
    dense: _Recorder | None = None,
    reranker: _Reranker | None = None,
) -> HybridPublicRetriever:
    return HybridPublicRetriever(
        lexical_retriever=lexical,
        dense_retriever=dense,
        reranker=reranker,
    )


def test_reranker_receives_only_top_twenty_and_result_is_limited_to_four() -> None:
    lexical = _Recorder([_chunk(f"lex-{index}") for index in range(40)])
    dense = _Recorder([_chunk(f"dense-{index}") for index in range(40)])
    reranker = _Reranker()

    result = _retriever(lexical, dense, reranker).retrieve(
        "pump alert", candidate_limit=40, rerank_limit=20, answer_limit=4
    )

    assert lexical.calls == [("pump alert", 40)]
    assert dense.calls == [("pump alert", 40)]
    assert len(reranker.calls[0][1]) == 20
    assert len(result.evidence) == 4
    assert len(result.fused_candidate_ids) == 80
    assert not hasattr(result.evidence[0], "section_path")
    assert not hasattr(result.evidence[0], "raw")


@pytest.mark.parametrize(
    ("query", "kwargs"),
    [
        (" ", {}),
        ("alert", {"candidate_limit": True}),
        ("alert", {"candidate_limit": 19, "rerank_limit": 20}),
        ("alert", {"candidate_limit": 41}),
        ("alert", {"answer_limit": 0}),
    ],
)
def test_invalid_input_does_not_call_providers(
    query: str, kwargs: dict[str, object]
) -> None:
    lexical = _Recorder([_chunk("lex")])
    dense = _Recorder([_chunk("dense")])
    reranker = _Reranker()

    with pytest.raises(HybridRetrievalValidationError):
        _retriever(lexical, dense, reranker).retrieve(query, **kwargs)

    assert lexical.calls == []
    assert dense.calls == []
    assert reranker.calls == []


def test_duplicate_candidate_is_fused_once() -> None:
    duplicate = _chunk("same")
    lexical = _Recorder([duplicate, _chunk("lex-only")])
    dense = _Recorder([replace(duplicate, score=0.2), _chunk("dense-only")])
    reranker = _Reranker()

    result = _retriever(lexical, dense, reranker).retrieve(
        "alert", candidate_limit=3, rerank_limit=3, answer_limit=3
    )

    assert result.fused_candidate_ids.count(_chunk_id("same")) == 1
    assert {item.chunk_id for item in reranker.calls[0][1]} == {
        _chunk_id("same"),
        _chunk_id("lex-only"),
        _chunk_id("dense-only"),
    }


def test_each_source_is_stably_deduplicated_before_its_candidate_limit() -> None:
    class _IgnoringLimitRecorder(_Recorder):
        def search(self, query: str, limit: int) -> list[StoredChunk]:
            self.calls.append((query, limit))
            return self.results

    first = _chunk("first")
    lexical = _IgnoringLimitRecorder(
        [first, replace(first, score=0.2), _chunk("lex-last")]
    )
    reranker = _Reranker()

    _retriever(lexical, _Recorder([]), reranker).retrieve(
        "alert", candidate_limit=3, rerank_limit=2, answer_limit=2
    )

    assert [item.chunk_id for item in reranker.calls[0][1]] == [
        _chunk_id("first"),
        _chunk_id("lex-last"),
    ]


def test_provider_over_candidate_limit_is_rejected_before_validation() -> None:
    class _IgnoringLimitRecorder(_Recorder):
        def search(self, query: str, limit: int) -> list[StoredChunk]:
            self.calls.append((query, limit))
            return self.results

    lexical = _IgnoringLimitRecorder([_chunk(f"lex-{index}") for index in range(41)])
    reranker = _Reranker()

    with pytest.raises(HybridRetrievalError, match="candidate limit"):
        _retriever(lexical, _Recorder([]), reranker).retrieve("alert")

    assert reranker.calls == []


def test_list_subclass_provider_result_is_rejected_without_iterating_elements() -> None:
    class _HugeResult(list[StoredChunk]):
        def __len__(self) -> int:
            return 1_000_000

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("huge provider result must not be iterated")

    class _HugeProvider:
        def search(self, query: str, limit: int) -> list[StoredChunk]:
            return _HugeResult()

    with pytest.raises(HybridRetrievalError, match="exact list"):
        _retriever(_HugeProvider(), _Recorder([]), _Reranker()).retrieve("alert")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid",
    [
        replace(_chunk("private"), data_class="private"),
        replace(_chunk("bad-score"), score=nan),
    ],
)
def test_non_public_or_non_finite_provider_candidate_is_rejected(
    invalid: StoredChunk,
) -> None:
    lexical = _Recorder([invalid])
    dense = _Recorder([_chunk("dense")])

    with pytest.raises(HybridRetrievalError):
        _retriever(lexical, dense, _Reranker()).retrieve("alert")


def test_reranker_cannot_inject_a_new_candidate() -> None:
    class _InjectingReranker(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            super().rerank(query, candidates)
            return [*candidates, _chunk("injected")]

    with pytest.raises(HybridRetrievalError, match="subset"):
        _retriever(
            _Recorder([_chunk("lex")]),
            _Recorder([_chunk("dense")]),
            _InjectingReranker(),
        ).retrieve("alert")


def test_reranker_cannot_append_a_candidate_to_its_mutable_input_list() -> None:
    class _AppendingReranker(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            candidates.append(_chunk("appended"))
            return candidates

    with pytest.raises(HybridRetrievalError, match="subset"):
        _retriever(
            _Recorder([_chunk("lex")]), _Recorder([]), _AppendingReranker()
        ).retrieve("alert")


def test_list_subclass_reranker_result_is_rejected_without_iterating_elements() -> None:
    class _HugeResult(list[StoredChunk]):
        def __len__(self) -> int:
            return 1_000_000

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("huge reranker result must not be iterated")

    class _HugeReranker(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            return _HugeResult()

    with pytest.raises(HybridRetrievalError, match="exact list"):
        _retriever(_Recorder([_chunk("lex")]), _Recorder([]), _HugeReranker()).retrieve(
            "alert"
        )


def test_reranker_cannot_mutate_frozen_candidate_in_place() -> None:
    class _MutatingReranker(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            object.__setattr__(candidates[0], "text", "poisoned evidence")
            return candidates

    with pytest.raises(HybridRetrievalError, match="snapshot"):
        _retriever(
            _Recorder([_chunk("lex")]), _Recorder([]), _MutatingReranker()
        ).retrieve("alert")


@pytest.mark.parametrize(
    "replacement",
    [
        lambda chunk: replace(chunk, text="changed evidence"),
        lambda chunk: replace(chunk, source_url="https://example.invalid/changed"),
        lambda chunk: replace(chunk, source_version="v2"),
        lambda chunk: replace(chunk, section_path=("Changed",)),
        lambda chunk: replace(chunk, score=0.2),
    ],
)
def test_reranker_cannot_replace_an_allowed_id_with_changed_content(
    replacement: object,
) -> None:
    class _ReplacingReranker(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            return [replacement(candidates[0])]  # type: ignore[operator]

    with pytest.raises(HybridRetrievalError, match="snapshot"):
        _retriever(
            _Recorder([_chunk("lex")]), _Recorder([]), _ReplacingReranker()
        ).retrieve("alert")


def test_cross_source_same_id_with_conflicting_evidence_is_rejected() -> None:
    lexical = _Recorder([_chunk("same", score=0.9)])
    dense = _Recorder([replace(_chunk("same", score=0.1), text="conflicting evidence")])

    with pytest.raises(HybridRetrievalError, match="conflicting"):
        _retriever(lexical, dense, _Reranker()).retrieve("alert")


def test_public_rrf_rejects_conflicting_evidence_within_one_ranked_list() -> None:
    original = _chunk("same-list")
    changed = replace(original, text="changed same-list evidence")

    with pytest.raises(ValueError, match="conflicting"):
        fuse_public_rrf([original, changed])


def test_provider_candidate_must_preserve_the_public_store_point_identity() -> None:
    invalid = replace(_chunk("invalid-identity"), qdrant_point_id=str(uuid4()))

    with pytest.raises(HybridRetrievalError, match="identity"):
        _retriever(_Recorder([invalid]), _Recorder([]), _Reranker()).retrieve("alert")


def test_provider_rejects_malformed_section_path_without_leaking_a_type_error() -> None:
    malformed = replace(_chunk("bad-section"), section_path=None)  # type: ignore[arg-type]
    with pytest.raises(HybridRetrievalError, match="section_path") as error:
        _retriever(_Recorder([malformed]), _Recorder([]), _Reranker()).retrieve("alert")

    assert error.value.__cause__ is None


def test_retrieval_result_contains_immutable_tuple_evidence_with_rrf_score() -> None:
    class _IdentityReranker(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            return candidates

    result = _retriever(
        _Recorder([_chunk("first"), _chunk("second")]),
        _Recorder([_chunk("second"), _chunk("first")]),
        _IdentityReranker(),
    ).retrieve("alert", candidate_limit=2, rerank_limit=2, answer_limit=2)

    assert isinstance(result.evidence, tuple)
    assert [item.chunk_id for item in result.evidence] == [
        _chunk_id("first"),
        _chunk_id("second"),
    ]
    assert result.evidence[0].score == pytest.approx(1 / 61 + 1 / 62)
    with pytest.raises(FrozenInstanceError):
        result.evidence[0].text = "modified"  # type: ignore[misc]


def test_cross_ranked_rrf_score_beats_a_single_top_rank_and_keeps_canonical_evidence() -> (
    None
):
    class _IdentityReranker(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            return candidates

    result = _retriever(
        _Recorder([_chunk("first"), _chunk("second")]),
        _Recorder([_chunk("second")]),
        _IdentityReranker(),
    ).retrieve("alert", candidate_limit=2, rerank_limit=2, answer_limit=2)

    assert [item.chunk_id for item in result.evidence] == [
        _chunk_id("second"),
        _chunk_id("first"),
    ]
    assert result.evidence[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert result.evidence[0].text == "synthetic public evidence second"


def test_reranker_can_not_change_the_answer_copy_by_mutating_its_input() -> None:
    original = _chunk("original", score=1 / 61)

    class _MutateThenReturnCanonical(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            object.__setattr__(candidates[0], "text", "poisoned evidence")
            return [original]

    result = _retriever(
        _Recorder([original]), _Recorder([]), _MutateThenReturnCanonical()
    ).retrieve("alert")

    assert result.evidence[0].text == "synthetic public evidence original"


def test_reranker_non_list_result_is_rejected() -> None:
    class _TupleReranker(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            return tuple(candidates)  # type: ignore[return-value]

    with pytest.raises(HybridRetrievalError, match="list"):
        _retriever(
            _Recorder([_chunk("lex")]), _Recorder([]), _TupleReranker()
        ).retrieve("alert")


def test_reranker_duplicate_result_is_rejected() -> None:
    class _DuplicateReranker(_Reranker):
        def rerank(
            self, query: str, candidates: list[StoredChunk]
        ) -> list[StoredChunk]:
            return [candidates[0], candidates[0]]

    with pytest.raises(HybridRetrievalError, match="duplicate"):
        _retriever(
            _Recorder([_chunk("lex")]), _Recorder([]), _DuplicateReranker()
        ).retrieve("alert")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_url", "", "source_url"),
        ("source_url", "file:///private/evidence.txt", "source_url"),
        ("source_url", "ftp://example.invalid/evidence", "source_url"),
        ("source_version", "", "source_version"),
        ("text", "", "text"),
    ],
)
def test_provider_candidate_rejects_non_public_evidence_fields(
    field: str,
    value: str,
    message: str,
) -> None:
    invalid = replace(_chunk("invalid-fields"), **{field: value})

    with pytest.raises(HybridRetrievalError, match=message):
        _retriever(_Recorder([invalid]), _Recorder([]), _Reranker()).retrieve("alert")


@pytest.mark.parametrize(
    "source_url", ["", "file:///private/evidence.txt", "ftp://example.invalid/evidence"]
)
def test_public_evidence_rejects_non_http_source_url(source_url: str) -> None:
    with pytest.raises(ValueError, match="source_url"):
        PublicEvidence(
            chunk_id=_chunk_id("direct-evidence"),
            source_url=source_url,
            source_version="v1",
            text="synthetic evidence",
            score=0.1,
        )


def test_same_source_same_id_with_conflicting_evidence_is_rejected() -> None:
    first = _chunk("same-source")
    conflicting = replace(first, text="conflicting same-source evidence")

    with pytest.raises(HybridRetrievalError, match="conflicting"):
        _retriever(
            _Recorder([first, conflicting]), _Recorder([]), _Reranker()
        ).retrieve("alert")


def test_public_rrf_reports_malformed_protocol_evidence_without_a_bare_type_error() -> (
    None
):
    malformed = replace(_chunk("malformed-rrf"), section_path=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="public evidence"):
        fuse_public_rrf([malformed])


def test_public_rrf_reports_non_materializable_protocol_evidence_explicitly() -> None:
    source = _chunk("protocol-only")

    class _ProtocolOnly:
        chunk_id = source.chunk_id
        qdrant_point_id = source.qdrant_point_id
        source_id = source.source_id
        source_url = source.source_url
        source_version = source.source_version
        parent_title = source.parent_title
        section_path = source.section_path
        text = source.text
        data_class = source.data_class
        access_policy = source.access_policy
        score = source.score

    with pytest.raises(ValueError, match="materialized"):
        fuse_public_rrf([_ProtocolOnly()])  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", ["lexical", "dense", "reranker"])
def test_missing_provider_is_explicitly_unavailable(missing: str) -> None:
    lexical = None if missing == "lexical" else _Recorder([_chunk("lex")])
    dense = None if missing == "dense" else _Recorder([_chunk("dense")])
    reranker = None if missing == "reranker" else _Reranker()

    with pytest.raises(RetrievalUnavailable):
        _retriever(lexical, dense, reranker).retrieve("alert")


def test_provider_failure_is_explicitly_unavailable() -> None:
    class _Broken(_Recorder):
        def search(self, query: str, limit: int) -> list[StoredChunk]:
            raise RuntimeError("synthetic outage")

    with pytest.raises(RetrievalUnavailable):
        _retriever(_Broken([]), _Recorder([_chunk("dense")]), _Reranker()).retrieve(
            "alert"
        )
