import pytest
from pydantic import ValidationError

from aquaops.rag.citations import build_grounded_answer
from aquaops.rag.retrieve import (
    RetrievedChunk,
    fuse_rrf,
    retrieve_bm25,
    retrieve_dense,
)


def _chunk(
    chunk_id: str,
    text: str = "synthetic public evidence",
    source_url: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_url=(
            source_url
            if source_url is not None
            else f"https://example.invalid/{chunk_id}"
        ),
        source_version="v1",
        text=text,
        score=0.0,
    )


def test_answer_is_refused_when_no_retrieved_evidence_exists() -> None:
    result = build_grounded_answer(question="如何处理异常？", chunks=[])

    assert result.answer == "信息不足，无法基于已公开证据给出建议。"
    assert result.requires_human_review is True
    assert result.citations == []


def test_answer_with_evidence_includes_source_url_citations() -> None:
    chunks = [_chunk("chunk-1")]

    result = build_grounded_answer(question="如何处理异常？", chunks=chunks)

    assert result.requires_human_review is True
    assert result.citations == [
        {
            "chunk_id": "chunk-1",
            "source_url": "https://example.invalid/chunk-1",
        }
    ]


def test_fuse_rrf_deduplicates_chunk_ids() -> None:
    first = _chunk("chunk-1")
    second = _chunk("chunk-2")

    fused = fuse_rrf([first, second], [first])

    assert len(fused) == 2
    assert [chunk.chunk_id for chunk in fused].count("chunk-1") == 1


@pytest.mark.parametrize("k", [0, -1])
def test_fuse_rrf_rejects_nonpositive_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        fuse_rrf([], k=k)


def test_bm25_and_rrf_accept_empty_chunks() -> None:
    assert retrieve_bm25("abnormal handling", []) == []
    assert fuse_rrf([], []) == []


def test_bm25_ranks_unique_query_term_without_mutating_input() -> None:
    chunks = [
        _chunk("chunk-1", "routine inspection procedures"),
        _chunk("chunk-2", "aquifer contamination response checklist"),
        _chunk("chunk-3", "scheduled pump maintenance"),
    ]

    ranked = retrieve_bm25("aquifer", chunks)

    assert ranked[0].chunk_id == "chunk-2"
    assert all(isinstance(chunk, RetrievedChunk) for chunk in ranked)
    assert ranked[0] is not chunks[1]
    assert [chunk.score for chunk in chunks] == [0.0, 0.0, 0.0]


def test_bm25_ranks_chinese_query_against_continuous_chinese_text() -> None:
    chunks = [
        _chunk("chunk-1", "管网常规巡检计划"),
        _chunk("chunk-2", "发现泄漏后立即隔离阀门"),
        _chunk("chunk-3", "水泵维护保养记录"),
    ]

    ranked = retrieve_bm25("泄漏", chunks)

    assert ranked[0].chunk_id == "chunk-2"


@pytest.mark.parametrize("source_url", ["", "not a valid url"])
def test_retrieved_chunk_rejects_blank_or_invalid_source_url(
    source_url: str,
) -> None:
    with pytest.raises(ValidationError):
        _chunk("chunk-1", source_url=source_url)


def test_dense_retrieval_is_unavailable_without_an_in_memory_scorer() -> None:
    assert retrieve_dense("abnormal handling", [_chunk("chunk-1")]) == []


def test_dense_retrieval_scores_each_chunk_once_and_uses_cached_scores() -> None:
    chunks = [_chunk("chunk-1"), _chunk("chunk-2")]
    calls: list[str] = []
    expected_scores = {"chunk-1": 0.1, "chunk-2": 0.9}

    def scorer(query: str, chunk: RetrievedChunk) -> float:
        assert query == "alert"
        calls.append(chunk.chunk_id)
        return expected_scores[chunk.chunk_id]

    ranked = retrieve_dense("alert", chunks, scorer=scorer)

    assert calls == ["chunk-1", "chunk-2"]
    assert [(chunk.chunk_id, chunk.score) for chunk in ranked] == [
        ("chunk-2", 0.9),
        ("chunk-1", 0.1),
    ]
