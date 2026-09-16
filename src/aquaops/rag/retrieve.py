from collections.abc import Callable
from dataclasses import replace
from math import isfinite
import re
from typing import Protocol, cast

from pydantic import BaseModel, HttpUrl


_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_WORD = re.compile(r"[a-z0-9]+")


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_url: HttpUrl
    source_version: str
    text: str
    score: float


InMemoryChunkScorer = Callable[[str, RetrievedChunk], float]


class PublicFusionChunk(Protocol):
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


def _tokenize(text: str) -> list[str]:
    """Tokenize supplied text locally, including CJK characters and bigrams.

    This deterministic baseline uses only the Python standard library and does
    not load a dictionary, model, network resource, or document store.
    """
    normalized = text.casefold()
    tokens: list[str] = []
    previous_end = 0
    for match in _CJK_RUN.finditer(normalized):
        tokens.extend(_WORD.findall(normalized[previous_end : match.start()]))
        cjk_text = match.group()
        tokens.extend(cjk_text)
        tokens.extend(cjk_text[index : index + 2] for index in range(len(cjk_text) - 1))
        previous_end = match.end()
    tokens.extend(_WORD.findall(normalized[previous_end:]))
    return tokens


def retrieve_bm25(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Rank caller-supplied chunks with deterministic, local in-memory BM25."""
    if not chunks:
        return []

    from rank_bm25 import BM25Okapi

    scores = BM25Okapi([_tokenize(chunk.text) for chunk in chunks]).get_scores(
        _tokenize(query)
    )
    return [
        chunk.model_copy(update={"score": float(score)})
        for chunk, score in sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def retrieve_dense(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    scorer: InMemoryChunkScorer | None = None,
) -> list[RetrievedChunk]:
    """Rank supplied chunks only when an in-memory scorer is explicitly injected.

    This baseline does not load embedding models, construct vectors, or access a
    network or document store. Without a scorer, dense retrieval is unavailable
    and returns no evidence rather than claiming model-backed retrieval.
    """
    if scorer is None:
        return []

    scored_chunks = [(chunk, float(scorer(query, chunk))) for chunk in chunks]
    return [
        chunk.model_copy(update={"score": score})
        for chunk, score in sorted(
            scored_chunks,
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def fuse_rrf(
    *ranked_lists: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Optionally fuse evaluated rankings; this is not a default production path."""
    if k <= 0:
        raise ValueError("k must be positive")

    fused: dict[str, tuple[RetrievedChunk, float]] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            current = fused.get(chunk.chunk_id, (chunk, 0.0))
            fused[chunk.chunk_id] = (chunk, current[1] + 1 / (k + rank))
    return [
        chunk.model_copy(update={"score": score})
        for chunk, score in sorted(
            fused.values(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def fuse_public_rrf(
    *ranked_lists: list[PublicFusionChunk],
    k: int = 60,
) -> list[PublicFusionChunk]:
    """Fuse already-validated public rankings without exposing another data path."""
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")

    fused: dict[str, tuple[PublicFusionChunk, float]] = {}
    for ranked in ranked_lists:
        unique_ranked: list[PublicFusionChunk] = []
        seen_fingerprints: dict[str, tuple[object, ...]] = {}
        for raw_chunk in ranked:
            chunk = _validate_public_fusion_chunk(raw_chunk)
            fingerprint = _public_evidence_fingerprint(chunk)
            existing = seen_fingerprints.get(chunk.chunk_id)
            if existing is None:
                unique_ranked.append(chunk)
                seen_fingerprints[chunk.chunk_id] = fingerprint
            elif existing != fingerprint:
                raise ValueError("conflicting public evidence for the same chunk_id")
        for rank, chunk in enumerate(unique_ranked, start=1):
            current = fused.get(chunk.chunk_id)
            if current is not None and _public_evidence_fingerprint(
                chunk
            ) != _public_evidence_fingerprint(current[0]):
                raise ValueError("conflicting public evidence for the same chunk_id")
            score = (current[1] if current else 0.0) + 1 / (k + rank)
            fused[chunk.chunk_id] = (current[0] if current else chunk, score)
    return [
        _with_fused_score(chunk, score)
        for chunk, score in sorted(
            fused.values(), key=lambda item: item[1], reverse=True
        )
    ]


def _validate_public_fusion_chunk(chunk: object) -> PublicFusionChunk:
    try:
        fingerprint = (
            chunk.chunk_id,  # type: ignore[attr-defined]
            chunk.qdrant_point_id,  # type: ignore[attr-defined]
            chunk.source_id,  # type: ignore[attr-defined]
            chunk.source_url,  # type: ignore[attr-defined]
            chunk.source_version,  # type: ignore[attr-defined]
            chunk.parent_title,  # type: ignore[attr-defined]
            tuple(chunk.section_path),  # type: ignore[attr-defined]
            chunk.text,  # type: ignore[attr-defined]
            chunk.data_class,  # type: ignore[attr-defined]
            chunk.access_policy,  # type: ignore[attr-defined]
        )
        score = chunk.score  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise ValueError("public evidence is missing required fields") from error
    if not all(isinstance(value, str) and value.strip() for value in fingerprint[:6]):
        raise ValueError("public evidence has invalid required fields")
    section_path = fingerprint[6]
    if not isinstance(section_path, tuple) or not all(
        isinstance(part, str) and part.strip() for part in section_path
    ):
        raise ValueError("public evidence has invalid required fields")
    if not all(isinstance(value, str) and value.strip() for value in fingerprint[7:]):
        raise ValueError("public evidence has invalid required fields")
    if type(score) not in (int, float) or not isfinite(score):
        raise ValueError("public evidence has a non-finite score")
    return cast(PublicFusionChunk, chunk)


def _public_evidence_fingerprint(chunk: PublicFusionChunk) -> tuple[object, ...]:
    return (
        chunk.qdrant_point_id,
        chunk.source_id,
        chunk.source_url,
        chunk.source_version,
        chunk.parent_title,
        chunk.section_path,
        chunk.text,
        chunk.data_class,
        chunk.access_policy,
    )


def _with_fused_score(chunk: PublicFusionChunk, score: float) -> PublicFusionChunk:
    try:
        scored_chunk = replace(chunk, score=score)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "public evidence could not be materialized with an RRF score"
        ) from error
    return _validate_public_fusion_chunk(scored_chunk)
