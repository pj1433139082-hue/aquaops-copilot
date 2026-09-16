"""Strictly public hybrid retrieval with injected local providers."""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Protocol
from uuid import UUID, uuid5

from pydantic import HttpUrl, TypeAdapter, ValidationError

from aquaops.rag.retrieve import fuse_public_rrf
from aquaops.rag.store import StoredChunk


_QDRANT_NAMESPACE = UUID("ce3be424-4f91-55a3-8ed0-dd56dc95c39a")
_HTTP_URL = TypeAdapter(HttpUrl)


class DenseRetriever(Protocol):
    def search(self, query: str, limit: int) -> list[StoredChunk]: ...


class LexicalRetriever(Protocol):
    def search(self, query: str, limit: int) -> list[StoredChunk]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[StoredChunk]
    ) -> list[StoredChunk]: ...


class HybridRetrievalError(Exception):
    """A provider result violates the public hybrid retrieval contract."""


class HybridRetrievalValidationError(HybridRetrievalError, ValueError):
    """The caller supplied an invalid hybrid-retrieval request."""


class RetrievalUnavailable(HybridRetrievalError):
    """A required injected retrieval capability is missing or failed."""


CandidateFingerprint = tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    tuple[str, ...],
    str,
    str,
    str,
    float,
]


@dataclass(frozen=True)
class PublicEvidence:
    chunk_id: str
    source_url: str
    source_version: str
    text: str
    score: float
    data_class: str = "public"
    access_policy: str = "public_read"

    def __post_init__(self) -> None:
        _validate_evidence_fields(self.source_url, self.source_version, self.text)
        if self.data_class != "public" or self.access_policy != "public_read":
            raise ValueError("PublicEvidence must be public/public_read")


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[PublicEvidence, ...]
    fused_candidate_ids: tuple[str, ...]


class HybridPublicRetriever:
    """Retrieve only public-read candidates, then rerank their fused subset."""

    def __init__(
        self,
        *,
        lexical_retriever: LexicalRetriever | None,
        dense_retriever: DenseRetriever | None,
        reranker: Reranker | None,
    ) -> None:
        self._lexical_retriever = lexical_retriever
        self._dense_retriever = dense_retriever
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        candidate_limit: int = 40,
        rerank_limit: int = 20,
        answer_limit: int = 6,
    ) -> RetrievalResult:
        self._validate_request(query, candidate_limit, rerank_limit, answer_limit)
        if (
            self._lexical_retriever is None
            or self._dense_retriever is None
            or self._reranker is None
        ):
            raise RetrievalUnavailable(
                "lexical retrieval, dense retrieval, and reranking are required"
            )

        lexical = self._search(
            self._lexical_retriever, query, candidate_limit, "lexical"
        )
        dense = self._search(self._dense_retriever, query, candidate_limit, "dense")
        try:
            fused = fuse_public_rrf(lexical, dense)
        except ValueError as error:
            raise HybridRetrievalError(
                f"public candidates could not be fused: {error}"
            ) from error
        rerank_candidates = fused[:rerank_limit]
        allowed_snapshot = {
            candidate.chunk_id: self._fingerprint(candidate)
            for candidate in rerank_candidates
        }
        answer_candidates = {
            candidate.chunk_id: self._copy_candidate(candidate)
            for candidate in rerank_candidates
        }
        rerank_input = [
            self._copy_candidate(candidate) for candidate in rerank_candidates
        ]
        reranked = self._rerank(query, rerank_input, self._reranker, rerank_limit)
        self._validate_reranked(reranked, allowed_snapshot)
        return RetrievalResult(
            evidence=tuple(
                self._evidence(answer_candidates[chunk.chunk_id])
                for chunk in reranked[:answer_limit]
            ),
            fused_candidate_ids=tuple(chunk.chunk_id for chunk in fused),
        )

    @staticmethod
    def _validate_request(
        query: object,
        candidate_limit: object,
        rerank_limit: object,
        answer_limit: object,
    ) -> None:
        if not isinstance(query, str) or not query.strip():
            raise HybridRetrievalValidationError("query must be a non-empty string")
        limits = (candidate_limit, rerank_limit, answer_limit)
        if any(type(limit) is not int for limit in limits) or not (
            1 <= answer_limit <= rerank_limit <= candidate_limit <= 40
        ):
            raise HybridRetrievalValidationError(
                "limits must satisfy 1 <= answer <= rerank <= candidate <= 40"
            )

    @staticmethod
    def _search(
        provider: LexicalRetriever | DenseRetriever, query: str, limit: int, name: str
    ) -> list[StoredChunk]:
        try:
            candidates = provider.search(query, limit)
        except Exception as error:
            raise RetrievalUnavailable(f"{name} retrieval is unavailable") from error
        if type(candidates) is not list:
            raise HybridRetrievalError(f"{name} retrieval must return an exact list")
        if len(candidates) > limit:
            raise HybridRetrievalError(f"{name} retrieval exceeded the candidate limit")
        HybridPublicRetriever._validate_candidates(candidates)
        return HybridPublicRetriever._deduplicate_by_chunk_id(candidates)[:limit]

    @staticmethod
    def _rerank(
        query: str,
        candidates: list[StoredChunk],
        reranker: Reranker | None = None,
        rerank_limit: int = 0,
    ) -> list[StoredChunk]:
        if reranker is None:
            raise RetrievalUnavailable("reranking is unavailable")
        try:
            result = reranker.rerank(query, candidates)
        except Exception as error:
            raise RetrievalUnavailable("reranking is unavailable") from error
        if type(result) is not list:
            raise HybridRetrievalError("reranker must return an exact list")
        if len(result) > rerank_limit:
            raise HybridRetrievalError("reranker exceeded the rerank limit")
        return result

    @staticmethod
    def _validate_candidates(candidates: list[StoredChunk]) -> None:
        for candidate in candidates:
            HybridPublicRetriever._validate_public_candidate(candidate)

    @staticmethod
    def _deduplicate_by_chunk_id(candidates: list[StoredChunk]) -> list[StoredChunk]:
        seen: dict[str, CandidateFingerprint] = {}
        unique: list[StoredChunk] = []
        for candidate in candidates:
            fingerprint = HybridPublicRetriever._fingerprint(candidate)
            existing = seen.get(candidate.chunk_id)
            if existing is None:
                unique.append(candidate)
                seen[candidate.chunk_id] = fingerprint
            elif existing[:-1] != fingerprint[:-1]:
                raise HybridRetrievalError(
                    "provider returned conflicting evidence for the same chunk_id"
                )
        return unique

    @staticmethod
    def _validate_reranked(
        reranked: list[StoredChunk],
        allowed_snapshot: Mapping[str, CandidateFingerprint],
    ) -> None:
        seen: set[str] = set()
        for candidate in reranked:
            HybridPublicRetriever._validate_public_candidate(candidate)
            allowed_fingerprint = allowed_snapshot.get(candidate.chunk_id)
            if allowed_fingerprint is None:
                raise HybridRetrievalError(
                    "reranker result must be a subset of fused candidates"
                )
            if HybridPublicRetriever._fingerprint(candidate) != allowed_fingerprint:
                raise HybridRetrievalError(
                    "reranker result does not match the immutable candidate snapshot"
                )
            if candidate.chunk_id in seen:
                raise HybridRetrievalError(
                    "reranker result contains duplicate chunk_id"
                )
            seen.add(candidate.chunk_id)

    @staticmethod
    def _validate_public_candidate(candidate: object) -> None:
        if not isinstance(candidate, StoredChunk):
            raise HybridRetrievalError("provider returned an invalid candidate")
        if not isinstance(candidate.chunk_id, str) or not candidate.chunk_id:
            raise HybridRetrievalError("provider returned an invalid chunk_id")
        if candidate.data_class != "public" or candidate.access_policy != "public_read":
            raise HybridRetrievalError("provider returned a non-public candidate")
        if type(candidate.score) not in (int, float) or not isfinite(candidate.score):
            raise HybridRetrievalError("provider returned a non-finite score")
        try:
            _validate_evidence_fields(
                candidate.source_url, candidate.source_version, candidate.text
            )
        except ValueError as error:
            raise HybridRetrievalError(str(error)) from error
        if not isinstance(candidate.section_path, tuple) or not all(
            isinstance(part, str) and part.strip() for part in candidate.section_path
        ):
            raise HybridRetrievalError(
                "provider returned an invalid section_path"
            ) from None
        if (
            not isinstance(candidate.chunk_id, str)
            or len(candidate.chunk_id) != 64
            or any(char not in "0123456789abcdef" for char in candidate.chunk_id)
        ):
            raise HybridRetrievalError(
                "provider returned an invalid public-store identity"
            )
        if not isinstance(candidate.qdrant_point_id, str):
            raise HybridRetrievalError(
                "provider returned an invalid public-store identity"
            )
        try:
            point_id = UUID(candidate.qdrant_point_id)
        except ValueError as error:
            raise HybridRetrievalError(
                "provider returned an invalid public-store identity"
            ) from error
        if point_id.version != 5 or point_id != uuid5(
            _QDRANT_NAMESPACE, candidate.chunk_id
        ):
            raise HybridRetrievalError(
                "provider returned an invalid public-store identity"
            )

    @staticmethod
    def _fingerprint(candidate: StoredChunk) -> CandidateFingerprint:
        return (
            candidate.chunk_id,
            candidate.qdrant_point_id,
            candidate.source_id,
            candidate.source_url,
            candidate.source_version,
            candidate.parent_title,
            tuple(candidate.section_path),
            candidate.text,
            candidate.data_class,
            candidate.access_policy,
            float(candidate.score),
        )

    @staticmethod
    def _copy_candidate(candidate: StoredChunk) -> StoredChunk:
        return StoredChunk(
            chunk_id=candidate.chunk_id,
            qdrant_point_id=candidate.qdrant_point_id,
            source_id=candidate.source_id,
            source_url=candidate.source_url,
            source_version=candidate.source_version,
            parent_title=candidate.parent_title,
            section_path=tuple(candidate.section_path),
            text=candidate.text,
            data_class=candidate.data_class,
            access_policy=candidate.access_policy,
            score=float(candidate.score),
        )

    @staticmethod
    def _evidence(chunk: StoredChunk) -> PublicEvidence:
        return PublicEvidence(
            chunk_id=chunk.chunk_id,
            source_url=chunk.source_url,
            source_version=chunk.source_version,
            text=chunk.text,
            score=float(chunk.score),
        )


def _validate_evidence_fields(
    source_url: object,
    source_version: object,
    text: object,
) -> None:
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("source_url must be a non-empty http or https URL")
    try:
        parsed_url = _HTTP_URL.validate_python(source_url)
    except ValidationError as error:
        raise ValueError("source_url must be a non-empty http or https URL") from error
    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError("source_url must be a non-empty http or https URL")
    if not isinstance(source_version, str) or not source_version.strip():
        raise ValueError("source_version must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
