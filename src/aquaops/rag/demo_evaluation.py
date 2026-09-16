"""Hash-locked real-public retrieval evaluation with aggregate-only output."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
import re
from time import perf_counter
from typing import Protocol

from aquaops.rag.demo_corpus import DEMO_CORPUS_SHA256, load_verified_demo_chunks
from aquaops.rag.demo_runtime import (
    PUBLIC_DEMO_VARIANT_NAMES,
    PublicQueryRefused,
    is_public_demo_query_allowed,
)
from aquaops.rag.hybrid import PublicEvidence, RetrievalResult


DEMO_EVALUATION_DATASET_ID = "public-water-rag-real-v1"
DEMO_EVALUATION_SCHEMA_VERSION = "public-water-rag-real-v1"
DEMO_EVALUATION_SHA256 = (
    "6f8e029394a803d6a11e6bc29587b026caf81e2160ef202868e2705f2ae936c3"
)
DEMO_VARIANT_NAMES = PUBLIC_DEMO_VARIANT_NAMES
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_EVALUATION_PATH = (
    _PROJECT_ROOT / "data" / "rag" / "evals" / "public-water-rag-real-v1.json"
)
_DATASET_KEYS = {
    "schema_version",
    "dataset_id",
    "corpus_sha256",
    "data_class",
    "access_policy",
    "cases",
}
_CASE_KEYS = {
    "case_id",
    "query",
    "topic",
    "expected_chunk_ids",
    "requires_refusal",
    "data_class",
    "access_policy",
}
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
_CHUNK_ID = re.compile(r"^[0-9a-f]{64}$")
_SUITE_TOKEN = object()


class DemoEvaluationValidationError(ValueError):
    """The request is not the registered real-public evaluation contract."""


class DemoRetriever(Protocol):
    def retrieve(self, query: str, answer_limit: int = 6) -> RetrievalResult: ...


@dataclass(frozen=True)
class DemoEvaluationCase:
    case_id: str
    query: str
    topic: str
    expected_chunk_ids: tuple[str, ...]
    requires_refusal: bool


@dataclass(frozen=True)
class VerifiedDemoEvaluationSuite:
    dataset_id: str
    schema_version: str
    corpus_sha256: str
    cases: tuple[DemoEvaluationCase, ...]
    _verification_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._verification_token is not _SUITE_TOKEN:
            raise DemoEvaluationValidationError(
                "evaluation suite must be loader-issued"
            )


@dataclass(frozen=True)
class DemoVariantMetrics:
    variant: str
    ordinary_case_count: int
    refusal_case_count: int
    recall_at_k: float
    mrr_at_k: float
    refusal_accuracy: float
    technical_refusal_rate: float
    p95_latency_ms: float | None


@dataclass(frozen=True)
class DemoEvaluationReport:
    dataset_id: str
    schema_version: str
    dataset_sha256: str
    corpus_sha256: str
    case_count: int
    k: int
    variants: tuple[DemoVariantMetrics, ...]


def load_verified_demo_evaluation_suite(
    path: str | Path | None = None,
) -> VerifiedDemoEvaluationSuite:
    _assert_canonical_path(path)
    try:
        raw = DEMO_EVALUATION_PATH.read_bytes()
    except OSError:
        raise DemoEvaluationValidationError(
            "registered evaluation could not be loaded"
        ) from None
    if sha256(raw).hexdigest() != DEMO_EVALUATION_SHA256:
        raise DemoEvaluationValidationError(
            "registered evaluation fingerprint is invalid"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise DemoEvaluationValidationError(
            "registered evaluation could not be decoded"
        ) from None
    cases = _parse_dataset(payload)
    return VerifiedDemoEvaluationSuite(
        dataset_id=DEMO_EVALUATION_DATASET_ID,
        schema_version=DEMO_EVALUATION_SCHEMA_VERSION,
        corpus_sha256=DEMO_CORPUS_SHA256,
        cases=cases,
        _verification_token=_SUITE_TOKEN,
    )


def evaluate_demo_variants(
    variants: Mapping[str, DemoRetriever],
    suite: VerifiedDemoEvaluationSuite,
    *,
    k: int = 5,
    latency_clock: Callable[[], float] = perf_counter,
) -> DemoEvaluationReport:
    if (
        type(variants) is not dict
        or tuple(variants) != DEMO_VARIANT_NAMES
        or type(suite) is not VerifiedDemoEvaluationSuite
        or suite._verification_token is not _SUITE_TOKEN
        or type(k) is not int
        or not 1 <= k <= 6
        or not callable(latency_clock)
    ):
        raise DemoEvaluationValidationError(
            "evaluation variants or request are invalid"
        )
    metrics = tuple(
        _evaluate_variant(name, variants[name], suite, k, latency_clock)
        for name in DEMO_VARIANT_NAMES
    )
    return DemoEvaluationReport(
        dataset_id=suite.dataset_id,
        schema_version=suite.schema_version,
        dataset_sha256=DEMO_EVALUATION_SHA256,
        corpus_sha256=suite.corpus_sha256,
        case_count=len(suite.cases),
        k=k,
        variants=metrics,
    )


def _evaluate_variant(
    name: str,
    retriever: DemoRetriever,
    suite: VerifiedDemoEvaluationSuite,
    k: int,
    latency_clock: Callable[[], float],
) -> DemoVariantMetrics:
    ordinary = 0
    refusal = 0
    hits = 0
    reciprocal_rank = 0.0
    technical_refusals = 0
    correct_refusals = 0
    latencies: list[float] = []
    for case in suite.cases:
        ordinary += int(not case.requires_refusal)
        refusal += int(case.requires_refusal)
        start = _safe_clock(latency_clock)
        try:
            result = retriever.retrieve(case.query, answer_limit=6)
        except PublicQueryRefused:
            if case.requires_refusal:
                correct_refusals += 1
            else:
                technical_refusals += 1
            result = None
        except Exception:
            if not case.requires_refusal:
                technical_refusals += 1
            result = None
        end = _safe_clock(latency_clock)
        if start is not None and end is not None and end >= start:
            elapsed = (end - start) * 1_000
            if isfinite(elapsed):
                latencies.append(elapsed)
        if case.requires_refusal or result is None:
            continue
        ids = _validated_result_ids(result, k)
        if ids is None:
            technical_refusals += 1
            continue
        expected = set(case.expected_chunk_ids)
        ranks = [
            index + 1 for index, chunk_id in enumerate(ids) if chunk_id in expected
        ]
        if ranks:
            hits += 1
            reciprocal_rank += 1.0 / min(ranks)
    return DemoVariantMetrics(
        variant=name,
        ordinary_case_count=ordinary,
        refusal_case_count=refusal,
        recall_at_k=hits / ordinary,
        mrr_at_k=reciprocal_rank / ordinary,
        refusal_accuracy=correct_refusals / refusal,
        technical_refusal_rate=technical_refusals / ordinary,
        p95_latency_ms=_p95(latencies),
    )


def _validated_result_ids(result: object, k: int) -> tuple[str, ...] | None:
    if type(result) is not RetrievalResult or type(result.evidence) is not tuple:
        return None
    ids: list[str] = []
    for evidence in result.evidence[:k]:
        if (
            type(evidence) is not PublicEvidence
            or evidence.data_class != "public"
            or evidence.access_policy != "public_read"
            or not _CHUNK_ID.fullmatch(evidence.chunk_id)
            or evidence.chunk_id in ids
        ):
            return None
        ids.append(evidence.chunk_id)
    if not ids or type(result.fused_candidate_ids) is not tuple:
        return None
    if any(not _CHUNK_ID.fullmatch(item) for item in result.fused_candidate_ids):
        return None
    if not set(ids) <= set(result.fused_candidate_ids):
        return None
    return tuple(ids)


def _parse_dataset(payload: object) -> tuple[DemoEvaluationCase, ...]:
    if type(payload) is not dict or set(payload) != _DATASET_KEYS:
        raise DemoEvaluationValidationError("evaluation schema is invalid")
    if (
        payload["schema_version"] != DEMO_EVALUATION_SCHEMA_VERSION
        or payload["dataset_id"] != DEMO_EVALUATION_DATASET_ID
        or payload["corpus_sha256"] != DEMO_CORPUS_SHA256
        or payload["data_class"] != "public"
        or payload["access_policy"] != "public_read"
        or type(payload["cases"]) is not list
        or len(payload["cases"]) > 200
    ):
        raise DemoEvaluationValidationError("evaluation header is invalid")
    known_chunks = {chunk.chunk_id for chunk in load_verified_demo_chunks()}
    cases: list[DemoEvaluationCase] = []
    case_ids: set[str] = set()
    queries: set[str] = set()
    for raw_case in payload["cases"]:
        case = _parse_case(raw_case, known_chunks)
        if case.case_id in case_ids or case.query in queries:
            raise DemoEvaluationValidationError("evaluation cases must be unique")
        case_ids.add(case.case_id)
        queries.add(case.query)
        cases.append(case)
    ordinary_count = sum(not case.requires_refusal for case in cases)
    refusal_count = sum(case.requires_refusal for case in cases)
    if ordinary_count < 50 or refusal_count < 5:
        raise DemoEvaluationValidationError("evaluation coverage is insufficient")
    return tuple(cases)


def _parse_case(payload: object, known_chunks: set[str]) -> DemoEvaluationCase:
    if type(payload) is not dict or set(payload) != _CASE_KEYS:
        raise DemoEvaluationValidationError("evaluation case schema is invalid")
    if (
        type(payload["case_id"]) is not str
        or not _CASE_ID.fullmatch(payload["case_id"])
        or type(payload["query"]) is not str
        or not payload["query"].strip()
        or len(payload["query"]) > 512
        or type(payload["topic"]) is not str
        or not payload["topic"].strip()
        or type(payload["expected_chunk_ids"]) is not list
        or type(payload["requires_refusal"]) is not bool
        or payload["data_class"] != "public"
        or payload["access_policy"] != "public_read"
    ):
        raise DemoEvaluationValidationError("evaluation case is invalid")
    expected = tuple(payload["expected_chunk_ids"])
    if (
        any(
            type(chunk_id) is not str or not _CHUNK_ID.fullmatch(chunk_id)
            for chunk_id in expected
        )
        or len(set(expected)) != len(expected)
        or not set(expected) <= known_chunks
        or (payload["requires_refusal"] and expected)
        or (not payload["requires_refusal"] and not expected)
        or (
            payload["requires_refusal"]
            == is_public_demo_query_allowed(payload["query"])
        )
    ):
        raise DemoEvaluationValidationError(
            "evaluation case evidence contract is invalid"
        )
    return DemoEvaluationCase(
        case_id=payload["case_id"],
        query=payload["query"],
        topic=payload["topic"],
        expected_chunk_ids=expected,
        requires_refusal=payload["requires_refusal"],
    )


def _assert_canonical_path(path: str | Path | None) -> None:
    if path is None:
        return
    try:
        requested = Path(path)
        resolved = requested.resolve(strict=True)
        canonical = DEMO_EVALUATION_PATH.resolve(strict=True)
    except (OSError, RuntimeError):
        raise DemoEvaluationValidationError(
            "evaluation path must be canonical"
        ) from None
    if requested.is_symlink() or resolved != canonical:
        raise DemoEvaluationValidationError("evaluation path must be canonical")


def _safe_clock(clock: Callable[[], float]) -> float | None:
    try:
        value = clock()
    except Exception:
        return None
    return (
        float(value) if type(value) in (int, float) and isfinite(float(value)) else None
    )


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(len(ordered) * 0.95) - 1)]
