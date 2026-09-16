"""Hash-locked, synthetic public-only retrieval regression evaluation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
import re
from time import perf_counter
from typing import Protocol

from pydantic import HttpUrl, TypeAdapter, ValidationError

from aquaops.rag.hybrid import PublicEvidence, RetrievalResult


_CHUNK_ID = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_DATASET_KEYS = {
    "schema_version",
    "dataset_id",
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
_HTTP_URL = TypeAdapter(HttpUrl)
_SUITE_TOKEN = object()
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PUBLIC_EVALUATIONS_DIR = _PROJECT_ROOT / "data" / "rag" / "evals"
_PUBLIC_EVALUATION_FILENAME = "public-water-rag-v1.json"

PUBLIC_EVALUATION_DATASET_ID = "public-water-rag-v1"
PUBLIC_EVALUATION_SCHEMA_VERSION = "public-water-rag-v1"
PUBLIC_EVALUATION_DATASET_PATH = _PUBLIC_EVALUATIONS_DIR / _PUBLIC_EVALUATION_FILENAME
PUBLIC_EVALUATION_DATASET_SHA256 = (
    "9818bfdbf4f7b441bef890d132f82e070654bdbb2fbb03d79e1b56907e7d8884"
)


class EvaluationDataValidationError(ValueError):
    """A dataset is not the registered synthetic public evaluation suite."""


class EvaluationInputValidationError(ValueError):
    """The evaluator was not given a verified deterministic request."""


class PublicRetriever(Protocol):
    def retrieve(self, query: str) -> object: ...


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    query: str
    topic: str
    expected_chunk_ids: tuple[str, ...]
    requires_refusal: bool
    data_class: str
    access_policy: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _CASE_ID.fullmatch(self.case_id):
            raise EvaluationDataValidationError("case_id is invalid")
        if not isinstance(self.query, str) or not self.query.strip():
            raise EvaluationDataValidationError("query is invalid")
        if not isinstance(self.topic, str) or not self.topic.strip():
            raise EvaluationDataValidationError("topic is invalid")
        if self.data_class != "public" or self.access_policy != "public_read":
            raise EvaluationDataValidationError(
                "evaluation cases must be public/public_read"
            )
        if type(self.requires_refusal) is not bool:
            raise EvaluationDataValidationError("requires_refusal must be a boolean")
        if (
            not isinstance(self.expected_chunk_ids, tuple)
            or any(
                not isinstance(chunk_id, str) or not _CHUNK_ID.fullmatch(chunk_id)
                for chunk_id in self.expected_chunk_ids
            )
            or len(set(self.expected_chunk_ids)) != len(self.expected_chunk_ids)
        ):
            raise EvaluationDataValidationError("expected_chunk_ids are invalid")
        if self.requires_refusal and self.expected_chunk_ids:
            raise EvaluationDataValidationError(
                "refusal cases must not include expected_chunk_ids"
            )
        if not self.requires_refusal and not self.expected_chunk_ids:
            raise EvaluationDataValidationError(
                "ordinary cases must include expected_chunk_ids"
            )


@dataclass(frozen=True)
class PublicEvaluationSuite:
    """A loader-issued capability for the one registered public eval suite."""

    dataset_id: str
    schema_version: str
    cases: tuple[EvaluationCase, ...]
    _verification_token: object | None = field(default=None, repr=False, compare=False)
    _case_fingerprint: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._verification_token is not _SUITE_TOKEN:
            raise EvaluationInputValidationError(
                "PublicEvaluationSuite must be issued by the registered loader"
            )


@dataclass(frozen=True)
class EvaluationOutcome:
    """Safe per-case telemetry: no query, body, source, or raw exception text."""

    case_id: str
    hit_chunk_ids: tuple[str, ...]
    latency_ms: float | None
    failure_reason: str | None


@dataclass(frozen=True)
class EvaluationReport:
    """Immutable aggregate for the hash-locked public regression suite."""

    dataset_id: str
    schema_version: str
    case_count: int
    answered_count: int
    refusal_count: int
    recall_at_k: float | None
    mrr_at_k: float | None
    refusal_accuracy: float | None
    p95_latency_ms: float | None
    outcomes: tuple[EvaluationOutcome, ...]


def load_public_evaluation_cases(
    path: str | Path | None = None,
) -> PublicEvaluationSuite:
    """Issue the only registered, hash-locked synthetic public evaluation suite.

    ``path`` exists only for an explicit canonical-path assertion; any other
    path is rejected before file access. Callers should normally omit it.
    """
    _assert_requested_path_is_canonical(path)
    raw = _read_registered_dataset_bytes()
    try:
        decoded = raw.decode("utf-8")
        data = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EvaluationDataValidationError(
            "registered evaluation dataset could not be loaded"
        ) from None
    cases = _parse_registered_dataset(data)
    return PublicEvaluationSuite(
        dataset_id=PUBLIC_EVALUATION_DATASET_ID,
        schema_version=PUBLIC_EVALUATION_SCHEMA_VERSION,
        cases=cases,
        _verification_token=_SUITE_TOKEN,
        _case_fingerprint=_fingerprint_cases(cases),
    )


def evaluate_retriever(
    retriever: PublicRetriever,
    suite: PublicEvaluationSuite,
    *,
    k: int = 5,
    latency_clock: Callable[[], float] = perf_counter,
) -> EvaluationReport:
    """Evaluate only a verified suite without retaining query or evidence text."""
    _validate_evaluation_request(suite, k, latency_clock)

    outcomes: list[EvaluationOutcome] = []
    latency_values_ms: list[float] = []
    answered_count = 0
    technical_refusal_count = 0
    reciprocal_rank_total = 0.0
    hit_case_count = 0
    expected_refusal_case_count = 0
    correct_refusal_case_count = 0

    for case in suite.cases:
        if case.requires_refusal:
            expected_refusal_case_count += 1
        start = _read_clock(latency_clock)
        if start is None:
            outcome, technical_refusal, correct_refusal = _controlled_non_answer(
                case, None, "clock_error"
            )
            outcomes.append(outcome)
            technical_refusal_count += technical_refusal
            correct_refusal_case_count += correct_refusal
            continue

        try:
            response = retriever.retrieve(case.query)
        except Exception:
            response = None
            response_failure = "retriever_error"
        else:
            response_failure = None

        end = _read_clock(latency_clock)
        if end is None or end < start:
            outcome, technical_refusal, correct_refusal = _controlled_non_answer(
                case, None, "clock_error"
            )
            outcomes.append(outcome)
            technical_refusal_count += technical_refusal
            correct_refusal_case_count += correct_refusal
            continue

        latency_ms = (end - start) * 1000.0
        if not isfinite(latency_ms) or latency_ms < 0.0:
            outcome, technical_refusal, correct_refusal = _controlled_non_answer(
                case, None, "clock_error"
            )
            outcomes.append(outcome)
            technical_refusal_count += technical_refusal
            correct_refusal_case_count += correct_refusal
            continue
        latency_values_ms.append(latency_ms)
        if response_failure is not None:
            outcome, technical_refusal, correct_refusal = _controlled_non_answer(
                case, latency_ms, response_failure
            )
            outcomes.append(outcome)
            technical_refusal_count += technical_refusal
            correct_refusal_case_count += correct_refusal
            continue

        hit_chunk_ids, reciprocal_rank, failure_reason = _evaluate_response(
            response, case.expected_chunk_ids, k
        )
        if failure_reason is not None:
            outcome, technical_refusal, correct_refusal = _controlled_non_answer(
                case, latency_ms, failure_reason
            )
            outcomes.append(outcome)
            technical_refusal_count += technical_refusal
            correct_refusal_case_count += correct_refusal
            continue

        if case.requires_refusal:
            outcomes.append(
                EvaluationOutcome(case.case_id, (), latency_ms, "unsafe_answer")
            )
            continue

        answered_count += 1
        if hit_chunk_ids:
            hit_case_count += 1
            reciprocal_rank_total += reciprocal_rank
        outcomes.append(
            EvaluationOutcome(case.case_id, hit_chunk_ids, latency_ms, None)
        )

    case_count = len(suite.cases)
    ordinary_case_count = case_count - expected_refusal_case_count
    return EvaluationReport(
        dataset_id=suite.dataset_id,
        schema_version=suite.schema_version,
        case_count=case_count,
        answered_count=answered_count,
        refusal_count=technical_refusal_count,
        recall_at_k=(
            hit_case_count / ordinary_case_count if ordinary_case_count else None
        ),
        mrr_at_k=(
            reciprocal_rank_total / ordinary_case_count if ordinary_case_count else None
        ),
        refusal_accuracy=(
            correct_refusal_case_count / expected_refusal_case_count
            if expected_refusal_case_count
            else None
        ),
        p95_latency_ms=_nearest_rank_p95(latency_values_ms),
        outcomes=tuple(outcomes),
    )


def _assert_requested_path_is_canonical(path: str | Path | None) -> None:
    if path is None:
        return
    try:
        requested = Path(path)
    except (TypeError, ValueError):
        raise EvaluationDataValidationError("evaluation path is invalid") from None
    if requested != PUBLIC_EVALUATION_DATASET_PATH:
        raise EvaluationDataValidationError(
            "only the registered evaluation dataset path is allowed"
        ) from None


def _read_registered_dataset_bytes() -> bytes:
    try:
        project_root = _PROJECT_ROOT.resolve(strict=True)
        expected_directory = project_root / "data" / "rag" / "evals"
        if _PUBLIC_EVALUATIONS_DIR != expected_directory:
            raise EvaluationDataValidationError(
                "registered evaluation directory is invalid"
            )
        expected_path = expected_directory / _PUBLIC_EVALUATION_FILENAME
        if PUBLIC_EVALUATION_DATASET_PATH != expected_path:
            raise EvaluationDataValidationError("registered evaluation path is invalid")
        for segment in (
            project_root / "data",
            project_root / "data" / "rag",
            expected_directory,
            expected_path,
        ):
            if segment.is_symlink():
                raise EvaluationDataValidationError(
                    "registered evaluation dataset must not use symlinks"
                )
        resolved_directory = expected_directory.resolve(strict=True)
        resolved_path = expected_path.resolve(strict=True)
        if (
            not resolved_directory.is_relative_to(project_root)
            or resolved_path.parent != resolved_directory
        ):
            raise EvaluationDataValidationError(
                "registered evaluation dataset escapes the allowed directory"
            )
        raw = expected_path.read_bytes()
    except EvaluationDataValidationError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise EvaluationDataValidationError(
            "registered evaluation dataset could not be loaded"
        ) from None
    if sha256(raw).hexdigest() != PUBLIC_EVALUATION_DATASET_SHA256:
        raise EvaluationDataValidationError(
            "registered evaluation dataset manifest does not match"
        ) from None
    return raw


def _parse_registered_dataset(data: object) -> tuple[EvaluationCase, ...]:
    if type(data) is not dict or set(data) != _DATASET_KEYS:
        raise EvaluationDataValidationError("evaluation dataset schema is invalid")
    if (
        data["schema_version"] != PUBLIC_EVALUATION_SCHEMA_VERSION
        or data["dataset_id"] != PUBLIC_EVALUATION_DATASET_ID
        or data["data_class"] != "public"
        or data["access_policy"] != "public_read"
        or type(data["cases"]) is not list
        or not data["cases"]
    ):
        raise EvaluationDataValidationError(
            "evaluation dataset must be public/public_read"
        )

    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    for raw_case in data["cases"]:
        if type(raw_case) is not dict or set(raw_case) != _CASE_KEYS:
            raise EvaluationDataValidationError("evaluation case schema is invalid")
        expected_ids = raw_case["expected_chunk_ids"]
        if type(expected_ids) is not list:
            raise EvaluationDataValidationError("expected_chunk_ids are invalid")
        case = EvaluationCase(
            case_id=raw_case["case_id"],
            query=raw_case["query"],
            topic=raw_case["topic"],
            expected_chunk_ids=tuple(expected_ids),
            requires_refusal=raw_case["requires_refusal"],
            data_class=raw_case["data_class"],
            access_policy=raw_case["access_policy"],
        )
        if case.case_id in seen_case_ids:
            raise EvaluationDataValidationError("evaluation case ids must be unique")
        seen_case_ids.add(case.case_id)
        cases.append(case)
    return tuple(cases)


def _validate_evaluation_request(
    suite: object,
    k: object,
    latency_clock: object,
) -> None:
    if type(k) is not int or k <= 0:
        raise EvaluationInputValidationError("k must be a positive integer") from None
    if not callable(latency_clock):
        raise EvaluationInputValidationError("latency_clock must be callable") from None
    if type(suite) is not PublicEvaluationSuite:
        raise EvaluationInputValidationError(
            "evaluate_retriever requires a registered PublicEvaluationSuite"
        ) from None
    if suite._verification_token is not _SUITE_TOKEN:
        raise EvaluationInputValidationError(
            "evaluate_retriever requires a registered PublicEvaluationSuite"
        ) from None
    try:
        valid_suite = (
            suite.dataset_id == PUBLIC_EVALUATION_DATASET_ID
            and suite.schema_version == PUBLIC_EVALUATION_SCHEMA_VERSION
            and type(suite.cases) is tuple
            and bool(suite.cases)
            and suite._case_fingerprint == _fingerprint_cases(suite.cases)
        )
        if not valid_suite:
            raise EvaluationInputValidationError(
                "evaluate_retriever requires an unmodified registered suite"
            )
        for case in suite.cases:
            if type(case) is not EvaluationCase:
                raise EvaluationInputValidationError(
                    "registered suite has invalid cases"
                )
            _validate_case_for_evaluation(case)
    except EvaluationInputValidationError:
        raise
    except Exception:
        raise EvaluationInputValidationError(
            "registered suite has invalid cases"
        ) from None


def _validate_case_for_evaluation(case: EvaluationCase) -> None:
    try:
        EvaluationCase(
            case_id=case.case_id,
            query=case.query,
            topic=case.topic,
            expected_chunk_ids=case.expected_chunk_ids,
            requires_refusal=case.requires_refusal,
            data_class=case.data_class,
            access_policy=case.access_policy,
        )
    except EvaluationDataValidationError:
        raise EvaluationInputValidationError(
            "registered suite has invalid cases"
        ) from None


def _fingerprint_cases(cases: tuple[EvaluationCase, ...]) -> str:
    serializable_cases = [
        {
            "case_id": case.case_id,
            "query": case.query,
            "topic": case.topic,
            "expected_chunk_ids": list(case.expected_chunk_ids),
            "requires_refusal": case.requires_refusal,
            "data_class": case.data_class,
            "access_policy": case.access_policy,
        }
        for case in cases
    ]
    encoded = json.dumps(
        serializable_cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_clock(latency_clock: Callable[[], float]) -> float | None:
    try:
        value = latency_clock()
    except Exception:
        return None
    if type(value) not in (int, float) or not isfinite(value):
        return None
    return float(value)


def _evaluate_response(
    response: object,
    expected_chunk_ids: tuple[str, ...],
    k: int,
) -> tuple[tuple[str, ...], float, str | None]:
    if type(response) is not RetrievalResult:
        return (), 0.0, "malformed_result"
    evidence = response.evidence
    if type(evidence) is not tuple:
        return (), 0.0, "malformed_result"
    fused_candidate_ids = response.fused_candidate_ids
    if (
        type(fused_candidate_ids) is not tuple
        or any(
            type(chunk_id) is not str or not _CHUNK_ID.fullmatch(chunk_id)
            for chunk_id in fused_candidate_ids
        )
        or len(set(fused_candidate_ids)) != len(fused_candidate_ids)
    ):
        return (), 0.0, "invalid_fused_candidate_ids"
    if not evidence:
        return (), 0.0, "no_evidence"

    expected_ids = set(expected_chunk_ids)
    fused_id_set = set(fused_candidate_ids)
    seen_chunk_ids: set[str] = set()
    hit_chunk_ids: list[str] = []
    reciprocal_rank = 0.0
    for rank, item in enumerate(evidence, start=1):
        evidence_error = _validate_public_evidence(item)
        if evidence_error is not None:
            return (), 0.0, evidence_error
        chunk_id = item.chunk_id
        if chunk_id in seen_chunk_ids:
            return (), 0.0, "duplicate_evidence_id"
        seen_chunk_ids.add(chunk_id)
        if chunk_id not in fused_id_set:
            return (), 0.0, "invalid_fused_candidate_ids"
        if rank <= k and chunk_id in expected_ids:
            hit_chunk_ids.append(chunk_id)
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
    return tuple(hit_chunk_ids), reciprocal_rank, None


def _validate_public_evidence(item: object) -> str | None:
    if type(item) is not PublicEvidence:
        return "invalid_public_evidence"
    try:
        chunk_id = item.chunk_id
        source_url = item.source_url
        source_version = item.source_version
        text = item.text
        score = item.score
        data_class = item.data_class
        access_policy = item.access_policy
    except Exception:
        return "invalid_public_evidence"
    if (
        type(data_class) is not str
        or type(access_policy) is not str
        or data_class != "public"
        or access_policy != "public_read"
    ):
        return "non_public_evidence"
    if type(chunk_id) is not str or not _CHUNK_ID.fullmatch(chunk_id):
        return "invalid_public_evidence"
    if type(source_url) is not str or not source_url.strip():
        return "invalid_public_evidence"
    try:
        parsed_url = _HTTP_URL.validate_python(source_url)
    except (ValidationError, ValueError, TypeError):
        return "invalid_public_evidence"
    if parsed_url.scheme not in {"http", "https"}:
        return "invalid_public_evidence"
    if type(source_version) is not str or not source_version.strip():
        return "invalid_public_evidence"
    if type(text) is not str or not text.strip():
        return "invalid_public_evidence"
    if type(score) is not float or not isfinite(score):
        return "invalid_public_evidence"
    return None


def _controlled_non_answer(
    case: EvaluationCase,
    latency_ms: float | None,
    failure_reason: str,
) -> tuple[EvaluationOutcome, int, int]:
    if case.requires_refusal:
        return (
            EvaluationOutcome(case.case_id, (), latency_ms, "correct_refusal"),
            0,
            1,
        )
    return (
        EvaluationOutcome(case.case_id, (), latency_ms, failure_reason),
        1,
        0,
    )


def _nearest_rank_p95(latency_values_ms: list[float]) -> float | None:
    if not latency_values_ms:
        return None
    if not all(isfinite(value) and value >= 0.0 for value in latency_values_ms):
        return None
    rank = ceil(0.95 * len(latency_values_ms))
    return sorted(latency_values_ms)[rank - 1]
