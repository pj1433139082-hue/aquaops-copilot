from dataclasses import FrozenInstanceError
from hashlib import sha256
from math import inf, isfinite, nan
from pathlib import Path

import pytest

import aquaops.rag.evaluation as evaluation
from aquaops.rag.evaluation import (
    EvaluationCase,
    EvaluationDataValidationError,
    EvaluationInputValidationError,
    PUBLIC_EVALUATION_DATASET_ID,
    PUBLIC_EVALUATION_DATASET_PATH,
    PUBLIC_EVALUATION_SCHEMA_VERSION,
    PublicEvaluationSuite,
    evaluate_retriever,
    load_public_evaluation_cases,
)
from aquaops.rag.hybrid import PublicEvidence, RetrievalResult


_SYNTHETIC_REFUSAL_QUERY = "请提供虚构水厂的未公开原始监测记录。"


def _suite() -> PublicEvaluationSuite:
    return load_public_evaluation_cases()


def _evidence(chunk_id: str) -> PublicEvidence:
    return PublicEvidence(
        chunk_id=chunk_id,
        source_url="https://example.invalid/public-evidence",
        source_version="synthetic-v1",
        text="synthetic public evidence",
        score=0.1,
    )


def _result(*evidence: PublicEvidence) -> RetrievalResult:
    return RetrievalResult(
        evidence=tuple(evidence),
        fused_candidate_ids=tuple(item.chunk_id for item in evidence),
    )


class _Retriever:
    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def retrieve(self, query: str) -> object:
        self.calls.append(query)
        response = self._responses[query]
        if isinstance(response, BaseException):
            raise response
        return response


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _clock_for_suite(suite: PublicEvaluationSuite) -> _Clock:
    values: list[float] = []
    for index, _ in enumerate(suite.cases):
        start = float(index)
        values.extend((start, start + 0.01))
    return _Clock(values)


def _valid_responses(suite: PublicEvaluationSuite) -> dict[str, object]:
    return {
        case.query: (
            _result()
            if case.requires_refusal
            else _result(_evidence(case.expected_chunk_ids[0]))
        )
        for case in suite.cases
    }


def _replace_registry(
    monkeypatch: pytest.MonkeyPatch,
    dataset_path: Path,
    dataset_sha256: str,
) -> None:
    project_root = dataset_path.parents[3]
    monkeypatch.setattr(evaluation, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(evaluation, "_PUBLIC_EVALUATIONS_DIR", dataset_path.parent)
    monkeypatch.setattr(evaluation, "PUBLIC_EVALUATION_DATASET_PATH", dataset_path)
    monkeypatch.setattr(evaluation, "PUBLIC_EVALUATION_DATASET_SHA256", dataset_sha256)


def test_loads_the_registered_versioned_synthetic_public_evaluation_suite() -> None:
    suite = _suite()

    assert suite.dataset_id == PUBLIC_EVALUATION_DATASET_ID
    assert suite.schema_version == PUBLIC_EVALUATION_SCHEMA_VERSION
    assert len(suite.cases) >= 9
    assert all(case.data_class == "public" for case in suite.cases)
    assert all(case.access_policy == "public_read" for case in suite.cases)
    refusal_cases = [case for case in suite.cases if case.requires_refusal]
    ordinary_cases = [case for case in suite.cases if not case.requires_refusal]
    assert len(refusal_cases) >= 1
    assert all(case.expected_chunk_ids == () for case in refusal_cases)
    assert any(case.query == _SYNTHETIC_REFUSAL_QUERY for case in refusal_cases)
    assert all(case.expected_chunk_ids for case in ordinary_cases)


def test_loader_rejects_an_external_path_without_leaking_a_cause(
    tmp_path: Path,
) -> None:
    external_path = tmp_path / "external-public-evaluation.json"
    external_path.write_text("{}", encoding="utf-8")

    with pytest.raises(EvaluationDataValidationError) as error:
        load_public_evaluation_cases(external_path)

    assert error.value.__cause__ is None


def test_loader_rejects_a_hash_mismatched_canonical_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "data" / "rag" / "evals" / "public-water-rag-v1.json"
    dataset_path.parent.mkdir(parents=True)
    original = PUBLIC_EVALUATION_DATASET_PATH.read_bytes()
    dataset_path.write_bytes(original + b"\n")
    _replace_registry(monkeypatch, dataset_path, sha256(original).hexdigest())

    with pytest.raises(EvaluationDataValidationError, match="manifest") as error:
        load_public_evaluation_cases()

    assert error.value.__cause__ is None


def test_loader_rejects_a_symlinked_canonical_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allowed_directory = tmp_path / "data" / "rag" / "evals"
    allowed_directory.mkdir(parents=True)
    outside_target = tmp_path / "outside-public-water-rag-v1.json"
    source = PUBLIC_EVALUATION_DATASET_PATH.read_bytes()
    outside_target.write_bytes(source)
    dataset_path = allowed_directory / "public-water-rag-v1.json"
    try:
        dataset_path.symlink_to(outside_target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {type(error).__name__}")
    _replace_registry(monkeypatch, dataset_path, sha256(source).hexdigest())

    with pytest.raises(EvaluationDataValidationError, match="symlink") as error:
        load_public_evaluation_cases()

    assert error.value.__cause__ is None


def test_evaluator_rejects_handmade_cases_and_an_unregistered_suite() -> None:
    case = EvaluationCase(
        case_id="handmade-public-case",
        query="合成问题",
        topic="合成主题",
        expected_chunk_ids=("a" * 64,),
        requires_refusal=False,
        data_class="public",
        access_policy="public_read",
    )
    retriever = _Retriever({case.query: _result(_evidence("a" * 64))})

    with pytest.raises(EvaluationInputValidationError) as cases_error:
        evaluate_retriever(retriever, (case,))  # type: ignore[arg-type]
    with pytest.raises(EvaluationInputValidationError) as suite_error:
        PublicEvaluationSuite(
            dataset_id=PUBLIC_EVALUATION_DATASET_ID,
            schema_version=PUBLIC_EVALUATION_SCHEMA_VERSION,
            cases=(case,),
        )

    assert cases_error.value.__cause__ is None
    assert suite_error.value.__cause__ is None
    assert retriever.calls == []


def test_evaluate_registered_suite_reports_metrics_and_version_without_queries() -> (
    None
):
    suite = _suite()
    report = evaluate_retriever(
        _Retriever(_valid_responses(suite)),
        suite,
        latency_clock=_clock_for_suite(suite),
    )

    assert report.dataset_id == PUBLIC_EVALUATION_DATASET_ID
    assert report.schema_version == PUBLIC_EVALUATION_SCHEMA_VERSION
    assert report.answered_count == 8
    assert report.refusal_count == 0
    assert report.recall_at_k == 1.0
    assert report.mrr_at_k == 1.0
    assert report.refusal_accuracy == 1.0
    assert report.p95_latency_ms == pytest.approx(10.0)
    assert not hasattr(report.outcomes[0], "query")
    with pytest.raises(FrozenInstanceError):
        report.case_count = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", "not-a-chunk-id"),
        ("source_url", "file:///private/evidence"),
        ("source_version", ""),
        ("text", ""),
        ("score", 1),
        ("score", nan),
        ("data_class", "private"),
        ("access_policy", "private_read"),
    ],
)
def test_evaluation_rejects_every_malformed_public_evidence_field(
    field: str,
    value: object,
) -> None:
    suite = _suite()
    ordinary_case = next(case for case in suite.cases if not case.requires_refusal)
    evidence = _evidence(ordinary_case.expected_chunk_ids[0])
    object.__setattr__(evidence, field, value)
    responses = _valid_responses(suite)
    responses[ordinary_case.query] = RetrievalResult(
        evidence=(evidence,),
        fused_candidate_ids=(ordinary_case.expected_chunk_ids[0],),
    )

    report = evaluate_retriever(
        _Retriever(responses), suite, latency_clock=_clock_for_suite(suite)
    )

    assert report.answered_count == 7
    assert report.refusal_count == 1
    assert report.outcomes[0].failure_reason in {
        "invalid_public_evidence",
        "non_public_evidence",
    }


def test_evaluation_rejects_foreign_evidence_without_reading_its_fields() -> None:
    class _ForeignEvidence:
        chunk_id = "a" * 64
        data_class = "public"
        access_policy = "public_read"

    suite = _suite()
    ordinary_case = next(case for case in suite.cases if not case.requires_refusal)
    responses = _valid_responses(suite)
    responses[ordinary_case.query] = RetrievalResult(
        evidence=(_ForeignEvidence(),),  # type: ignore[arg-type]
        fused_candidate_ids=(),
    )

    report = evaluate_retriever(
        _Retriever(responses), suite, latency_clock=_clock_for_suite(suite)
    )

    assert report.refusal_count == 1
    assert report.outcomes[0].failure_reason == "invalid_public_evidence"


@pytest.mark.parametrize(
    "fused_candidate_ids",
    [
        ["a" * 64],
        ("not-a-chunk-id",),
        ("a" * 64, "a" * 64),
        ("b" * 64,),
    ],
)
def test_evaluation_rejects_malformed_or_inconsistent_fused_candidate_ids(
    fused_candidate_ids: object,
) -> None:
    suite = _suite()
    ordinary_case = next(case for case in suite.cases if not case.requires_refusal)
    evidence = _evidence(ordinary_case.expected_chunk_ids[0])
    responses = _valid_responses(suite)
    responses[ordinary_case.query] = RetrievalResult(
        evidence=(evidence,),
        fused_candidate_ids=fused_candidate_ids,  # type: ignore[arg-type]
    )

    report = evaluate_retriever(
        _Retriever(responses), suite, latency_clock=_clock_for_suite(suite)
    )

    assert report.answered_count == 7
    assert report.refusal_count == 1
    assert report.outcomes[0].failure_reason == "invalid_fused_candidate_ids"


def test_refusal_case_turns_valid_public_evidence_into_unsafe_answer() -> None:
    suite = _suite()
    refusal_case = next(case for case in suite.cases if case.requires_refusal)
    responses = _valid_responses(suite)
    responses[refusal_case.query] = _result(_evidence("f" * 64))

    report = evaluate_retriever(
        _Retriever(responses), suite, latency_clock=_clock_for_suite(suite)
    )

    assert report.answered_count == 8
    assert report.refusal_count == 0
    assert report.refusal_accuracy == 0.0
    assert report.outcomes[-1].failure_reason == "unsafe_answer"


def test_invalid_clock_duration_is_a_controlled_failure_and_never_reaches_p95() -> None:
    suite = _suite()
    values = [0.0, inf]
    for index in range(1, len(suite.cases)):
        values.extend((float(index), float(index) + 0.01))

    report = evaluate_retriever(
        _Retriever(_valid_responses(suite)), suite, latency_clock=_Clock(values)
    )

    assert report.outcomes[0].failure_reason == "clock_error"
    assert report.p95_latency_ms is not None
    assert isfinite(report.p95_latency_ms)
    assert report.p95_latency_ms == pytest.approx(10.0)


def test_evaluation_rejects_boolean_k_before_the_retriever_is_called() -> None:
    suite = _suite()
    retriever = _Retriever(_valid_responses(suite))

    with pytest.raises(EvaluationInputValidationError, match="positive integer"):
        evaluate_retriever(retriever, suite, k=True)

    assert retriever.calls == []
