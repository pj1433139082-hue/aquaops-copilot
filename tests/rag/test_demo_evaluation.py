from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pytest

from aquaops.rag.demo_corpus import load_verified_demo_chunks
from aquaops.rag.demo_evaluation import (
    DEMO_EVALUATION_DATASET_ID,
    DEMO_EVALUATION_PATH,
    DEMO_VARIANT_NAMES,
    DemoEvaluationValidationError,
    evaluate_demo_variants,
    load_verified_demo_evaluation_suite,
)
from aquaops.rag.demo_runtime import PublicQueryRefused
from aquaops.rag.hybrid import PublicEvidence, RetrievalResult


class _Retriever:
    def __init__(self, responses: dict[str, RetrievalResult | Exception]) -> None:
        self.responses = responses

    def retrieve(self, query: str, answer_limit: int = 6) -> RetrievalResult:
        response = self.responses[query]
        if isinstance(response, Exception):
            raise response
        return response


def _responses() -> dict[str, RetrievalResult | Exception]:
    chunks = {chunk.chunk_id: chunk for chunk in load_verified_demo_chunks()}
    suite = load_verified_demo_evaluation_suite()
    responses: dict[str, RetrievalResult | Exception] = {}
    for case in suite.cases:
        if case.requires_refusal:
            responses[case.query] = PublicQueryRefused("controlled refusal")
            continue
        evidence = tuple(
            PublicEvidence(
                chunk_id=chunk_id,
                source_url=chunks[chunk_id].source_url,
                source_version=chunks[chunk_id].source_version,
                text=chunks[chunk_id].text,
                score=1.0,
            )
            for chunk_id in case.expected_chunk_ids[:1]
        )
        responses[case.query] = RetrievalResult(
            evidence=evidence,
            fused_candidate_ids=tuple(item.chunk_id for item in evidence),
        )
    return responses


def test_real_public_evaluation_suite_has_fifty_questions_and_refusal_cases() -> None:
    suite = load_verified_demo_evaluation_suite()
    ordinary = [case for case in suite.cases if not case.requires_refusal]
    refusals = [case for case in suite.cases if case.requires_refusal]
    chunk_ids = {chunk.chunk_id for chunk in load_verified_demo_chunks()}

    assert suite.dataset_id == DEMO_EVALUATION_DATASET_ID
    assert len(ordinary) >= 50
    assert len(refusals) >= 5
    assert all(set(case.expected_chunk_ids) <= chunk_ids for case in ordinary)
    assert all(not case.expected_chunk_ids for case in refusals)


def test_real_public_evaluation_loader_rejects_noncanonical_copy(
    tmp_path: Path,
) -> None:
    copied = tmp_path / DEMO_EVALUATION_PATH.name
    copied.write_bytes(DEMO_EVALUATION_PATH.read_bytes())

    with pytest.raises(DemoEvaluationValidationError, match="canonical"):
        load_verified_demo_evaluation_suite(copied)


def test_evaluate_four_variants_emits_only_safe_aggregate_metrics() -> None:
    suite = load_verified_demo_evaluation_suite()
    responses = _responses()
    variants = {name: _Retriever(responses) for name in DEMO_VARIANT_NAMES}

    report = evaluate_demo_variants(
        variants,
        suite,
        k=5,
        latency_clock=perf_counter,
    )

    assert tuple(item.variant for item in report.variants) == DEMO_VARIANT_NAMES
    assert all(item.recall_at_k == 1.0 for item in report.variants)
    assert all(item.mrr_at_k == 1.0 for item in report.variants)
    assert all(item.refusal_accuracy == 1.0 for item in report.variants)
    assert all(item.technical_refusal_rate == 0.0 for item in report.variants)
    serialized = repr(asdict(report)).casefold()
    assert "query" not in serialized
    assert "text" not in serialized
    assert "source_url" not in serialized
    assert "path" not in serialized
    assert "controlled refusal" not in serialized


def test_evaluation_requires_exact_four_variant_names() -> None:
    suite = load_verified_demo_evaluation_suite()

    with pytest.raises(DemoEvaluationValidationError, match="variants"):
        evaluate_demo_variants({"hybrid_rerank": _Retriever({})}, suite)
