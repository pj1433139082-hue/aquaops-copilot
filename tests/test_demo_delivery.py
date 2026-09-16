from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import aquaops.demo_delivery as delivery_module
from aquaops.demo import PublicDemoSmokeReport
from aquaops.rag.demo_evaluation import DemoEvaluationReport, DemoVariantMetrics
from aquaops.rag.demo_runtime import PublicDemoRuntime
from aquaops.rag.local_models import PublicModelSpec
from aquaops.rag.local_models import PublicRerankerBackendStatus
from aquaops.demo_delivery import (
    build_public_demo_delivery_document,
    write_public_demo_delivery_document,
)


def _evaluation() -> DemoEvaluationReport:
    metric = DemoVariantMetrics(
        variant="bm25",
        ordinary_case_count=50,
        refusal_case_count=6,
        recall_at_k=0.8,
        mrr_at_k=0.7,
        refusal_accuracy=1.0,
        technical_refusal_rate=0.0,
        p95_latency_ms=10.0,
    )
    return DemoEvaluationReport(
        dataset_id="public-water-rag-real-v1",
        schema_version="public-water-rag-real-v1",
        dataset_sha256="a" * 64,
        corpus_sha256="b" * 64,
        case_count=56,
        k=5,
        variants=(metric, metric, metric, metric),
    )


def _smoke() -> PublicDemoSmokeReport:
    return PublicDemoSmokeReport(
        ok=True,
        chunk_count=24,
        agent_evidence_count=6,
        api_evidence_count=6,
        mcp_evidence_count=6,
        citation_sets_match=True,
        abc_dimension_count=3,
        human_review_required=True,
    )


def test_delivery_document_contains_only_reproducibility_and_aggregate_results() -> (
    None
):
    runtime = PublicDemoRuntime(
        retriever=object(),  # type: ignore[arg-type]
        variants={},
        chunk_count=24,
        corpus_sha256="b" * 64,
        model_spec=PublicModelSpec.default(local_files_only=True),
        reranker_backend_status=PublicRerankerBackendStatus(
            requested_backend="openvino",
            active_backend="openvino",
            fallback_used=False,
            status_code="openvino_ready",
        ),
    )
    document = build_public_demo_delivery_document(
        runtime,
        _evaluation(),
        _smoke(),
        generated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )

    serialized = json.dumps(asdict(document), ensure_ascii=False).casefold()
    assert document.schema_version == "aquaops-public-demo-report-v3"
    assert document.generated_at_utc == "2026-08-12T00:00:00Z"
    assert document.chunk_count == 24
    assert document.candidate_limit == 24
    assert document.rerank_limit == 8
    assert document.reranker_backend_requested == "openvino"
    assert document.reranker_backend_active == "openvino"
    assert document.reranker_backend_fallback_used is False
    assert document.reranker_backend_status_code == "openvino_ready"
    assert len(document.uv_lock_sha256) == 64
    assert "source_url" not in serialized
    assert "chunk_id" not in serialized
    assert "query" not in serialized
    assert "path" not in serialized


def test_delivery_writer_uses_fixed_atomic_destination_and_returns_only_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "public-demo-report.json"
    monkeypatch.setattr(delivery_module, "PUBLIC_DEMO_REPORT_PATH", destination)
    runtime = PublicDemoRuntime(
        retriever=object(),  # type: ignore[arg-type]
        variants={},
        chunk_count=24,
        corpus_sha256="b" * 64,
        model_spec=PublicModelSpec.default(local_files_only=True),
        reranker_backend_status=PublicRerankerBackendStatus(
            requested_backend="openvino",
            active_backend="torch",
            fallback_used=True,
            status_code="openvino_unavailable_fallback_torch",
        ),
    )
    document = build_public_demo_delivery_document(
        runtime,
        _evaluation(),
        _smoke(),
        generated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )

    report_sha256 = write_public_demo_delivery_document(document)

    assert destination.is_file()
    assert len(report_sha256) == 64
    assert list(tmp_path.glob("*.tmp")) == []
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["schema_version"] == "aquaops-public-demo-report-v3"
    assert written["reranker_backend_active"] == "torch"
    assert written["reranker_backend_fallback_used"] is True
