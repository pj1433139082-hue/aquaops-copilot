import json
import os
import warnings

import aquaops.public_demo_cli as cli_module
from aquaops.demo import PublicDemoComponents, PublicDemoSmokeReport
from aquaops.rag.demo_evaluation import DemoEvaluationReport, DemoVariantMetrics
from aquaops.rag.demo_runtime import PublicDemoRuntime
from aquaops.rag.local_models import PublicModelSpec
from aquaops.rag.local_models import PublicRerankerBackendStatus


def _components(*, closer=None) -> PublicDemoComponents:
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
        _closer=closer,
    )
    return PublicDemoComponents(
        runtime=runtime,
        operations_graph=object(),
        app=object(),  # type: ignore[arg-type]
        mcp_handlers=object(),  # type: ignore[arg-type]
    )


def _evaluation() -> DemoEvaluationReport:
    variants = tuple(
        DemoVariantMetrics(
            variant=name,
            ordinary_case_count=50,
            refusal_case_count=6,
            recall_at_k=0.8,
            mrr_at_k=0.7,
            refusal_accuracy=1.0,
            technical_refusal_rate=0.0,
            p95_latency_ms=10.0,
        )
        for name in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")
    )
    return DemoEvaluationReport(
        dataset_id="public-water-rag-real-v1",
        schema_version="public-water-rag-real-v1",
        dataset_sha256="a" * 64,
        corpus_sha256="b" * 64,
        case_count=56,
        k=5,
        variants=variants,
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


def test_prepare_command_outputs_only_safe_runtime_metadata(
    monkeypatch, capsys
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        cli_module,
        "compose_public_demo",
        lambda *, local_files_only, reranker_backend: (
            calls.append((local_files_only, reranker_backend)) or _components()
        ),
    )

    assert cli_module.main(("prepare", "--offline", "--reranker-backend=openvino")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert calls == [(True, "openvino")]
    assert payload["ok"] is True
    assert payload["chunk_count"] == 24
    assert payload["candidate_limit"] == 24
    assert payload["rerank_limit"] == 8
    assert payload["corpus_sha256"] == "b" * 64
    assert payload["reranker_backend"] == {
        "active": "torch",
        "fallback_used": True,
        "requested": "openvino",
        "status_code": "openvino_unavailable_fallback_torch",
    }
    assert "path" not in json.dumps(payload).casefold()
    assert "query" not in json.dumps(payload).casefold()


def test_offline_command_sets_and_restores_model_offline_flags(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    observed: list[tuple[str | None, str | None]] = []

    def _offline_components(**_kwargs):
        observed.append(
            (
                os.environ.get("HF_HUB_OFFLINE"),
                os.environ.get("TRANSFORMERS_OFFLINE"),
            )
        )
        return _components()

    monkeypatch.setattr(cli_module, "compose_public_demo", _offline_components)

    assert cli_module.main(("prepare", "--offline")) == 0
    capsys.readouterr()

    assert observed == [("1", "1")]
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None


def test_command_closes_owned_runtime_before_returning(monkeypatch, capsys) -> None:
    closed: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "compose_public_demo",
        lambda **_kwargs: _components(closer=lambda: closed.append("closed")),
    )

    assert cli_module.main(("prepare", "--offline")) == 0

    assert closed == ["closed"]
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_evaluate_command_writes_aggregate_delivery_report(monkeypatch, capsys) -> None:
    components = _components()
    evaluation = _evaluation()
    smoke = _smoke()
    monkeypatch.setattr(cli_module, "compose_public_demo", lambda **_kwargs: components)
    monkeypatch.setattr(cli_module, "evaluate_public_demo", lambda _value: evaluation)
    monkeypatch.setattr(cli_module, "smoke_public_demo", lambda _value: smoke)
    monkeypatch.setattr(
        cli_module,
        "build_public_demo_delivery_document",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        cli_module,
        "write_public_demo_delivery_document",
        lambda _document: "c" * 64,
    )

    assert cli_module.main(("evaluate",)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["case_count"] == 56
    assert payload["candidate_limit"] == 24
    assert payload["rerank_limit"] == 8
    assert payload["report_sha256"] == "c" * 64
    assert payload["reranker_backend"] == {
        "active": "torch",
        "fallback_used": True,
        "requested": "openvino",
        "status_code": "openvino_unavailable_fallback_torch",
    }
    assert [item["variant"] for item in payload["variants"]] == [
        "bm25",
        "dense",
        "hybrid_rrf",
        "hybrid_rerank",
    ]
    serialized = json.dumps(payload).casefold()
    assert "source_url" not in serialized
    assert "chunk_id" not in serialized
    assert "query" not in serialized
    assert "path" not in serialized


def test_smoke_command_outputs_only_aggregate_cross_entry_results(
    monkeypatch, capsys
) -> None:
    components = _components()
    monkeypatch.setattr(cli_module, "compose_public_demo", lambda **_kwargs: components)
    monkeypatch.setattr(cli_module, "smoke_public_demo", lambda _value: _smoke())

    assert cli_module.main(("smoke",)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "abc_dimension_count": 3,
        "agent_evidence_count": 6,
        "api_evidence_count": 6,
        "chunk_count": 24,
        "citation_sets_match": True,
        "human_review_required": True,
        "mcp_evidence_count": 6,
        "ok": True,
        "reranker_backend": {
            "active": "torch",
            "fallback_used": True,
            "requested": "openvino",
            "status_code": "openvino_unavailable_fallback_torch",
        },
    }


def test_cli_rejects_unknown_arguments_and_redacts_failures(
    monkeypatch, capsys
) -> None:
    assert cli_module.main(("evaluate", "--destination", "private.csv")) == 2
    invalid = json.loads(capsys.readouterr().out)
    assert invalid == {"error": "public_demo_request_invalid", "ok": False}

    assert cli_module.main(("prepare", "--reranker-backend=onnx")) == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "public_demo_request_invalid",
        "ok": False,
    }

    def _failed(**_kwargs):
        raise RuntimeError("D:/private/root/secret.csv")

    monkeypatch.setattr(cli_module, "compose_public_demo", _failed)
    assert cli_module.main(("smoke",)) == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed == {"error": "public_demo_failed", "ok": False}
    assert "secret" not in json.dumps(failed).casefold()


def test_cli_suppresses_third_party_warnings_without_hiding_safe_json(
    monkeypatch, capsys, recwarn
) -> None:
    def _warning_components(**_kwargs):
        warnings.warn("D:/private/root/from-third-party", UserWarning, stacklevel=1)
        return _components()

    monkeypatch.setattr(cli_module, "compose_public_demo", _warning_components)

    assert cli_module.main(("prepare",)) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is True
    assert captured.err == ""
    assert list(recwarn) == []
