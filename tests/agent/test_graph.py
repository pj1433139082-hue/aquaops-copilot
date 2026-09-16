import ast
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import aquaops.agent.graph as operations_graph
import aquaops.agent.tools as operations_tools
import pytest
from aquaops.agent.coverage import ABCPlan, mint_verified_public_evidence
from aquaops.agent.policy import (
    ToolBudget,
    ToolPolicyDecision,
    consume_issued_public_tool_grant,
    create_public_tool_budget_session,
    issue_public_tool_grant,
)
from aquaops.agent.tools import ToolResult
from aquaops.rag.hybrid import PublicEvidence, RetrievalResult


SAFE_REFUSAL = "信息不足，无法基于已公开证据给出建议。"
GRAPH_MODULE_PATHS = (
    Path(__file__).resolve().parents[2] / "src" / "aquaops" / "agent" / "graph.py",
    Path(__file__).resolve().parents[2] / "src" / "aquaops" / "research" / "graph.py",
)
PUBLIC_DIAGNOSTICS_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "aquaops"
    / "agent"
    / "public_diagnostics.py"
)
OUTBOUND_IMPORT_ROOTS = frozenset(
    {
        "anthropic",
        "aiohttp",
        "fastapi",
        "httpx",
        "openai",
        "qdrant_client",
        "requests",
        "socket",
        "subprocess",
        "urllib",
        "websockets",
    }
)
OUTBOUND_CALL_IDENTITIES = frozenset(
    {
        "anthropic.Anthropic",
        "fastapi.BackgroundTasks",
        "httpx.Client",
        "httpx.get",
        "httpx.post",
        "openai.OpenAI",
        "qdrant_client.QdrantClient",
        "requests.get",
        "requests.post",
        "socket.create_connection",
        "subprocess.run",
        "urllib.request.urlopen",
    }
)
FORBIDDEN_GRAPH_MODULES = frozenset(
    {
        "aquaops.agent.tools",
        "aquaops.rag.ingest",
        "aquaops.rag.retrieve",
        "aquaops.research.search_provider",
    }
)


def _assert_safe_refusal(state: dict[str, object]) -> None:
    assert state["requires_human_review"] is True
    assert state["answer"] == SAFE_REFUSAL
    assert state["evidence"] == []


def _verified_aggregate_evidence() -> object:
    session = create_public_tool_budget_session()
    decision, grant = issue_public_tool_grant(
        "aggregate_public_history",
        {
            "indicator": "nh4",
            "hours": 24,
            "end_at": "2025-01-02T00:00:00Z",
        },
        session,
    )
    assert decision.allowed is True and grant is not None
    consumed = consume_issued_public_tool_grant(
        grant, expected_name="aggregate_public_history", session=session
    )
    assert consumed is not None
    receipt, _ = consumed
    return mint_verified_public_evidence(
        receipt=receipt,
        evidence_id="public:a:1",
        source_url="https://zenodo.org/records/15285089",
        source_version="v1.0.0 (2025-04-26)",
    )


def _public_historical_result(**overrides: object) -> ToolResult:
    payload: dict[str, object] = {
        "source_id": "co-udlabs-wwtp-lpicm-2025",
        "source_url": "https://zenodo.org/records/15285089",
        "source_version": "v1.0.0 (2025-04-26)",
        "indicator": "nh4",
        "unit": "mg/L",
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-02T00:00:00Z",
        "row_count": 12,
        "null_count": 2,
        "minimum": 0.1,
        "maximum": 0.5,
        "mean": 0.3,
    }
    payload.update(overrides.pop("payload", {}))
    return ToolResult(
        ok=overrides.pop("ok", True),
        code=overrides.pop("code", "ok"),
        source_kind=overrides.pop("source_kind", "public_historical"),
        data_version=overrides.pop("data_version", payload["source_version"]),
        payload=payload,
        **overrides,
    )


def _public_historical_anomaly_result(**overrides: object) -> ToolResult:
    payload: dict[str, object] = {
        "source_id": "co-udlabs-wwtp-lpicm-2025",
        "source_url": "https://zenodo.org/records/15285089",
        "source_version": "v1.0.0 (2025-04-26)",
        "indicator": "nh4",
        "unit": "mg/L",
        "target_time": "2025-01-02T00:00:00Z",
        "baseline_start": "2025-01-01T00:00:00Z",
        "baseline_end": "2025-01-02T00:00:00Z",
        "history_count": 50,
        "history_null_count": 2,
        "baseline_median": 0.3,
        "baseline_scale": 0.1,
        "robust_score": 4.0,
        "threshold": 3.5,
        "direction": "above_baseline",
        "method": "rolling_median_mad",
        "outcome": "anomaly",
    }
    payload.update(overrides.pop("payload", {}))
    return ToolResult(
        ok=overrides.pop("ok", True),
        code=overrides.pop("code", "ok"),
        source_kind=overrides.pop("source_kind", "public_historical_anomaly"),
        data_version=overrides.pop("data_version", payload["source_version"]),
        payload=payload,
        **overrides,
    )


def _call_identity(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_identity(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _imported_module_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _assert_no_forbidden_graph_dependencies(tree: ast.AST) -> None:
    imported_modules = _imported_module_names(tree)
    forbidden_modules = imported_modules & FORBIDDEN_GRAPH_MODULES
    assert not forbidden_modules, (
        f"forbidden capability adapters: {sorted(forbidden_modules)}"
    )


def test_graph_dependency_guard_rejects_internal_capability_adapter_imports() -> None:
    tree = ast.parse("from aquaops.agent import tools")

    with pytest.raises(AssertionError, match="aquaops.agent.tools"):
        _assert_no_forbidden_graph_dependencies(tree)


def test_execution_graph_modules_have_no_direct_outbound_capability_imports_or_calls() -> (
    None
):
    """Guards direct graph imports/calls, not third-party or dynamic behavior."""
    for module_path in GRAPH_MODULE_PATHS:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imports = {name.split(".")[0] for name in _imported_module_names(tree)}
        calls = {
            identity
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            if (identity := _call_identity(node.func)) is not None
        }

        assert not imports & OUTBOUND_IMPORT_ROOTS, module_path
        assert not calls & OUTBOUND_CALL_IDENTITIES, module_path
        _assert_no_forbidden_graph_dependencies(tree)


def test_public_diagnostics_has_no_direct_outbound_capability_imports_or_calls() -> (
    None
):
    """The local public-history adapter may use tools, but never outbound clients."""
    tree = ast.parse(PUBLIC_DIAGNOSTICS_MODULE_PATH.read_text(encoding="utf-8"))
    imports = {name.split(".")[0] for name in _imported_module_names(tree)}
    calls = {
        identity
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (identity := _call_identity(node.func)) is not None
    }

    assert not imports & OUTBOUND_IMPORT_ROOTS, PUBLIC_DIAGNOSTICS_MODULE_PATH
    assert not calls & OUTBOUND_CALL_IDENTITIES, PUBLIC_DIAGNOSTICS_MODULE_PATH


@pytest.mark.parametrize("mode", ["operations", "research"])
def test_execution_graphs_remain_evidence_gated_safe_refusals(mode: str) -> None:
    if mode == "operations":
        graph = operations_graph.operations_graph
    else:
        from aquaops.research.graph import research_graph

        graph = research_graph

    result = graph.invoke(
        {
            "request_id": f"safe-{mode}",
            "mode": mode,
            "question": "请评估 NH3-N 异常风险",
        }
    )

    _assert_safe_refusal(result)


def test_operations_graph_requires_human_review_when_evidence_is_missing() -> None:
    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-1",
            "mode": "operations",
            "question": "NH3-N 是否异常？",
        }
    )

    _assert_safe_refusal(state)


def test_compiled_operations_graph_preserves_safe_refusal() -> None:
    state = operations_graph.operations_graph.invoke(
        {
            "request_id": "case-2",
            "mode": "operations",
            "question": "NH3-N 是否异常？",
        }
    )

    _assert_safe_refusal(state)


def test_operations_graph_renders_valid_explicit_public_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _public_historical_result()
    monkeypatch.setattr(
        operations_graph,
        "run_public_historical_diagnostic",
        lambda question: expected,
    )

    state = operations_graph.operations_graph.invoke(
        {
            "request_id": "case-public-history-success",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    assert state["requires_human_review"] is True
    assert "非实时、不可用于自动控制" in state["answer"]
    assert "需人工复核" in state["answer"]
    assert state["evidence"] == [
        {
            "chunk_id": (
                "co-udlabs-wwtp-lpicm-2025:nh4:"
                "2025-01-01T00:00:00Z:2025-01-02T00:00:00Z"
            ),
            "source_url": "https://zenodo.org/records/15285089",
            "source_version": "v1.0.0 (2025-04-26)",
        }
    ]


def test_operations_graph_policy_denial_happens_before_diagnostic_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    denied = ToolPolicyDecision(
        allowed=False,
        code="tool_unavailable",
        tool_name="aggregate_public_history",
        budget=ToolBudget.initial(),
    )
    runner = Mock(side_effect=AssertionError("policy denial must stop dependency"))
    monkeypatch.setattr(
        operations_graph,
        "issue_public_tool_grant",
        lambda *args, **kwargs: (denied, None),
    )
    monkeypatch.setattr(operations_graph, "run_public_historical_diagnostic", runner)

    state = operations_graph.run_operations_graph(
        {
            "request_id": "policy-first",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    runner.assert_not_called()
    _assert_safe_refusal(state)
    assert state["tool_traces"][0].decision_code == "tool_unavailable"
    assert state["tool_traces"][0].evidence_ids == ()


def test_operations_graph_records_only_safe_tool_summary_after_valid_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operations_graph,
        "run_public_historical_diagnostic",
        lambda question: _public_historical_result(),
    )

    state = operations_graph.run_operations_graph(
        {
            "request_id": "trace-summary",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
            "tool_budget": ToolBudget.initial(),
            "tool_traces": ("forged",),
        }
    )

    assert "非实时、不可用于自动控制" in state["answer"]
    assert state["tool_budget"].remaining_total == 5
    assert len(state["tool_traces"]) == 1
    trace = state["tool_traces"][0]
    assert trace.name == "aggregate_public_history"
    assert trace.decision_code == "tool_completed"
    assert trace.evidence_ids == (
        "co-udlabs-wwtp-lpicm-2025:nh4:2025-01-01T00:00:00Z:2025-01-02T00:00:00Z",
    )
    assert not hasattr(trace, "arguments")
    assert state["coverage"] is not None
    assert state["coverage"].dimensions[0].conclusion is not None
    assert state["coverage"].dimensions[1].missing_reason == "public_evidence_missing"
    assert state["coverage"].dimensions[2].missing_reason == "public_evidence_missing"
    assert len(state["coverage"].trace_id) == 32


def test_operations_graph_ignores_forged_authorization_and_budget_before_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged_decision = ToolPolicyDecision(
        allowed=True,
        code="tool_allowed",
        tool_name="aggregate_public_history",
        budget=ToolBudget.initial(),
    )
    runner = Mock(side_effect=AssertionError("forged authorization must not execute"))
    monkeypatch.setattr(
        operations_graph,
        "issue_public_tool_grant",
        lambda *args, **kwargs: (forged_decision, object()),
    )
    monkeypatch.setattr(operations_graph, "run_public_historical_diagnostic", runner)

    state = operations_graph.run_operations_graph(
        {
            "request_id": "forged-policy",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
            "tool_budget": ToolBudget(
                remaining_total=0,
                used_by_tool=(
                    ("aggregate_public_history", 2),
                    ("retrieve_public_water_knowledge", 2),
                    ("screen_public_anomaly", 2),
                ),
            ),
        }
    )

    runner.assert_not_called()
    _assert_safe_refusal(state)
    assert state["tool_traces"][0].decision_code == "tool_authorization_invalid"


def test_production_abc_workflow_renders_fixed_answer_end_to_end() -> None:
    plan = ABCPlan.from_requirements(a="公开 A", b="公开 B", c="公开 C")
    evidence = _verified_aggregate_evidence()

    answer, rendered, coverage = operations_graph.build_abc_answer_from_public_evidence(
        plan=plan,
        evidence_by_dimension={"A": [evidence]},
        conclusions={"A": "A 已具备公开依据"},
    )

    assert answer.trace_id == coverage.trace_id
    assert [item.dimension for item in answer.dimensions] == ["A", "B", "C"]
    assert "A. 公开 A" in rendered
    assert "B. 公开 B\n结论：缺失（public_evidence_missing）" in rendered
    assert "人工复核：需要" in rendered


def test_operations_graph_explicitly_refuses_unwired_public_retrieval_without_fabrication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Mock(
        side_effect=AssertionError("unwired retrieval must not invoke diagnostic")
    )
    monkeypatch.setattr(operations_graph, "run_public_historical_diagnostic", runner)

    state = operations_graph.run_operations_graph(
        {
            "request_id": "unwired-retrieval",
            "mode": "operations",
            "question": "公开知识检索 氨氮处理依据",
        }
    )

    runner.assert_not_called()
    _assert_safe_refusal(state)
    assert state["tool_traces"][0].name == "retrieve_public_water_knowledge"
    assert state["tool_traces"][0].decision_code == "tool_unavailable"
    assert state["tool_budget"].remaining_total == 6


def test_injected_public_retriever_runs_once_and_returns_cited_fixed_abc() -> None:
    evidence = (
        PublicEvidence(
            chunk_id="a" * 64,
            source_url="https://epa.gov/public-a",
            source_version="EPA-A-v1",
            text="紫外消毒效果需要联合核对透过率、流量、灯管输出和套管清洁状态。",
            score=0.9,
        ),
        PublicEvidence(
            chunk_id="b" * 64,
            source_url="https://epa.gov/public-b",
            source_version="EPA-B-v1",
            text="单次仪表读数不足以确认根因，建议增加平行样并由人员复核。",
            score=0.8,
        ),
    )

    class Retriever:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def retrieve(self, query: str, answer_limit: int = 6) -> RetrievalResult:
            self.calls.append((query, answer_limit))
            return RetrievalResult(
                evidence=evidence,
                fused_candidate_ids=tuple(item.chunk_id for item in evidence),
            )

    retriever = Retriever()
    graph = operations_graph.build_operations_graph(retriever=retriever)

    state = graph.invoke(
        {
            "request_id": "real-public-retrieval",
            "mode": "operations",
            "question": "公开知识检索 紫外消毒效果下降如何排查",
        }
    )

    assert retriever.calls == [("紫外消毒效果下降如何排查", 6)]
    assert "question" not in state
    assert state["requires_human_review"] is True
    assert state["tool_budget"].remaining_total == 5
    assert len(state["tool_traces"]) == 1
    assert state["tool_traces"][0].decision_code == "tool_completed"
    assert state["tool_traces"][0].evidence_ids == ("a" * 64, "b" * 64)
    assert [item["chunk_id"] for item in state["evidence"]] == ["a" * 64, "b" * 64]
    assert all(
        set(item) == {"chunk_id", "source_url", "source_version"}
        for item in state["evidence"]
    )
    assert "A. 公开历史趋势或聚合证据" in state["answer"]
    assert "B. 公开异常筛查证据" in state["answer"]
    assert "C. 公开知识依据" in state["answer"]
    assert "public_evidence_missing" in state["answer"]
    assert "人工复核：需要" in state["answer"]
    assert state["coverage"].dimensions[2].conclusion is not None
    assert tuple(
        item.evidence_id for item in state["coverage"].dimensions[2].evidence
    ) == ("a" * 64, "b" * 64)


def test_injected_public_retriever_failure_is_redacted_safe_refusal() -> None:
    class Retriever:
        def retrieve(self, query: str, answer_limit: int = 6) -> RetrievalResult:
            raise RuntimeError("must not escape")

    state = operations_graph.build_operations_graph(retriever=Retriever()).invoke(
        {
            "request_id": "failed-public-retrieval",
            "mode": "operations",
            "question": "公开知识检索 膜污染如何排查",
        }
    )

    _assert_safe_refusal(state)
    assert "question" not in state
    assert state["tool_traces"][0].decision_code == "tool_result_unavailable"
    assert not state["tool_traces"][0].evidence_ids


def test_compiled_operations_graph_refuses_non_operations_mode_before_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unexpected_runner = Mock(side_effect=AssertionError("runner must not be called"))
    monkeypatch.setattr(
        operations_graph, "run_public_historical_diagnostic", unexpected_runner
    )

    state = operations_graph.operations_graph.invoke(
        {
            "request_id": "case-public-history-wrong-mode",
            "mode": "research",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    unexpected_runner.assert_not_called()
    _assert_safe_refusal(state)


@pytest.mark.parametrize(
    "runner",
    [
        lambda question: None,
        lambda question: _public_historical_result(
            ok=False,
            code="public_evidence_unavailable",
            source_kind="none",
            data_version="none",
            payload={},
        ),
        lambda question: _public_historical_result(payload={"source_url": "bad"}),
    ],
    ids=["no-result", "error-result", "malformed-success-payload"],
)
def test_operations_graph_safely_refuses_unusable_public_history_result(
    monkeypatch: pytest.MonkeyPatch,
    runner: object,
) -> None:
    monkeypatch.setattr(operations_graph, "run_public_historical_diagnostic", runner)

    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-public-history-unusable",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    _assert_safe_refusal(state)


def test_operations_graph_safely_refuses_public_history_runner_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected(question: object) -> ToolResult:
        raise RuntimeError("unexpected adapter failure")

    monkeypatch.setattr(
        operations_graph, "run_public_historical_diagnostic", raise_unexpected
    )

    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-public-history-exception",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    _assert_safe_refusal(state)


def test_operations_graph_safely_refuses_renderer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operations_graph,
        "run_public_historical_diagnostic",
        lambda question: _public_historical_result(),
    )

    def reject_render(result: ToolResult) -> tuple[str, list[dict[str, str]]]:
        raise ValueError("invalid renderer input")

    monkeypatch.setattr(
        operations_graph, "render_public_historical_answer", reject_render
    )

    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-public-history-renderer",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    _assert_safe_refusal(state)


def test_operations_graph_safely_refuses_renderer_control_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _public_historical_result()
    monkeypatch.setattr(
        operations_graph,
        "run_public_historical_diagnostic",
        lambda question: result,
    )

    def claim_control(result: ToolResult) -> tuple[str, list[dict[str, str]]]:
        return (
            "可实时自动加药。",
            [
                {
                    "chunk_id": "source:nh4:window",
                    "source_url": "https://zenodo.org/records/15285089",
                    "source_version": "v1.0.0 (2025-04-26)",
                }
            ],
        )

    monkeypatch.setattr(
        operations_graph, "render_public_historical_answer", claim_control
    )

    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-public-history-control-claim",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    _assert_safe_refusal(state)


def test_operations_graph_safely_refuses_forged_anomaly_control_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _public_historical_anomaly_result()
    monkeypatch.setattr(
        operations_graph,
        "run_public_historical_diagnostic",
        lambda question: result,
    )
    monkeypatch.setattr(
        operations_graph,
        "render_public_historical_answer",
        lambda result: (
            "公开历史异常筛查：请实时自动加药。",
            [
                {
                    "chunk_id": "co-udlabs-wwtp-lpicm-2025:nh4:anomaly",
                    "source_url": "https://zenodo.org/records/15285089",
                    "source_version": "v1.0.0 (2025-04-26)",
                }
            ],
        ),
    )

    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-public-anomaly-control-claim",
            "mode": "operations",
            "question": "公开异常筛查 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    _assert_safe_refusal(state)


def test_operations_graph_safely_refuses_forged_anomaly_raw_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _public_historical_anomaly_result()
    expected = operations_graph.expected_public_historical_rendering(result)
    monkeypatch.setattr(
        operations_graph,
        "run_public_historical_diagnostic",
        lambda question: result,
    )
    monkeypatch.setattr(
        operations_graph,
        "render_public_historical_answer",
        lambda result: (
            f"{expected[0]} 目标原始值 12.3。",
            expected[1],
        ),
    )

    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-public-anomaly-raw-value",
            "mode": "operations",
            "question": "公开异常筛查 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    _assert_safe_refusal(state)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("chunk_id", "forged:nh4:window"),
        ("source_url", "https://example.invalid"),
        ("source_version", "forged-v1"),
    ],
)
def test_operations_graph_safely_refuses_renderer_forged_evidence(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    forged_value: str,
) -> None:
    result = _public_historical_result()
    answer, evidence = operations_graph.render_public_historical_answer(result)
    forged_evidence = [dict(evidence[0], **{field: forged_value})]
    monkeypatch.setattr(
        operations_graph,
        "run_public_historical_diagnostic",
        lambda question: result,
    )
    monkeypatch.setattr(
        operations_graph,
        "render_public_historical_answer",
        lambda result: (answer, forged_evidence),
    )

    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-public-history-forged-evidence",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    _assert_safe_refusal(state)


def test_operations_graph_safely_refuses_top_level_data_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operations_graph,
        "run_public_historical_diagnostic",
        lambda question: _public_historical_result(data_version="forged-v1"),
    )

    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-public-history-version-mismatch",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    _assert_safe_refusal(state)


@pytest.mark.parametrize(
    "rendered",
    [
        ("公开历史结果", []),
        (
            42,
            [
                {
                    "chunk_id": "source:nh4:window",
                    "source_url": "https://zenodo.org/records/15285089",
                    "source_version": "v1.0.0 (2025-04-26)",
                }
            ],
        ),
        (
            "公开历史结果",
            [
                {
                    "chunk_id": "source:nh4:window",
                    "source_url": "https://zenodo.org/records/15285089",
                }
            ],
        ),
        (
            "公开历史结果",
            [
                {
                    "chunk_id": "source:nh4:window",
                    "source_url": "https://zenodo.org/records/15285089",
                    "source_version": "v1.0.0 (2025-04-26)",
                    "raw_value": "must-not-leak",
                }
            ],
        ),
    ],
    ids=[
        "empty-evidence",
        "non-string-answer",
        "missing-evidence-key",
        "extra-evidence-key",
    ],
)
def test_operations_graph_safely_refuses_malformed_renderer_output(
    monkeypatch: pytest.MonkeyPatch,
    rendered: object,
) -> None:
    monkeypatch.setattr(
        operations_graph,
        "run_public_historical_diagnostic",
        lambda question: _public_historical_result(),
    )

    def return_malformed(result: ToolResult) -> object:
        return rendered

    monkeypatch.setattr(
        operations_graph, "render_public_historical_answer", return_malformed
    )

    state = operations_graph.run_operations_graph(
        {
            "request_id": "case-public-history-malformed-renderer-output",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
        }
    )

    _assert_safe_refusal(state)


def test_operations_graph_never_calls_signal_window_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unexpected_tool = Mock(side_effect=AssertionError("tool must not be called"))
    monkeypatch.setattr(operations_tools, "query_signal_window", unexpected_tool)
    assert "query_signal_window" not in vars(operations_graph)

    state = operations_graph.operations_graph.invoke(
        {
            "request_id": "case-operations-no-tool",
            "mode": "operations",
            "question": "NH3-N 是否异常？",
        }
    )

    unexpected_tool.assert_not_called()
    _assert_safe_refusal(state)


@pytest.mark.parametrize(
    "state",
    [
        {"request_id": "case-operations-missing", "mode": "operations"},
        {
            "request_id": "case-operations-blank",
            "mode": "operations",
            "question": "   ",
        },
    ],
    ids=["missing-question", "blank-question"],
)
def test_operations_graph_safely_refuses_invalid_questions(
    state: dict[str, object],
) -> None:
    result = operations_graph.operations_graph.invoke(state)

    _assert_safe_refusal(result)


def test_research_graph_import_does_not_load_operations_tool() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "importlib.import_module('aquaops.research.graph'); "
                "assert 'aquaops.agent.tools' not in sys.modules"
            ),
        ],
        capture_output=True,
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_research_graph_preserves_safe_refusal() -> None:
    from aquaops.research.graph import research_graph

    state = research_graph.invoke(
        {
            "request_id": "case-3",
            "mode": "research",
            "question": "氨氮异常的公开研究证据有哪些？",
        }
    )

    _assert_safe_refusal(state)


@pytest.mark.parametrize(
    "state",
    [
        {"request_id": "case-research-missing", "mode": "research"},
        {
            "request_id": "case-research-blank",
            "mode": "research",
            "question": "   ",
        },
    ],
    ids=["missing-question", "blank-question"],
)
def test_research_graph_safely_refuses_invalid_questions(
    state: dict[str, object],
) -> None:
    from aquaops.research.graph import research_graph

    result = research_graph.invoke(state)

    _assert_safe_refusal(result)
