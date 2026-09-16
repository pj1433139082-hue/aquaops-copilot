import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest

import aquaops.agent.graph as operations_graph_module
import aquaops.agent.public_diagnostics as public_diagnostics_module
import aquaops.api.routes.agent as agent_routes
from aquaops.agent.tools import ToolResult
from aquaops.api.app import create_app
from aquaops.research.graph import research_graph as research_graph_module


SAFE_REFUSAL = "信息不足，无法基于已公开证据给出建议。"


def _safe_graph_state(state: dict[str, object]) -> dict[str, object]:
    return {
        **state,
        "answer": SAFE_REFUSAL,
        "evidence": [],
        "requires_human_review": True,
    }


def _assert_completed_response(response: object) -> dict[str, object]:
    assert response.status_code == 200
    body = response.json()
    assert UUID(body["run_id"])
    assert body["status"] == "completed"
    assert body["answer"] == SAFE_REFUSAL
    assert body["evidence"] == []
    assert body["requires_human_review"] is True
    return body


@pytest.mark.parametrize("mode", ["operations", "research"])
def test_agent_request_returns_a_completed_queryable_result(mode: str) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/agent/runs",
        headers={"X-Request-ID": f"{mode}-trace-001"},
        json={"mode": mode, "question": "请评估 NH3-N 异常风险"},
    )

    body = _assert_completed_response(response)
    assert body["request_id"] == f"{mode}-trace-001"
    assert response.headers["X-Request-ID"] == body["request_id"]

    lookup = client.get(f"/v1/agent/runs/{body['run_id']}")

    assert lookup.status_code == 200
    assert lookup.json() == body


def test_duplicate_trace_ids_receive_distinct_server_generated_run_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_run_id = uuid4()
    second_run_id = uuid4()
    run_ids = iter([first_run_id, second_run_id])
    monkeypatch.setattr(agent_routes, "uuid4", lambda: next(run_ids), raising=False)
    client = TestClient(create_app())

    first = _assert_completed_response(
        client.post(
            "/v1/agent/runs",
            headers={"X-Request-ID": "duplicate-trace"},
            json={"mode": "operations", "question": "请评估 NH3-N 异常风险"},
        )
    )
    second = _assert_completed_response(
        client.post(
            "/v1/agent/runs",
            headers={"X-Request-ID": "duplicate-trace"},
            json={"mode": "research", "question": "调查总氮的公开研究证据"},
        )
    )

    assert first["request_id"] == second["request_id"] == "duplicate-trace"
    assert first["run_id"] == str(first_run_id)
    assert second["run_id"] == str(second_run_id)
    assert first["run_id"] != second["run_id"]
    assert client.get(f"/v1/agent/runs/{first['run_id']}").json() == first
    assert client.get(f"/v1/agent/runs/{second['run_id']}").json() == second


def test_server_regenerates_a_run_id_that_collides_with_a_stored_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    colliding_run_id = uuid4()
    replacement_run_id = uuid4()
    run_ids = iter([colliding_run_id, colliding_run_id, replacement_run_id])
    monkeypatch.setattr(agent_routes, "uuid4", lambda: next(run_ids), raising=False)
    client = TestClient(create_app())

    first = _assert_completed_response(
        client.post(
            "/v1/agent/runs",
            json={"mode": "operations", "question": "请评估 NH3-N 异常风险"},
        )
    )
    second = _assert_completed_response(
        client.post(
            "/v1/agent/runs",
            json={"mode": "research", "question": "调查总氮的公开研究证据"},
        )
    )

    assert first["run_id"] == str(colliding_run_id)
    assert second["run_id"] == str(replacement_run_id)
    assert client.get(f"/v1/agent/runs/{first['run_id']}").json() == first
    assert client.get(f"/v1/agent/runs/{second['run_id']}").json() == second


def test_empty_trace_id_is_replaced_with_a_generated_trace_id() -> None:
    response = TestClient(create_app()).post(
        "/v1/agent/runs",
        headers={"X-Request-ID": ""},
        json={"mode": "operations", "question": "请评估 NH3-N 异常风险"},
    )

    body = _assert_completed_response(response)
    assert UUID(body["request_id"])
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_agent_request_renders_a_safe_public_historical_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-task regression: keep parser, graph, renderer, and API real."""
    public_result = ToolResult(
        ok=True,
        code="ok",
        source_kind="public_historical",
        data_version="v1.0.0 (2025-04-26)",
        payload={
            "source_id": "co-udlabs-wwtp-lpicm-2025",
            "source_url": "https://zenodo.org/records/15285089",
            "source_version": "v1.0.0 (2025-04-26)",
            "indicator": "nh4",
            "unit": "mg/L",
            "start": "2020-06-11T22:55:00Z",
            "end": "2020-06-11T23:55:00Z",
            "row_count": 12,
            "null_count": 0,
            "minimum": 1.0,
            "mean": 2.0,
            "maximum": 3.0,
        },
    )
    public_window_query = Mock(return_value=public_result)
    monkeypatch.setattr(
        public_diagnostics_module,
        "query_public_historical_window",
        public_window_query,
    )

    response = TestClient(create_app()).post(
        "/v1/agent/runs",
        headers={"X-Request-ID": "public-history-demo"},
        json={
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2020-06-11T23:55:00Z",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["request_id"] == "public-history-demo"
    assert UUID(body["run_id"])
    assert body["requires_human_review"] is True
    assert "公开历史入流监测" in body["answer"]
    assert "非实时、不可用于自动控制" in body["answer"]
    assert body["evidence"] == [
        {
            "chunk_id": (
                "co-udlabs-wwtp-lpicm-2025:nh4:"
                "2020-06-11T22:55:00Z:2020-06-11T23:55:00Z"
            ),
            "source_url": "https://zenodo.org/records/15285089",
            "source_version": "v1.0.0 (2025-04-26)",
        }
    ]
    evidence = body["evidence"][0]
    assert set(evidence) == {"chunk_id", "source_url", "source_version"}
    assert not {"values", "rows", "csv_path"} & set(evidence)
    public_window_query.assert_called_once_with(
        "nh4",
        24,
        "2020-06-11T23:55:00Z",
    )


def test_agent_request_renders_a_safe_public_historical_anomaly_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API exposes only the reviewed, aggregate-only anomaly result."""
    public_result = ToolResult(
        ok=True,
        code="ok",
        source_kind="public_historical_anomaly",
        data_version="v1.0.0 (2025-04-26)",
        payload={
            "source_id": "co-udlabs-wwtp-lpicm-2025",
            "source_url": "https://zenodo.org/records/15285089",
            "source_version": "v1.0.0 (2025-04-26)",
            "indicator": "nh4",
            "unit": "mg/L",
            "target_time": "2020-06-11T23:55:00Z",
            "baseline_start": "2020-06-10T23:55:00Z",
            "baseline_end": "2020-06-11T23:55:00Z",
            "history_count": 96,
            "history_null_count": 0,
            "baseline_median": 10.0,
            "baseline_scale": 1.0,
            "robust_score": 4.0,
            "threshold": 3.5,
            "direction": "above_baseline",
            "method": "rolling_median_mad",
            "outcome": "anomaly",
        },
    )
    anomaly_screen = Mock(return_value=public_result)
    monkeypatch.setattr(
        public_diagnostics_module,
        "screen_public_historical_anomaly",
        anomaly_screen,
    )

    response = TestClient(create_app()).post(
        "/v1/agent/runs",
        headers={"X-Request-ID": "public-anomaly-demo"},
        json={
            "mode": "operations",
            "question": "公开异常筛查 氨氮 24h 截止 2020-06-11T23:55:00Z",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["request_id"] == "public-anomaly-demo"
    assert UUID(body["run_id"])
    assert body["requires_human_review"] is True
    assert "公开历史异常筛查" in body["answer"]
    assert "非实时、不可用于自动控制" in body["answer"]
    assert "values" not in body["answer"]
    assert "target_value" not in body["answer"]
    assert body["evidence"] == [
        {
            "chunk_id": (
                "co-udlabs-wwtp-lpicm-2025:nh4:anomaly:"
                "2020-06-10T23:55:00Z:2020-06-11T23:55:00Z"
            ),
            "source_url": "https://zenodo.org/records/15285089",
            "source_version": "v1.0.0 (2025-04-26)",
        }
    ]
    assert len(body["evidence"]) == 1
    anomaly_screen.assert_called_once_with(
        "nh4",
        24,
        "2020-06-11T23:55:00Z",
    )


@pytest.mark.parametrize(
    ("mode", "selected_graph", "unselected_graph"),
    [
        ("operations", "operations_graph", "research_graph"),
        ("research", "research_graph", "operations_graph"),
    ],
)
def test_agent_request_invokes_only_the_graph_selected_by_mode(
    mode: str,
    selected_graph: str,
    unselected_graph: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations_invoke = Mock(side_effect=_safe_graph_state)
    research_invoke = Mock(side_effect=_safe_graph_state)
    monkeypatch.setattr(
        operations_graph_module.operations_graph, "invoke", operations_invoke
    )
    monkeypatch.setattr(research_graph_module, "invoke", research_invoke)

    response = TestClient(create_app()).post(
        "/v1/agent/runs",
        headers={"X-Request-ID": "routing-001"},
        json={"mode": mode, "question": "  abc  "},
    )

    _assert_completed_response(response)
    selected_invoke = {
        "operations_graph": operations_invoke,
        "research_graph": research_invoke,
    }[selected_graph]
    unselected_invoke = {
        "operations_graph": operations_invoke,
        "research_graph": research_invoke,
    }[unselected_graph]
    selected_invoke.assert_called_once_with(
        {"request_id": "routing-001", "mode": mode, "question": "abc"}
    )
    unselected_invoke.assert_not_called()


def test_graph_failure_returns_a_safe_traceable_error_without_a_completed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app()
    run_id = uuid4()
    monkeypatch.setattr(agent_routes, "uuid4", lambda: run_id, raising=False)
    monkeypatch.setattr(
        operations_graph_module.operations_graph,
        "invoke",
        Mock(side_effect=RuntimeError("graph failure")),
    )

    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/agent/runs",
        headers={"X-Request-ID": "failure-001"},
        json={"mode": "operations", "question": "请评估 NH3-N 异常风险"},
    )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "failure-001"
    assert response.json() == {"detail": "Internal Server Error"}
    assert app.state.agent_run_store.get(run_id) is None


def test_agent_run_lookup_returns_404_for_an_unknown_run_id() -> None:
    response = TestClient(create_app()).get(f"/v1/agent/runs/{uuid4()}")

    assert response.status_code == 404


def test_trace_id_cannot_be_used_as_a_run_id_lookup_key() -> None:
    trace_id = "trace-id-is-not-a-run-id"
    client = TestClient(create_app())
    completed = _assert_completed_response(
        client.post(
            "/v1/agent/runs",
            headers={"X-Request-ID": trace_id},
            json={"mode": "operations", "question": "请评估 NH3-N 异常风险"},
        )
    )

    assert client.get(f"/v1/agent/runs/{trace_id}").status_code == 404
    assert client.get(f"/v1/agent/runs/{completed['run_id']}").json() == completed


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "operations"},
        {"mode": "operations", "question": "   "},
        {"mode": "operations", "question": "\t"},
        {"mode": "operations", "question": "\n"},
        {"mode": "operations", "question": 123},
        {"mode": "operations", "question": " a "},
        {"mode": "operations", "question": "  ab  "},
        {"mode": "operations", "question": f"  {'a' * 1001}  "},
    ],
    ids=[
        "missing",
        "spaces",
        "tab",
        "newline",
        "non-string",
        "trimmed-one",
        "trimmed-two",
        "trimmed-over-limit",
    ],
)
def test_agent_request_rejects_missing_blank_non_string_or_invalid_trimmed_questions(
    payload: dict[str, object],
) -> None:
    response = TestClient(create_app()).post("/v1/agent/runs", json=payload)

    assert response.status_code == 422


def test_agent_request_normalizes_surrounding_whitespace_before_running_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations_invoke = Mock(side_effect=_safe_graph_state)
    monkeypatch.setattr(
        operations_graph_module.operations_graph, "invoke", operations_invoke
    )

    response = TestClient(create_app()).post(
        "/v1/agent/runs",
        json={"mode": "operations", "question": "  abc  "},
    )

    _assert_completed_response(response)
    assert operations_invoke.call_args.args[0]["question"] == "abc"


def test_agent_request_accepts_a_question_at_the_trimmed_length_limit() -> None:
    response = TestClient(create_app()).post(
        "/v1/agent/runs",
        json={"mode": "operations", "question": f"  {'a' * 1000}  "},
    )

    _assert_completed_response(response)


def test_agent_request_requires_mode() -> None:
    response = TestClient(create_app()).post(
        "/v1/agent/runs",
        json={"question": "模拟 NH3-N 异常"},
    )

    assert response.status_code == 422


def test_agent_request_rejects_unknown_fields_and_returns_request_id() -> None:
    response = TestClient(create_app()).post(
        "/v1/agent/runs",
        headers={"X-Request-ID": "validation-001"},
        json={
            "mode": "operations",
            "question": "模拟 NH3-N 异常",
            "unknown_option": True,
        },
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "validation-001"


def test_research_request_does_not_load_operations_tools() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fastapi.testclient import TestClient; "
                "from aquaops.api.app import create_app; "
                "response = TestClient(create_app()).post("
                "'/v1/agent/runs', "
                "json={'mode': 'research', 'question': '调查总氮的公开研究证据'}); "
                "assert response.status_code == 200; "
                "assert 'aquaops.research.graph' in __import__('sys').modules; "
                "assert 'aquaops.agent.graph' not in __import__('sys').modules; "
                "assert 'aquaops.agent.tools' not in __import__('sys').modules"
            ),
        ],
        capture_output=True,
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_unhandled_error_returns_a_safe_response_with_request_id() -> None:
    app = create_app()

    @app.get("/test-error")
    def raise_error() -> None:
        raise RuntimeError("unhandled test error")

    response = TestClient(app, raise_server_exceptions=False).get(
        "/test-error",
        headers={"X-Request-ID": "error-001"},
    )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "error-001"
    assert response.json() == {"detail": "Internal Server Error"}
