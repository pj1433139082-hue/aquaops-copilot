import pytest

import aquaops.agent.public_diagnostics as diagnostics_module
from aquaops.agent.public_diagnostics import (
    expected_public_historical_rendering,
    parse_public_anomaly_request,
    parse_public_historical_request,
    render_public_historical_answer,
    run_public_historical_diagnostic,
)
from aquaops.agent.tools import ToolResult


def historical_result(**overrides: object) -> ToolResult:
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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
            ("nh4", 24, "2025-01-02T00:00:00Z"),
        ),
        (
            "公开历史诊断 COND 1h 截止 2025-01-02T00:00:00Z",
            ("cond", 1, "2025-01-02T00:00:00Z"),
        ),
        (
            "  公开历史诊断 q 168h 截止 2025-01-02T00:00:00Z  ",
            ("q", 168, "2025-01-02T00:00:00Z"),
        ),
    ],
)
def test_parser_accepts_only_complete_explicit_historical_requests(
    question: str, expected: tuple[str, int, str]
) -> None:
    assert parse_public_historical_request(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        None,
        [],
        "最近24小时氨氮怎么样",
        "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z 并自动调节曝气",
        "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z 多余文本",
        "公开历史诊断 氨氮 0h 截止 2025-01-02T00:00:00Z",
        "公开历史诊断 氨氮 169h 截止 2025-01-02T00:00:00Z",
        "公开历史诊断 氨氮 01h 截止 2025-01-02T00:00:00Z",
        "公开历史诊断 温度 24h 截止 2025-01-02T00:00:00Z",
        "公开历史诊断 NH4 24H 截止 2025-01-02T00:00:00Z",
        "公开历史诊断 nh4 24h 截止 2025-01-02 00:00:00Z",
    ],
)
def test_parser_rejects_non_explicit_or_out_of_range_requests(question: object) -> None:
    assert parse_public_historical_request(question) is None


def test_run_diagnostic_dispatches_only_parsed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = historical_result()
    calls: list[tuple[str, int, str]] = []

    def query(indicator: str, hours: int, end_at: str) -> ToolResult:
        calls.append((indicator, hours, end_at))
        return expected

    monkeypatch.setattr(diagnostics_module, "query_public_historical_window", query)

    result = run_public_historical_diagnostic(
        "公开历史诊断 电导 24h 截止 2025-01-02T00:00:00Z"
    )

    assert result is expected
    assert calls == [("cond", 24, "2025-01-02T00:00:00Z")]


def test_run_diagnostic_does_not_dispatch_unparsed_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_run(*args: object) -> ToolResult:
        raise AssertionError("ordinary questions must not access public history")

    monkeypatch.setattr(
        diagnostics_module, "query_public_historical_window", should_not_run
    )

    assert run_public_historical_diagnostic("最近氨氮异常吗") is None


def test_renderer_returns_safety_marked_aggregate_and_stable_evidence() -> None:
    answer, evidence = render_public_historical_answer(historical_result())

    assert "公开历史入流监测" in answer
    assert "非实时、不可用于自动控制" in answer
    assert "需人工复核" in answer
    assert "2025-01-01T00:00:00Z 至 2025-01-02T00:00:00Z" in answer
    assert "样本数 12" in answer
    assert "缺失 2" in answer
    assert "最小值 0.1" in answer
    assert "均值 0.3" in answer
    assert "最大值 0.5" in answer
    assert "mg/L" in answer
    assert evidence == [
        {
            "chunk_id": (
                "co-udlabs-wwtp-lpicm-2025:nh4:"
                "2025-01-01T00:00:00Z:2025-01-02T00:00:00Z"
            ),
            "source_url": "https://zenodo.org/records/15285089",
            "source_version": "v1.0.0 (2025-04-26)",
        }
    ]
    assert all(
        set(item) == {"chunk_id", "source_url", "source_version"} for item in evidence
    )


def test_expected_rendering_is_deterministic_and_renderer_reuses_it() -> None:
    result = historical_result()

    assert render_public_historical_answer(
        result
    ) == expected_public_historical_rendering(result)


def test_expected_rendering_rejects_top_level_data_version_mismatch() -> None:
    with pytest.raises(ValueError, match="invalid"):
        expected_public_historical_rendering(
            historical_result(data_version="forged-v1")
        )


def test_renderer_describes_all_null_statistics_without_none_or_raw_values() -> None:
    answer, evidence = render_public_historical_answer(
        historical_result(
            payload={
                "null_count": 12,
                "minimum": None,
                "maximum": None,
                "mean": None,
            }
        )
    )

    assert "该窗口无有效数值" in answer
    assert "None" not in answer
    assert "values" not in answer
    assert evidence[0]["chunk_id"].startswith("co-udlabs-wwtp-lpicm-2025:nh4:")


@pytest.mark.parametrize(
    "extra_payload",
    [
        {"values": []},
        {"rows": []},
        {"csv_path": "data/public/raw.csv"},
        {"nested": {"raw_values": []}},
    ],
)
def test_renderer_rejects_raw_or_unknown_aggregate_payload_fields(
    extra_payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="invalid"):
        render_public_historical_answer(historical_result(payload=extra_payload))


def test_renderer_rejects_forged_public_source_url() -> None:
    with pytest.raises(ValueError, match="invalid"):
        render_public_historical_answer(
            historical_result(payload={"source_url": "https://example.invalid"})
        )


def test_renderer_rejects_top_level_data_version_mismatch() -> None:
    with pytest.raises(ValueError, match="invalid"):
        render_public_historical_answer(historical_result(data_version="forged-v1"))


@pytest.mark.parametrize(
    "window",
    [
        {
            "start": "2025-01-02T00:00:00Z",
            "end": "2025-01-02T00:00:00Z",
        },
        {
            "start": "2025-01-03T00:00:00Z",
            "end": "2025-01-02T00:00:00Z",
        },
    ],
)
def test_renderer_rejects_non_increasing_public_window(
    window: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="invalid"):
        render_public_historical_answer(historical_result(payload=window))


def test_renderer_rejects_non_public_success_result() -> None:
    with pytest.raises(ValueError, match="public historical result"):
        render_public_historical_answer(
            historical_result(source_kind="public_or_synthetic")
        )


def anomaly_result(**overrides: object) -> ToolResult:
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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "公开异常筛查 氨氮 24h 截止 2025-01-02T00:00:00Z",
            ("nh4", 24, "2025-01-02T00:00:00Z"),
        ),
        (
            "  公开异常筛查 COND 1h 截止 2025-01-02T00:00:00Z  ",
            ("cond", 1, "2025-01-02T00:00:00Z"),
        ),
    ],
)
def test_parse_public_anomaly_request_accepts_only_explicit_syntax(
    question: str, expected: tuple[str, int, str]
) -> None:
    assert parse_public_anomaly_request(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "公开异常筛查 氨氮 24h",
        "公开异常筛查 氨氮 24h 截止 2025-01-02T00:00:00Z 自动加药",
        "公开异常筛查 氨氮 0h 截止 2025-01-02T00:00:00Z",
        "公开异常筛查 氨氮 169h 截止 2025-01-02T00:00:00Z",
        "公开异常筛查 氨氮 24H 截止 2025-01-02T00:00:00Z",
    ],
)
def test_parse_public_anomaly_request_rejects_deviation(question: object) -> None:
    assert parse_public_anomaly_request(question) is None


def test_run_diagnostic_dispatches_anomaly_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = anomaly_result()
    monkeypatch.setattr(
        diagnostics_module, "screen_public_historical_anomaly", lambda *args: expected
    )

    result = run_public_historical_diagnostic(
        "公开异常筛查 电导 24h 截止 2025-01-02T00:00:00Z"
    )

    assert result is expected


@pytest.mark.parametrize(
    ("payload", "required_text"),
    [
        ({}, "公开历史异常筛查"),
        (
            {
                "outcome": "normal",
                "robust_score": 1.0,
                "direction": "above_baseline",
            },
            "未超过公开历史筛查阈值，不等同于正常运行结论",
        ),
        (
            {
                "outcome": "insufficient_history",
                "method": "insufficient_history",
                "direction": "unavailable",
                "baseline_median": None,
                "baseline_scale": None,
                "robust_score": None,
                "history_count": 47,
                "history_null_count": 0,
            },
            "历史证据不足以完成筛查",
        ),
        (
            {
                "outcome": "target_unavailable",
                "method": "target_unavailable",
                "direction": "unavailable",
                "baseline_median": None,
                "baseline_scale": None,
                "robust_score": None,
            },
            "历史证据不足以完成筛查",
        ),
        (
            {
                "outcome": "insufficient_variability",
                "method": "insufficient_variability",
                "direction": "unavailable",
                "baseline_scale": 0.0,
                "robust_score": None,
            },
            "历史证据不足以完成筛查",
        ),
    ],
)
def test_anomaly_renderer_is_safe_and_outcome_specific(
    payload: dict[str, object], required_text: str
) -> None:
    answer, evidence = render_public_historical_answer(anomaly_result(payload=payload))

    assert required_text in answer
    assert "非实时、不可用于自动控制" in answer
    assert "需人工复核" in answer
    assert "observed_value" not in answer
    assert "values" not in answer
    assert evidence[0]["source_url"] == "https://zenodo.org/records/15285089"


def test_anomaly_renderer_rejects_target_value_or_forged_source() -> None:
    with pytest.raises(ValueError, match="invalid"):
        render_public_historical_answer(anomaly_result(payload={"target_value": 12.3}))
    with pytest.raises(ValueError, match="invalid"):
        render_public_historical_answer(
            anomaly_result(payload={"source_version": "forged-v1"})
        )
