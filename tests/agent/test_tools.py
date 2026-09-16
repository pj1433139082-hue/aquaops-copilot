import pytest

import aquaops.agent.tools as tools_module
from dataclasses import asdict, replace

from aquaops.data.public_window import PublicWindowError, PublicWindowSummary
from aquaops.agent.tools import (
    PublicHistoricalAnomaly,
    query_public_historical_window,
    query_signal_window,
    retrieve_knowledge,
    screen_public_historical_anomaly,
    score_anomaly,
)
from aquaops.data.public_anomaly import PublicAnomalySummary


def public_summary() -> PublicWindowSummary:
    return PublicWindowSummary(
        source_id="co-udlabs-wwtp-lpicm-2025",
        source_url="https://zenodo.org/records/15285089",
        source_version="v1.0.0 (2025-04-26)",
        indicator="nh4",
        unit="mg/L",
        start="2025-01-01T00:00:00Z",
        end="2025-01-02T00:00:00Z",
        row_count=12,
        null_count=2,
        minimum=0.1,
        maximum=0.5,
        mean=0.3,
    )


def test_query_public_historical_window_returns_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = public_summary()
    calls: list[tuple[str, int, str]] = []

    def summarize(indicator: str, hours: int, end_at: str) -> PublicWindowSummary:
        calls.append((indicator, hours, end_at))
        return expected

    monkeypatch.setattr(tools_module, "summarize_public_window", summarize)

    result = query_public_historical_window("nh4", 24, "2025-01-02T00:00:00Z")

    assert calls == [("nh4", 24, "2025-01-02T00:00:00Z")]
    assert result.ok is True
    assert result.code == "ok"
    assert result.source_kind == "public_historical"
    assert result.data_version == expected.source_version
    assert result.payload == {
        "source_id": expected.source_id,
        "source_url": expected.source_url,
        "source_version": expected.source_version,
        "indicator": expected.indicator,
        "unit": expected.unit,
        "start": expected.start,
        "end": expected.end,
        "row_count": expected.row_count,
        "null_count": expected.null_count,
        "minimum": expected.minimum,
        "maximum": expected.maximum,
        "mean": expected.mean,
    }
    assert not {"values", "rows", "samples", "csv_path"} & set(result.payload)


@pytest.mark.parametrize(
    ("indicator", "hours", "end_at"),
    [
        ([], 24, "2025-01-02T00:00:00Z"),
        ("nh3_n", 24, "2025-01-02T00:00:00Z"),
        ("nh4", True, "2025-01-02T00:00:00Z"),
        ("nh4", 1.0, "2025-01-02T00:00:00Z"),
        ("nh4", 0, "2025-01-02T00:00:00Z"),
        ("nh4", 169, "2025-01-02T00:00:00Z"),
        ("nh4", 24, None),
        ("nh4", 24, "not-a-timestamp"),
        ("nh4", 24, "2025-01-02T00:00:00"),
        ("nh4", 24, "2025-01-02T08:00:00+08:00"),
        ("nh4", 24, "2025-01-02T00:00Z"),
        ("nh4", 24, "2025-02-30T00:00:00Z"),
    ],
)
def test_query_public_historical_window_rejects_invalid_requests_without_data_read(
    monkeypatch: pytest.MonkeyPatch,
    indicator: object,
    hours: object,
    end_at: object,
) -> None:
    def should_not_run(*args: object) -> PublicWindowSummary:
        raise AssertionError("invalid request must not invoke data access")

    monkeypatch.setattr(tools_module, "summarize_public_window", should_not_run)

    result = query_public_historical_window(indicator, hours, end_at)

    assert result.ok is False
    assert result.code == "invalid_public_request"
    assert result.source_kind == "none"
    assert result.data_version == "none"
    assert result.payload == {}


@pytest.mark.parametrize("failure", [PublicWindowError("unavailable"), OSError("io")])
def test_query_public_historical_window_masks_safe_data_failures(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def raise_failure(*args: object) -> PublicWindowSummary:
        raise failure

    monkeypatch.setattr(tools_module, "summarize_public_window", raise_failure)

    result = query_public_historical_window("nh4", 24, "2025-01-02T00:00:00Z")

    assert result.ok is False
    assert result.code == "public_evidence_unavailable"
    assert result.source_kind == "none"
    assert result.data_version == "none"
    assert result.payload == {}


def test_query_public_historical_window_rejects_untrusted_summary_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools_module,
        "summarize_public_window",
        lambda *args: replace(public_summary(), source_url="https://example.invalid"),
    )

    result = query_public_historical_window("nh4", 24, "2025-01-02T00:00:00Z")

    assert result.ok is False
    assert result.code == "public_evidence_unavailable"
    assert result.source_kind == "none"
    assert result.data_version == "none"
    assert result.payload == {}


def test_query_signal_window_rejects_unknown_indicator() -> None:
    result = query_signal_window(indicator="password", hours=24)
    assert result.ok is False
    assert result.code == "unsupported_indicator"


def test_query_signal_window_returns_versioned_evidence() -> None:
    result = query_signal_window(indicator="nh3_n", hours=24)
    assert result.ok is True
    assert result.data_version == "demo-v1"
    assert result.source_kind == "public_or_synthetic"
    assert result.payload == {"indicator": "nh3_n", "hours": 24, "values": []}


@pytest.mark.parametrize("hours", [1, 168])
def test_query_signal_window_accepts_hour_boundaries(hours: int) -> None:
    result = query_signal_window(indicator="nh3_n", hours=hours)
    assert result.ok is True
    assert result.payload == {"indicator": "nh3_n", "hours": hours, "values": []}


@pytest.mark.parametrize("hours", [True, 1.0, "24", 0, 169])
def test_query_signal_window_rejects_invalid_hours(hours: object) -> None:
    result = query_signal_window(indicator="nh3_n", hours=hours)
    assert result.ok is False
    assert result.code == "invalid_hours"
    assert result.source_kind == "none"
    assert result.data_version == "none"


def test_query_signal_window_rejects_non_string_indicator() -> None:
    result = query_signal_window(indicator=[], hours=24)
    assert result.ok is False
    assert result.code == "unsupported_indicator"
    assert result.source_kind == "none"
    assert result.data_version == "none"


@pytest.mark.parametrize("query", [None, [], "  ", "ab"])
def test_retrieve_knowledge_rejects_invalid_query(query: object) -> None:
    result = retrieve_knowledge(query)
    assert result.ok is False
    assert result.code == "invalid_query"
    assert result.source_kind == "none"
    assert result.data_version == "none"


def test_retrieve_knowledge_does_not_fabricate_evidence() -> None:
    result = retrieve_knowledge("  ammonia nitrogen guidance  ")
    assert result.ok is False
    assert result.code == "evidence_unavailable"
    assert result.source_kind == "none"
    assert result.data_version == "none"
    assert result.payload == {}


def test_score_anomaly_rejects_unknown_indicator() -> None:
    result = score_anomaly(indicator="password", hours=24)
    assert result.ok is False
    assert result.code == "unsupported_indicator"
    assert result.source_kind == "none"
    assert result.data_version == "none"


def test_score_anomaly_rejects_non_string_indicator() -> None:
    result = score_anomaly(indicator=[], hours=24)
    assert result.ok is False
    assert result.code == "unsupported_indicator"
    assert result.source_kind == "none"
    assert result.data_version == "none"


@pytest.mark.parametrize("hours", [True, 1.0, "24", 0, 169])
def test_score_anomaly_rejects_invalid_hours(hours: object) -> None:
    result = score_anomaly(indicator="do", hours=hours)
    assert result.ok is False
    assert result.code == "invalid_hours"
    assert result.source_kind == "none"
    assert result.data_version == "none"


@pytest.mark.parametrize("hours", [1, 168])
def test_score_anomaly_does_not_fabricate_evidence(hours: int) -> None:
    result = score_anomaly(indicator="do", hours=hours)
    assert result.ok is False
    assert result.code == "evidence_unavailable"
    assert result.source_kind == "none"
    assert result.data_version == "none"
    assert result.payload == {}


def public_anomaly_summary(
    **overrides: object,
) -> PublicAnomalySummary:
    values: dict[str, object] = {
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
        "method": "rolling_median_mad",
        "direction": "above_baseline",
        "is_anomaly": True,
        "outcome": "anomaly",
    }
    values.update(overrides)
    return PublicAnomalySummary(**values)


def test_screen_public_historical_anomaly_returns_strict_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = public_anomaly_summary()
    calls: list[tuple[str, int, str]] = []

    def screen(indicator: str, hours: int, end_at: str) -> PublicAnomalySummary:
        calls.append((indicator, hours, end_at))
        return expected

    monkeypatch.setattr(tools_module, "screen_public_anomaly", screen)

    result = screen_public_historical_anomaly("nh4", 24, "2025-01-02T00:00:00Z")

    assert calls == [("nh4", 24, "2025-01-02T00:00:00Z")]
    assert result.ok is True
    assert result.code == "ok"
    assert result.source_kind == "public_historical_anomaly"
    assert result.data_version == expected.source_version
    assert set(result.payload) == set(PublicHistoricalAnomaly.model_fields)
    assert "observed_value" not in result.payload
    assert not {"values", "rows", "csv_path", "is_anomaly"} & set(result.payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"values": []},
        {"source_url": "https://example.invalid"},
        {"baseline_end": "2025-01-02T00:05:00Z"},
        {"outcome": "normal", "robust_score": 4.0},
        {
            "outcome": "insufficient_history",
            "method": "rolling_median_mad",
            "robust_score": None,
            "baseline_median": None,
            "baseline_scale": None,
            "direction": "unavailable",
        },
    ],
)
def test_public_historical_anomaly_rejects_forged_or_invalid_semantics(
    payload: dict[str, object],
) -> None:
    base = {
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
        "method": "rolling_median_mad",
        "direction": "above_baseline",
        "outcome": "anomaly",
    }
    base.update(payload)

    with pytest.raises(ValueError):
        PublicHistoricalAnomaly.model_validate(base)


def _public_anomaly_payload(**overrides: object) -> dict[str, object]:
    payload = asdict(public_anomaly_summary(**overrides))
    payload.pop("is_anomaly")
    payload.pop("requires_human_review")
    return payload


@pytest.mark.parametrize(
    "payload",
    [
        _public_anomaly_payload(history_count=47, history_null_count=0),
        _public_anomaly_payload(
            history_count=47,
            history_null_count=0,
            outcome="normal",
            robust_score=1.0,
            is_anomaly=False,
        ),
        _public_anomaly_payload(
            history_count=47,
            history_null_count=0,
            outcome="insufficient_variability",
            method="insufficient_variability",
            direction="unavailable",
            baseline_scale=0.0,
            robust_score=None,
            is_anomaly=False,
        ),
    ],
)
def test_public_historical_anomaly_rejects_screening_outcomes_with_under_48_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="48"):
        PublicHistoricalAnomaly.model_validate(payload)


def test_public_historical_anomaly_enforces_47_48_boundary_and_target_unavailable() -> (
    None
):
    accepted = PublicHistoricalAnomaly.model_validate(
        _public_anomaly_payload(history_count=48, history_null_count=0)
    )
    assert accepted.history_count - accepted.history_null_count == 48

    unavailable = PublicHistoricalAnomaly.model_validate(
        _public_anomaly_payload(
            history_count=0,
            history_null_count=0,
            outcome="target_unavailable",
            method="target_unavailable",
            direction="unavailable",
            baseline_median=None,
            baseline_scale=None,
            robust_score=None,
            is_anomaly=None,
        )
    )
    assert unavailable.outcome == "target_unavailable"


@pytest.mark.parametrize(
    ("indicator", "hours", "end_at"),
    [
        ("nh3_n", 24, "2025-01-02T00:00:00Z"),
        ("nh4", True, "2025-01-02T00:00:00Z"),
        ("nh4", 24, "2025-01-02T00:00:00"),
    ],
)
def test_screen_public_historical_anomaly_rejects_invalid_request_without_data_access(
    monkeypatch: pytest.MonkeyPatch,
    indicator: object,
    hours: object,
    end_at: object,
) -> None:
    monkeypatch.setattr(
        tools_module,
        "screen_public_anomaly",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not read data")),
    )

    result = screen_public_historical_anomaly(indicator, hours, end_at)

    assert result.ok is False
    assert result.code == "invalid_public_request"


@pytest.mark.parametrize("failure", [PublicWindowError("unavailable"), OSError("io")])
def test_screen_public_historical_anomaly_masks_public_evidence_failures(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def raise_failure(*args: object) -> PublicAnomalySummary:
        raise failure

    monkeypatch.setattr(tools_module, "screen_public_anomaly", raise_failure)

    result = screen_public_historical_anomaly("nh4", 24, "2025-01-02T00:00:00Z")

    assert result.ok is False
    assert result.code == "public_evidence_unavailable"
    assert result.source_kind == "none"


@pytest.mark.parametrize(
    "summary",
    [
        public_anomaly_summary(
            target_time="2025-01-02T01:00:00Z",
            baseline_start="2025-01-01T01:00:00Z",
            baseline_end="2025-01-02T01:00:00Z",
        ),
        public_anomaly_summary(baseline_start="2024-12-31T23:00:00Z"),
    ],
)
def test_screen_public_historical_anomaly_binds_summary_window_to_request(
    monkeypatch: pytest.MonkeyPatch,
    summary: PublicAnomalySummary,
) -> None:
    monkeypatch.setattr(tools_module, "screen_public_anomaly", lambda *args: summary)

    result = screen_public_historical_anomaly("nh4", 24, "2025-01-02T00:00:00Z")

    assert result.ok is False
    assert result.code == "public_evidence_unavailable"
