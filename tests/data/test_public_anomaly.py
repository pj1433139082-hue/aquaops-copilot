from dataclasses import asdict
from datetime import datetime, timezone
import inspect
from types import MappingProxyType

import pytest

import aquaops.data.public_anomaly as public_anomaly
from aquaops.data.public_history import PublicObservation


def observation(hour: int, nh4: float | None) -> PublicObservation:
    return PublicObservation(
        timestamp=datetime(2025, 1, 1, hour, tzinfo=timezone.utc),
        values=MappingProxyType({"nh4": nh4, "cond": 100.0, "q": 10.0}),
    )


def install_history(
    monkeypatch: pytest.MonkeyPatch, observations: tuple[PublicObservation, ...]
) -> None:
    monkeypatch.setattr(
        public_anomaly, "_load_verified_public_observations", lambda: observations
    )


def screen(
    monkeypatch: pytest.MonkeyPatch, observations: tuple[PublicObservation, ...]
):
    install_history(monkeypatch, observations)
    return public_anomaly.screen_public_anomaly(
        "nh4",
        baseline_hours=4,
        end_at="2025-01-01T04:00:00Z",
        min_history_points=4,
    )


def test_screens_normal_target_with_mad_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, 10.0),
        ),
    )

    assert result.outcome == "normal"
    assert result.method == "rolling_median_mad"
    assert result.is_anomaly is False
    assert result.direction == "unavailable"
    assert result.baseline_median == 10.0
    assert result.baseline_scale == pytest.approx(0.7413)
    assert result.robust_score == 0.0
    assert result.threshold == 3.5
    assert result.requires_human_review is True


def test_flags_high_target_with_robust_mad_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, 20.0),
        ),
    )

    assert result.outcome == "anomaly"
    assert result.method == "rolling_median_mad"
    assert result.is_anomaly is True
    assert result.direction == "above_baseline"
    assert result.robust_score == pytest.approx(10.0 / (0.5 * 1.4826))


def test_reports_below_baseline_direction_for_a_normal_low_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, 9.0),
        ),
    )

    assert result.outcome == "normal"
    assert result.direction == "below_baseline"


def test_flags_anomaly_when_robust_score_equals_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(public_anomaly, "_MAD_NORMALIZATION", 1.0)

    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, 11.75),
        ),
    )

    assert result.robust_score == 3.5
    assert result.outcome == "anomaly"
    assert result.is_anomaly is True


def test_flags_anomaly_when_negative_robust_score_equals_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(public_anomaly, "_MAD_NORMALIZATION", 1.0)

    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, 8.25),
        ),
    )

    assert result.robust_score == -3.5
    assert result.outcome == "anomaly"
    assert result.is_anomaly is True
    assert result.direction == "below_baseline"


def test_excludes_target_measurement_from_baseline_to_prevent_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, 100.0),
        ),
    )

    assert result.history_count == 4
    assert result.history_null_count == 0
    assert result.baseline_median == 10.0
    assert result.is_anomaly is True


@pytest.mark.parametrize(
    ("target", "expected_score", "expected_outcome"),
    [
        (1.25, 1.349, "normal"),
        (1.75, 4.047, "anomaly"),
        (1.875, 4.7215, "anomaly"),
        (8.0, 37.772, "anomaly"),
    ],
)
def test_uses_normalized_iqr_fallback_when_mad_is_zero(
    monkeypatch: pytest.MonkeyPatch,
    target: float,
    expected_score: float,
    expected_outcome: str,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 1.0),
            observation(1, 1.0),
            observation(2, 1.0),
            observation(3, 2.0),
            observation(4, target),
        ),
    )

    assert result.outcome == expected_outcome
    assert result.method == "rolling_median_iqr_fallback"
    assert result.direction == "above_baseline"
    assert result.baseline_scale == pytest.approx(0.25 / 1.349)
    assert result.robust_score == pytest.approx(expected_score)
    assert result.is_anomaly is (expected_outcome == "anomaly")


def test_returns_target_unavailable_for_non_exact_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(5, 20.0),
        ),
    )

    assert result.outcome == "target_unavailable"
    assert result.is_anomaly is None
    assert result.method == "target_unavailable"
    assert result.direction == "unavailable"
    assert result.robust_score is None


def test_returns_target_unavailable_for_null_target_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, None),
        ),
    )

    assert result.outcome == "target_unavailable"
    assert result.method == "target_unavailable"
    assert result.is_anomaly is None
    assert result.direction == "unavailable"


def test_returns_target_unavailable_for_multiple_exact_target_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, 20.0),
            observation(4, None),
        ),
    )

    assert result.outcome == "target_unavailable"
    assert result.method == "target_unavailable"
    assert result.direction == "unavailable"
    assert result.is_anomaly is None
    assert result.robust_score is None
    assert result.baseline_median is None
    assert result.baseline_scale is None


def test_returns_insufficient_history_when_baseline_has_too_few_numeric_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, None),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, 20.0),
        ),
    )

    assert result.outcome == "insufficient_history"
    assert result.method == "insufficient_history"
    assert result.history_count == 4
    assert result.history_null_count == 1
    assert result.is_anomaly is None
    assert result.baseline_median is None
    assert result.direction == "unavailable"


def test_returns_insufficient_variability_when_mad_and_iqr_are_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 1.0),
            observation(1, 1.0),
            observation(2, 1.0),
            observation(3, 1.0),
            observation(4, 8.0),
        ),
    )

    assert result.outcome == "insufficient_variability"
    assert result.method == "insufficient_variability"
    assert result.baseline_scale == 0.0
    assert result.is_anomaly is False
    assert result.robust_score is None
    assert result.direction == "unavailable"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"indicator": "temperature"}, "unsupported public indicator"),
        ({"source_id": "private-plant"}, "unsupported public source"),
        ({"baseline_hours": 0}, "baseline_hours must be an integer from 1 to 168"),
        ({"baseline_hours": True}, "baseline_hours must be an integer from 1 to 168"),
        ({"end_at": "2025-01-01T04:00:00+00:00"}, "end_at must be canonical UTC Z"),
        ({"end_at": "2025-01-01T04:00:00"}, "end_at must be canonical UTC Z"),
        ({"min_history_points": 0}, "min_history_points must be a positive integer"),
    ],
)
def test_rejects_invalid_source_indicator_and_query_inputs(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object], message: str
) -> None:
    install_history(monkeypatch, tuple())
    arguments: dict[str, object] = {
        "indicator": "nh4",
        "baseline_hours": 4,
        "end_at": "2025-01-01T04:00:00Z",
        "min_history_points": 4,
    }
    arguments.update(kwargs)

    with pytest.raises(public_anomaly.PublicAnomalyError, match=message):
        public_anomaly.screen_public_anomaly(**arguments)  # type: ignore[arg-type]


def test_summary_has_no_raw_observation_or_storage_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = screen(
        monkeypatch,
        (
            observation(0, 9.0),
            observation(1, 10.0),
            observation(2, 10.0),
            observation(3, 11.0),
            observation(4, 20.0),
        ),
    )

    serialized = asdict(result)
    forbidden_fields = {
        "value",
        "values",
        "observed_value",
        "rows",
        "samples",
        "csv_path",
        "raw_data",
        "scale",
        "history_point_count",
        "history_total_count",
        "status",
    }
    assert not forbidden_fields & set(serialized)
    assert 20.0 not in serialized.values()
    assert "PublicObservation" not in repr(result)


def test_data_api_does_not_accept_a_caller_controlled_csv_path() -> None:
    parameters = inspect.signature(public_anomaly.screen_public_anomaly).parameters

    assert "csv_path" not in parameters
    assert "path" not in parameters
