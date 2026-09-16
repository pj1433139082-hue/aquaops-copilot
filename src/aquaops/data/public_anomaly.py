"""Explainable anomaly screening over the fixed approved public history.

This module is deliberately limited to one historical, registered source.  It
does not predict, control equipment, or expose individual measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Literal

from aquaops.data.public_history import (
    PUBLIC_SOURCE_ID,
    PUBLIC_SOURCE_URL,
    PUBLIC_SOURCE_VERSION,
    PublicHistoryError,
    _INDICATOR_UNITS,
    _load_verified_public_observations,
)


DEFAULT_MIN_HISTORY_POINTS = 48
ROBUST_SCORE_THRESHOLD = 3.5
_MAD_NORMALIZATION = 1.4826
_IQR_NORMALIZATION = 1.349
AnomalyMethod = Literal[
    "rolling_median_mad",
    "rolling_median_iqr_fallback",
    "insufficient_variability",
    "insufficient_history",
    "target_unavailable",
]
AnomalyDirection = Literal["above_baseline", "below_baseline", "unavailable"]
AnomalyOutcome = Literal[
    "anomaly",
    "normal",
    "target_unavailable",
    "insufficient_history",
    "insufficient_variability",
]

_MAD_METHOD: AnomalyMethod = "rolling_median_mad"
_IQR_METHOD: AnomalyMethod = "rolling_median_iqr_fallback"


class PublicAnomalyError(ValueError):
    """Raised when public anomaly screening cannot be safely performed."""


@dataclass(frozen=True)
class PublicAnomalySummary:
    """Aggregate-only anomaly-screening result requiring human review."""

    source_id: str
    source_url: str
    source_version: str
    indicator: str
    unit: str
    target_time: str
    baseline_start: str
    baseline_end: str
    history_count: int
    history_null_count: int
    baseline_median: float | None
    baseline_scale: float | None
    robust_score: float | None
    threshold: float
    method: AnomalyMethod
    direction: AnomalyDirection
    is_anomaly: bool | None
    outcome: AnomalyOutcome
    requires_human_review: bool = True


def _format_utc(timestamp: datetime) -> str:
    normalized = timestamp.astimezone(timezone.utc).isoformat()
    return f"{normalized[:-6]}Z"


def _parse_canonical_end_at(end_at: str) -> datetime:
    if not isinstance(end_at, str) or not end_at.endswith("Z"):
        raise PublicAnomalyError("end_at must be canonical UTC Z")
    try:
        timestamp = datetime.fromisoformat(f"{end_at[:-1]}+00:00")
    except ValueError as error:
        raise PublicAnomalyError("end_at must be canonical UTC Z") from error
    if timestamp.tzinfo is None or _format_utc(timestamp) != end_at:
        raise PublicAnomalyError("end_at must be canonical UTC Z")
    return timestamp.astimezone(timezone.utc)


def _validate_query(
    indicator: str,
    baseline_hours: int,
    end_at: str,
    source_id: str,
    min_history_points: int,
) -> tuple[str, datetime]:
    if source_id != PUBLIC_SOURCE_ID:
        raise PublicAnomalyError("unsupported public source")
    if not isinstance(indicator, str) or indicator not in _INDICATOR_UNITS:
        raise PublicAnomalyError("unsupported public indicator")
    if type(baseline_hours) is not int or not 1 <= baseline_hours <= 168:
        raise PublicAnomalyError("baseline_hours must be an integer from 1 to 168")
    if type(min_history_points) is not int or min_history_points < 1:
        raise PublicAnomalyError("min_history_points must be a positive integer")
    return indicator, _parse_canonical_end_at(end_at)


def _require_finite(value: float) -> float:
    if not math.isfinite(value):
        raise PublicAnomalyError("public anomaly statistics overflow")
    return value


def _linear_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise PublicAnomalyError("cannot calculate a percentile without history")
    position = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    interpolated = lower + (upper - lower) * (position - lower_index)
    return _require_finite(interpolated)


def _summary(
    *,
    indicator: str,
    end: datetime,
    baseline_start: datetime,
    history_count: int,
    history_null_count: int,
    baseline_median: float | None,
    baseline_scale: float | None,
    robust_score: float | None,
    method: AnomalyMethod,
    direction: AnomalyDirection,
    is_anomaly: bool | None,
    outcome: AnomalyOutcome,
) -> PublicAnomalySummary:
    return PublicAnomalySummary(
        source_id=PUBLIC_SOURCE_ID,
        source_url=PUBLIC_SOURCE_URL,
        source_version=PUBLIC_SOURCE_VERSION,
        indicator=indicator,
        unit=_INDICATOR_UNITS[indicator],
        target_time=_format_utc(end),
        baseline_start=_format_utc(baseline_start),
        baseline_end=_format_utc(end),
        history_count=history_count,
        history_null_count=history_null_count,
        baseline_median=baseline_median,
        baseline_scale=baseline_scale,
        robust_score=robust_score,
        threshold=ROBUST_SCORE_THRESHOLD,
        method=method,
        direction=direction,
        is_anomaly=is_anomaly,
        outcome=outcome,
    )


def screen_public_anomaly(
    indicator: str,
    baseline_hours: int,
    end_at: str,
    *,
    source_id: str = PUBLIC_SOURCE_ID,
    min_history_points: int = DEFAULT_MIN_HISTORY_POINTS,
) -> PublicAnomalySummary:
    """Screen one exact public-history target against its preceding baseline.

    The baseline is ``[end_at - baseline_hours, end_at)``.  The target is
    therefore never included in the reference distribution.
    """

    selected_indicator, end = _validate_query(
        indicator, baseline_hours, end_at, source_id, min_history_points
    )
    baseline_start = end - timedelta(hours=baseline_hours)
    try:
        observations = _load_verified_public_observations()
    except PublicHistoryError as error:
        raise PublicAnomalyError(str(error)) from None

    history_values: list[float] = []
    history_count = 0
    history_null_count = 0
    target_records: list[float | None] = []
    for observation in observations:
        value = observation.values[selected_indicator]
        if baseline_start <= observation.timestamp < end:
            history_count += 1
            if value is None:
                history_null_count += 1
                continue
            history_values.append(_require_finite(value))
        elif observation.timestamp == end:
            target_records.append(value)

    if len(target_records) != 1 or target_records[0] is None:
        return _summary(
            indicator=selected_indicator,
            end=end,
            baseline_start=baseline_start,
            history_count=history_count,
            history_null_count=history_null_count,
            baseline_median=None,
            baseline_scale=None,
            robust_score=None,
            method="target_unavailable",
            direction="unavailable",
            is_anomaly=None,
            outcome="target_unavailable",
        )

    if len(history_values) < min_history_points:
        return _summary(
            indicator=selected_indicator,
            end=end,
            baseline_start=baseline_start,
            history_count=history_count,
            history_null_count=history_null_count,
            baseline_median=None,
            baseline_scale=None,
            robust_score=None,
            method="insufficient_history",
            direction="unavailable",
            is_anomaly=None,
            outcome="insufficient_history",
        )

    baseline_value = _require_finite(float(median(history_values)))
    deviations = [abs(value - baseline_value) for value in history_values]
    mad = _require_finite(float(median(deviations)))
    method = _MAD_METHOD
    scale = _require_finite(_MAD_NORMALIZATION * mad)
    if mad == 0.0:
        sorted_history = sorted(history_values)
        first_quartile = _linear_percentile(sorted_history, 0.25)
        third_quartile = _linear_percentile(sorted_history, 0.75)
        iqr = _require_finite(third_quartile - first_quartile)
        scale = _require_finite(iqr / _IQR_NORMALIZATION)
        method = _IQR_METHOD

    if scale == 0.0:
        return _summary(
            indicator=selected_indicator,
            end=end,
            baseline_start=baseline_start,
            history_count=history_count,
            history_null_count=history_null_count,
            baseline_median=baseline_value,
            baseline_scale=scale,
            robust_score=None,
            method="insufficient_variability",
            direction="unavailable",
            is_anomaly=False,
            outcome="insufficient_variability",
        )

    target_value = _require_finite(target_records[0])
    robust_score = _require_finite((target_value - baseline_value) / scale)
    if robust_score > 0:
        direction: AnomalyDirection = "above_baseline"
    elif robust_score < 0:
        direction = "below_baseline"
    else:
        direction = "unavailable"
    return _summary(
        indicator=selected_indicator,
        end=end,
        baseline_start=baseline_start,
        history_count=history_count,
        history_null_count=history_null_count,
        baseline_median=baseline_value,
        baseline_scale=scale,
        robust_score=robust_score,
        method=method,
        direction=direction,
        is_anomaly=abs(robust_score) >= ROBUST_SCORE_THRESHOLD,
        outcome=(
            "anomaly" if abs(robust_score) >= ROBUST_SCORE_THRESHOLD else "normal"
        ),
    )
