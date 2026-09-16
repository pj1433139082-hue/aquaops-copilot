from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aquaops.data.public_anomaly import screen_public_anomaly
from aquaops.data.public_window import summarize_public_window

ALLOWED_INDICATORS: frozenset[str] = frozenset({"nh3_n", "cod", "tn", "do"})
_PUBLIC_INDICATOR_UNITS = {"nh4": "mg/L", "cond": "µS/cm", "q": "L/s"}
_MINIMUM_ANOMALY_HISTORY_POINTS = 48


class ToolResult(BaseModel):
    ok: bool
    code: str
    source_kind: str
    data_version: str
    payload: dict[str, object] = Field(default_factory=dict)


class PublicHistoricalAggregate(BaseModel):
    """Strict boundary for the sole approved public-history aggregate."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: Literal["co-udlabs-wwtp-lpicm-2025"]
    source_url: Literal["https://zenodo.org/records/15285089"]
    source_version: Literal["v1.0.0 (2025-04-26)"]
    indicator: Literal["nh4", "cond", "q"]
    unit: str
    start: str
    end: str
    row_count: int
    null_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None

    @field_validator("start", "end")
    @classmethod
    def validate_canonical_utc_z(cls, value: str) -> str:
        if not _is_canonical_utc_z(value):
            raise ValueError("timestamp must be canonical UTC Z")
        return value

    @field_validator("minimum", "maximum", "mean")
    @classmethod
    def validate_finite_statistic(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("statistic must be finite")
        return value

    @model_validator(mode="after")
    def validate_aggregate_invariants(self) -> "PublicHistoricalAggregate":
        if self.unit != _PUBLIC_INDICATOR_UNITS[self.indicator]:
            raise ValueError("indicator unit does not match approved public schema")
        start = _parse_canonical_utc_z(self.start)
        end = _parse_canonical_utc_z(self.end)
        if start is None or end is None or start >= end:
            raise ValueError("public historical window must increase")
        if self.row_count < 1 or not 0 <= self.null_count <= self.row_count:
            raise ValueError("public historical counts are invalid")

        valid_count = self.row_count - self.null_count
        statistics = (self.minimum, self.mean, self.maximum)
        if valid_count == 0:
            if any(value is not None for value in statistics):
                raise ValueError("all-null windows must not contain statistics")
        elif any(value is None for value in statistics):
            raise ValueError("non-empty windows require all statistics")
        elif not self.minimum <= self.mean <= self.maximum:
            raise ValueError("public historical statistics are not ordered")
        return self


class PublicHistoricalAnomaly(BaseModel):
    """Strict boundary for aggregate-only public historical anomaly screening."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: Literal["co-udlabs-wwtp-lpicm-2025"]
    source_url: Literal["https://zenodo.org/records/15285089"]
    source_version: Literal["v1.0.0 (2025-04-26)"]
    indicator: Literal["nh4", "cond", "q"]
    unit: str
    target_time: str
    baseline_start: str
    baseline_end: str
    history_count: int
    history_null_count: int
    baseline_median: float | None
    baseline_scale: float | None
    robust_score: float | None
    threshold: Literal[3.5]
    direction: Literal["above_baseline", "below_baseline", "unavailable"]
    method: Literal[
        "rolling_median_mad",
        "rolling_median_iqr_fallback",
        "insufficient_variability",
        "insufficient_history",
        "target_unavailable",
    ]
    outcome: Literal[
        "anomaly",
        "normal",
        "target_unavailable",
        "insufficient_history",
        "insufficient_variability",
    ]

    @field_validator("target_time", "baseline_start", "baseline_end")
    @classmethod
    def validate_canonical_anomaly_timestamp(cls, value: str) -> str:
        if not _is_canonical_utc_z(value):
            raise ValueError("timestamp must be canonical UTC Z")
        return value

    @field_validator("baseline_median", "baseline_scale", "robust_score")
    @classmethod
    def validate_finite_anomaly_statistic(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("anomaly statistic must be finite")
        return value

    @model_validator(mode="after")
    def validate_anomaly_invariants(self) -> "PublicHistoricalAnomaly":
        if self.unit != _PUBLIC_INDICATOR_UNITS[self.indicator]:
            raise ValueError("indicator unit does not match approved public schema")
        target = _parse_canonical_utc_z(self.target_time)
        baseline_start = _parse_canonical_utc_z(self.baseline_start)
        baseline_end = _parse_canonical_utc_z(self.baseline_end)
        if (
            target is None
            or baseline_start is None
            or baseline_end is None
            or baseline_start >= target
            or baseline_end != target
        ):
            raise ValueError("public anomaly baseline timing is invalid")
        if (
            self.history_count < 0
            or not 0 <= self.history_null_count <= self.history_count
        ):
            raise ValueError("public anomaly counts are invalid")
        valid_history_count = self.history_count - self.history_null_count
        if self.outcome in {"anomaly", "normal", "insufficient_variability"}:
            if valid_history_count < _MINIMUM_ANOMALY_HISTORY_POINTS:
                raise ValueError(
                    "screening outcomes require at least 48 valid history points"
                )
        elif (
            self.outcome == "insufficient_history"
            and valid_history_count >= _MINIMUM_ANOMALY_HISTORY_POINTS
        ):
            raise ValueError(
                "insufficient history must contain fewer than 48 valid history points"
            )

        if self.outcome in {"anomaly", "normal"}:
            if self.method not in {
                "rolling_median_mad",
                "rolling_median_iqr_fallback",
            }:
                raise ValueError("screened anomaly requires a robust method")
            if (
                self.baseline_median is None
                or self.baseline_scale is None
                or self.robust_score is None
                or self.baseline_scale <= 0
            ):
                raise ValueError("screened anomaly requires robust statistics")
            if self.robust_score > 0 and self.direction != "above_baseline":
                raise ValueError("anomaly direction does not match robust score")
            if self.robust_score < 0 and self.direction != "below_baseline":
                raise ValueError("anomaly direction does not match robust score")
            if self.robust_score == 0 and self.direction != "unavailable":
                raise ValueError("zero robust score must have unavailable direction")
            is_anomaly = abs(self.robust_score) >= self.threshold
            if (self.outcome == "anomaly") != is_anomaly:
                raise ValueError("anomaly outcome does not match robust score")
            return self

        if self.outcome == "insufficient_variability":
            if (
                self.method != "insufficient_variability"
                or self.direction != "unavailable"
                or self.baseline_median is None
                or self.baseline_scale != 0
                or self.robust_score is not None
            ):
                raise ValueError("insufficient variability semantics are invalid")
            return self

        required_method = {
            "target_unavailable": "target_unavailable",
            "insufficient_history": "insufficient_history",
        }[self.outcome]
        if (
            self.method != required_method
            or self.direction != "unavailable"
            or self.baseline_median is not None
            or self.baseline_scale is not None
            or self.robust_score is not None
        ):
            raise ValueError("insufficient anomaly evidence semantics are invalid")
        return self


def _parse_canonical_utc_z(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return None
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return parsed if canonical == value else None


def _is_canonical_utc_z(value: object) -> bool:
    return _parse_canonical_utc_z(value) is not None


def query_public_historical_window(
    indicator: object, hours: object, end_at: object
) -> ToolResult:
    """Return one aggregate from the fixed public historical source only."""

    if (
        not isinstance(indicator, str)
        or indicator not in {"nh4", "cond", "q"}
        or type(hours) is not int
        or not 1 <= hours <= 168
        or not _is_canonical_utc_z(end_at)
    ):
        return ToolResult(
            ok=False,
            code="invalid_public_request",
            source_kind="none",
            data_version="none",
        )

    try:
        summary = summarize_public_window(indicator, hours, end_at)
        aggregate = PublicHistoricalAggregate.model_validate(asdict(summary))
    except Exception:
        return ToolResult(
            ok=False,
            code="public_evidence_unavailable",
            source_kind="none",
            data_version="none",
        )

    return ToolResult(
        ok=True,
        code="ok",
        source_kind="public_historical",
        data_version=aggregate.source_version,
        payload=aggregate.model_dump(),
    )


def screen_public_historical_anomaly(
    indicator: object, baseline_hours: object, end_at: object
) -> ToolResult:
    """Return a validated, aggregate-only public historical anomaly screen."""

    if (
        not isinstance(indicator, str)
        or indicator not in {"nh4", "cond", "q"}
        or type(baseline_hours) is not int
        or not 1 <= baseline_hours <= 168
        or not _is_canonical_utc_z(end_at)
    ):
        return ToolResult(
            ok=False,
            code="invalid_public_request",
            source_kind="none",
            data_version="none",
        )

    try:
        summary = screen_public_anomaly(indicator, baseline_hours, end_at)
        summary_values = asdict(summary)
        anomaly = PublicHistoricalAnomaly.model_validate(
            {
                field_name: summary_values[field_name]
                for field_name in PublicHistoricalAnomaly.model_fields
            }
        )
        requested_end = _parse_canonical_utc_z(end_at)
        if requested_end is None:
            raise ValueError("invalid public anomaly end time")
        expected_baseline_start = (
            (requested_end - timedelta(hours=baseline_hours))
            .isoformat()
            .replace("+00:00", "Z")
        )
        if (
            anomaly.target_time != end_at
            or anomaly.baseline_end != end_at
            or anomaly.baseline_start != expected_baseline_start
        ):
            raise ValueError("public anomaly summary does not match request window")
    except Exception:
        return ToolResult(
            ok=False,
            code="public_evidence_unavailable",
            source_kind="none",
            data_version="none",
        )

    return ToolResult(
        ok=True,
        code="ok",
        source_kind="public_historical_anomaly",
        data_version=anomaly.source_version,
        payload=anomaly.model_dump(),
    )


def query_signal_window(indicator: object, hours: object) -> ToolResult:
    if not isinstance(indicator, str) or indicator not in ALLOWED_INDICATORS:
        return ToolResult(
            ok=False,
            code="unsupported_indicator",
            source_kind="none",
            data_version="none",
        )
    if type(hours) is not int or not 1 <= hours <= 168:
        return ToolResult(
            ok=False,
            code="invalid_hours",
            source_kind="none",
            data_version="none",
        )
    return ToolResult(
        ok=True,
        code="ok",
        source_kind="public_or_synthetic",
        data_version="demo-v1",
        payload={"indicator": indicator, "hours": hours, "values": []},
    )


def retrieve_knowledge(query: object) -> ToolResult:
    if not isinstance(query, str) or len(query.strip()) < 3:
        return ToolResult(
            ok=False,
            code="invalid_query",
            source_kind="none",
            data_version="none",
        )
    return ToolResult(
        ok=False,
        code="evidence_unavailable",
        source_kind="none",
        data_version="none",
    )


def score_anomaly(indicator: object, hours: object) -> ToolResult:
    if not isinstance(indicator, str) or indicator not in ALLOWED_INDICATORS:
        return ToolResult(
            ok=False,
            code="unsupported_indicator",
            source_kind="none",
            data_version="none",
        )
    if type(hours) is not int or not 1 <= hours <= 168:
        return ToolResult(
            ok=False,
            code="invalid_hours",
            source_kind="none",
            data_version="none",
        )
    return ToolResult(
        ok=False,
        code="evidence_unavailable",
        source_kind="none",
        data_version="none",
    )
