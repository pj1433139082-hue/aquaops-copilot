"""Strict versioned contracts for enterprise forecast delivery."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SAFE_METRIC = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_REVISION = re.compile(r"^[0-9a-f]{7,64}$")
_TIMEZONE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_+.-]*(/[A-Za-z0-9_+.-]+)*$")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
_SAFE_UNIT = re.compile(r"^[A-Za-z0-9µμ°%][A-Za-z0-9µμ°%³²/(). _-]{0,31}$")


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("forecast timestamps must use UTC")
    return value


def _require_safe_token(value: str) -> str:
    if _SAFE_TOKEN.fullmatch(value) is None:
        raise ValueError("invalid forecast contract token")
    return value


def _require_sha256(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("invalid forecast contract hash")
    return value


def _require_finite(value: float | None) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError("forecast contract values must be finite")
    return value


class _ForecastContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )


class ForecastPointV1(_ForecastContractModel):
    schema_version: Literal[1]
    forecast_run_id: str
    plant_alias: str
    metric_id: str
    unit: str
    origin_time_utc: datetime
    target_time_utc: datetime
    horizon_hours: int = Field(ge=1, le=720)
    point_estimate: float | None
    lower_bound: float | None
    upper_bound: float | None
    interval_method: Literal[
        "none",
        "heuristic",
        "split_conformal",
        "rolling_conformal",
        "quantile",
    ]
    interval_level: float | None
    status: Literal[
        "ready",
        "insufficient_history",
        "irregular_cadence",
        "data_quality_blocked",
        "model_unavailable",
        "stale",
    ]
    data_quality_flag: Literal["pass", "warn", "fail"]
    model_name: str
    model_version: str
    feature_version: str
    training_cutoff_utc: datetime
    data_snapshot_sha256: str
    created_at_utc: datetime

    @field_validator(
        "origin_time_utc",
        "target_time_utc",
        "training_cutoff_utc",
        "created_at_utc",
    )
    @classmethod
    def validate_utc_timestamps(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @field_validator(
        "forecast_run_id",
        "plant_alias",
        "model_name",
        "model_version",
        "feature_version",
    )
    @classmethod
    def validate_tokens(cls, value: str) -> str:
        return _require_safe_token(value)

    @field_validator("metric_id")
    @classmethod
    def validate_metric_id(cls, value: str) -> str:
        if _SAFE_METRIC.fullmatch(value) is None:
            raise ValueError("invalid forecast metric identifier")
        return value

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str) -> str:
        if (
            not value
            or len(value) > 32
            or _CONTROL_CHARACTER.search(value)
            or _SAFE_UNIT.fullmatch(value) is None
        ):
            raise ValueError("invalid forecast unit")
        return value

    @field_validator("data_snapshot_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _require_sha256(value)

    @field_validator("point_estimate", "lower_bound", "upper_bound", "interval_level")
    @classmethod
    def validate_finite_values(cls, value: float | None) -> float | None:
        return _require_finite(value)

    @model_validator(mode="after")
    def validate_delivery_semantics(self) -> Self:
        if self.target_time_utc - self.origin_time_utc != timedelta(
            hours=self.horizon_hours
        ):
            raise ValueError("forecast horizon does not match target time")
        if self.training_cutoff_utc > self.origin_time_utc:
            raise ValueError("forecast training cutoff is after its origin")
        if not self.origin_time_utc <= self.created_at_utc < self.target_time_utc:
            raise ValueError("forecast creation time is outside its delivery window")

        values = (self.lower_bound, self.point_estimate, self.upper_bound)
        if self.status != "ready":
            if any(value is not None for value in values):
                raise ValueError("unavailable forecast cannot contain estimates")
            if self.interval_method != "none" or self.interval_level is not None:
                raise ValueError("unavailable forecast cannot claim an interval")
        else:
            if self.point_estimate is None:
                raise ValueError("ready forecast requires a point estimate")
            bounds_present = (
                self.lower_bound is not None and self.upper_bound is not None
            )
            bounds_absent = self.lower_bound is None and self.upper_bound is None
            if not (bounds_present or bounds_absent):
                raise ValueError("forecast bounds must be complete")
            if bounds_present:
                if not self.lower_bound <= self.point_estimate <= self.upper_bound:
                    raise ValueError("forecast bounds are not ordered")
                if self.interval_method == "none":
                    raise ValueError("forecast bounds require an interval method")
                if self.interval_method == "heuristic":
                    if self.interval_level is not None:
                        raise ValueError(
                            "heuristic range cannot claim calibrated coverage"
                        )
                elif self.interval_level is None or not 0.0 < self.interval_level < 1.0:
                    raise ValueError("calibrated interval requires a valid level")
            elif self.interval_method != "none" or self.interval_level is not None:
                raise ValueError("interval metadata requires forecast bounds")

        if (self.data_quality_flag == "fail") != (
            self.status == "data_quality_blocked"
        ):
            raise ValueError("forecast status and data quality are inconsistent")
        return self


class ForecastRunManifestV1(_ForecastContractModel):
    schema_version: Literal[1]
    forecast_run_id: str
    generated_at_utc: datetime
    model_kind: Literal["baseline", "learned"]
    model_name: str
    model_version: str
    feature_version: str
    model_artifact_sha256: str | None
    code_revision: str
    data_snapshot_sha256: str
    feature_spec_sha256: str
    training_start_utc: datetime
    training_cutoff_utc: datetime
    calibration_start_utc: datetime | None
    calibration_end_utc: datetime | None
    forecast_horizons_hours: tuple[int, ...]
    cadence_seconds: int = Field(ge=1, le=86_400)
    business_timezone: str
    validation_status: Literal["not_evaluated", "passed", "warning", "failed"]
    drift_status: Literal["not_evaluated", "stable", "warning", "critical"]
    forecast_points_sha256: str

    @field_validator(
        "forecast_run_id", "model_name", "model_version", "feature_version"
    )
    @classmethod
    def validate_tokens(cls, value: str) -> str:
        return _require_safe_token(value)

    @field_validator(
        "generated_at_utc",
        "training_start_utc",
        "training_cutoff_utc",
        "calibration_start_utc",
        "calibration_end_utc",
    )
    @classmethod
    def validate_utc_timestamps(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_utc(value)

    @field_validator(
        "model_artifact_sha256",
        "data_snapshot_sha256",
        "feature_spec_sha256",
        "forecast_points_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str | None) -> str | None:
        return None if value is None else _require_sha256(value)

    @field_validator("code_revision")
    @classmethod
    def validate_code_revision(cls, value: str) -> str:
        if _CODE_REVISION.fullmatch(value) is None:
            raise ValueError("invalid forecast code revision")
        return value

    @field_validator("business_timezone")
    @classmethod
    def validate_business_timezone(cls, value: str) -> str:
        if len(value) > 64 or _TIMEZONE_NAME.fullmatch(value) is None:
            raise ValueError("invalid business timezone")
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("unknown business timezone") from None
        return value

    @field_validator("forecast_horizons_hours")
    @classmethod
    def validate_horizons(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if (
            not value
            or value != tuple(sorted(set(value)))
            or any(type(item) is not int or item < 1 or item > 720 for item in value)
        ):
            raise ValueError("invalid forecast horizons")
        return value

    @model_validator(mode="after")
    def validate_run_lineage(self) -> Self:
        if (
            not self.training_start_utc
            < self.training_cutoff_utc
            <= self.generated_at_utc
        ):
            raise ValueError("invalid forecast training window")
        calibration_values = (self.calibration_start_utc, self.calibration_end_utc)
        if (calibration_values[0] is None) != (calibration_values[1] is None):
            raise ValueError("incomplete forecast calibration window")
        if calibration_values[0] is not None and not (
            self.training_cutoff_utc
            < calibration_values[0]
            <= calibration_values[1]
            <= self.generated_at_utc
        ):
            raise ValueError("invalid forecast calibration window")
        if (self.model_kind == "learned") != (self.model_artifact_sha256 is not None):
            raise ValueError("forecast model kind and artifact are inconsistent")
        return self


class ForecastMetricV1(_ForecastContractModel):
    schema_version: Literal[1]
    forecast_run_id: str
    plant_alias: str
    metric_id: str
    horizon_hours: int = Field(ge=1, le=720)
    evaluation_scope: Literal[
        "timecv", "calibration", "holdout", "production_monitoring"
    ]
    fold_id: str | None
    condition: str
    sample_count: int = Field(ge=1)
    mae: float
    rmse: float
    r2: float | None
    wape: float | None
    smape: float | None
    persistence_skill: float | None
    interval_coverage: float | None
    interval_mean_width: float | None
    evaluated_at_utc: datetime

    @field_validator("forecast_run_id", "plant_alias", "fold_id", "condition")
    @classmethod
    def validate_tokens(cls, value: str | None) -> str | None:
        return None if value is None else _require_safe_token(value)

    @field_validator("metric_id")
    @classmethod
    def validate_metric_id(cls, value: str) -> str:
        if _SAFE_METRIC.fullmatch(value) is None:
            raise ValueError("invalid forecast metric identifier")
        return value

    @field_validator(
        "mae",
        "rmse",
        "r2",
        "wape",
        "smape",
        "persistence_skill",
        "interval_coverage",
        "interval_mean_width",
    )
    @classmethod
    def validate_finite_metrics(cls, value: float | None) -> float | None:
        return _require_finite(value)

    @field_validator("evaluated_at_utc")
    @classmethod
    def validate_evaluated_at(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def validate_metric_semantics(self) -> Self:
        if self.mae < 0.0 or self.rmse < 0.0 or self.rmse < self.mae:
            raise ValueError("invalid forecast error metrics")
        if self.wape is not None and self.wape < 0.0:
            raise ValueError("invalid forecast WAPE")
        if self.smape is not None and not 0.0 <= self.smape <= 2.0:
            raise ValueError("invalid forecast sMAPE")
        if self.evaluation_scope == "timecv" and self.fold_id is None:
            raise ValueError("TimeCV forecast metric requires a fold")
        if (self.interval_coverage is None) != (self.interval_mean_width is None):
            raise ValueError("incomplete forecast interval metrics")
        if (
            self.interval_coverage is not None
            and not 0.0 <= self.interval_coverage <= 1.0
        ):
            raise ValueError("invalid forecast interval coverage")
        if self.interval_mean_width is not None and self.interval_mean_width < 0.0:
            raise ValueError("invalid forecast interval width")
        return self


class ForecastDeliveryV1(_ForecastContractModel):
    """In-memory gate proving all forecast delivery artifacts share one lineage."""

    manifest: ForecastRunManifestV1
    points: tuple[ForecastPointV1, ...] = Field(min_length=1)
    metrics: tuple[ForecastMetricV1, ...]

    @model_validator(mode="after")
    def validate_cross_artifact_lineage(self) -> Self:
        manifest = self.manifest
        horizon_set = set(manifest.forecast_horizons_hours)
        point_keys: set[tuple[str, str, datetime, datetime]] = set()
        evaluated_keys: set[tuple[str, str, int]] = set()
        horizons_by_forecast: dict[tuple[str, str, datetime], set[int]] = {}
        units_by_metric: dict[tuple[str, str], str] = {}
        uses_conformal_interval = False

        for point in self.points:
            if (
                point.forecast_run_id != manifest.forecast_run_id
                or point.model_name != manifest.model_name
                or point.model_version != manifest.model_version
                or point.feature_version != manifest.feature_version
                or point.data_snapshot_sha256 != manifest.data_snapshot_sha256
                or point.training_cutoff_utc != manifest.training_cutoff_utc
                or point.created_at_utc != manifest.generated_at_utc
                or point.horizon_hours not in horizon_set
            ):
                raise ValueError("forecast point lineage does not match its run")
            point_key = (
                point.plant_alias,
                point.metric_id,
                point.origin_time_utc,
                point.target_time_utc,
            )
            if point_key in point_keys:
                raise ValueError("duplicate forecast point delivery key")
            point_keys.add(point_key)
            evaluated_keys.add(
                (point.plant_alias, point.metric_id, point.horizon_hours)
            )
            forecast_key = (
                point.plant_alias,
                point.metric_id,
                point.origin_time_utc,
            )
            horizons_by_forecast.setdefault(forecast_key, set()).add(
                point.horizon_hours
            )
            metric_key = (point.plant_alias, point.metric_id)
            expected_unit = units_by_metric.setdefault(metric_key, point.unit)
            if point.unit != expected_unit:
                raise ValueError("forecast unit drift within one run")
            if point.interval_method in {"split_conformal", "rolling_conformal"}:
                uses_conformal_interval = True
                if (
                    manifest.calibration_end_utc is None
                    or manifest.calibration_end_utc > point.origin_time_utc
                ):
                    raise ValueError(
                        "forecast calibration uses data after the forecast origin"
                    )
            if point.model_name == "persistence" and point.status == "ready":
                if (
                    point.interval_method != "heuristic"
                    or point.interval_level is not None
                ):
                    raise ValueError("persistence forecast must remain heuristic")

        if any(
            delivered_horizons != horizon_set
            for delivered_horizons in horizons_by_forecast.values()
        ):
            raise ValueError("declared forecast horizons are not fully delivered")
        has_calibration_window = (
            manifest.calibration_start_utc is not None
            and manifest.calibration_end_utc is not None
        )
        if uses_conformal_interval and not has_calibration_window:
            raise ValueError("conformal forecast requires calibration lineage")

        if (manifest.validation_status == "not_evaluated") != (not self.metrics):
            raise ValueError("forecast validation status and metrics are inconsistent")

        metric_keys: set[tuple[str, str, int, str, str | None, str]] = set()
        for metric in self.metrics:
            if (
                metric.forecast_run_id != manifest.forecast_run_id
                or metric.horizon_hours not in horizon_set
                or (metric.plant_alias, metric.metric_id, metric.horizon_hours)
                not in evaluated_keys
                or metric.evaluated_at_utc > manifest.generated_at_utc
            ):
                raise ValueError("forecast metric lineage does not match its run")
            metric_key = (
                metric.plant_alias,
                metric.metric_id,
                metric.horizon_hours,
                metric.evaluation_scope,
                metric.fold_id,
                metric.condition,
            )
            if metric_key in metric_keys:
                raise ValueError("duplicate forecast metric delivery key")
            metric_keys.add(metric_key)
        return self


def forecast_point_arrow_schema_v1() -> pa.Schema:
    return pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("forecast_run_id", pa.string(), nullable=False),
            pa.field("plant_alias", pa.string(), nullable=False),
            pa.field("metric_id", pa.string(), nullable=False),
            pa.field("unit", pa.string(), nullable=False),
            pa.field("origin_time_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("target_time_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("horizon_hours", pa.int32(), nullable=False),
            pa.field("point_estimate", pa.float64()),
            pa.field("lower_bound", pa.float64()),
            pa.field("upper_bound", pa.float64()),
            pa.field("interval_method", pa.string(), nullable=False),
            pa.field("interval_level", pa.float64()),
            pa.field("status", pa.string(), nullable=False),
            pa.field("data_quality_flag", pa.string(), nullable=False),
            pa.field("model_name", pa.string(), nullable=False),
            pa.field("model_version", pa.string(), nullable=False),
            pa.field("feature_version", pa.string(), nullable=False),
            pa.field(
                "training_cutoff_utc", pa.timestamp("us", tz="UTC"), nullable=False
            ),
            pa.field("data_snapshot_sha256", pa.string(), nullable=False),
            pa.field("created_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        ],
        metadata={b"aquaops.schema": b"forecast-point-v1"},
    )


def forecast_metric_arrow_schema_v1() -> pa.Schema:
    return pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("forecast_run_id", pa.string(), nullable=False),
            pa.field("plant_alias", pa.string(), nullable=False),
            pa.field("metric_id", pa.string(), nullable=False),
            pa.field("horizon_hours", pa.int32(), nullable=False),
            pa.field("evaluation_scope", pa.string(), nullable=False),
            pa.field("fold_id", pa.string()),
            pa.field("condition", pa.string(), nullable=False),
            pa.field("sample_count", pa.int64(), nullable=False),
            pa.field("mae", pa.float64(), nullable=False),
            pa.field("rmse", pa.float64(), nullable=False),
            pa.field("r2", pa.float64()),
            pa.field("wape", pa.float64()),
            pa.field("smape", pa.float64()),
            pa.field("persistence_skill", pa.float64()),
            pa.field("interval_coverage", pa.float64()),
            pa.field("interval_mean_width", pa.float64()),
            pa.field("evaluated_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        ],
        metadata={b"aquaops.schema": b"forecast-metric-v1"},
    )
