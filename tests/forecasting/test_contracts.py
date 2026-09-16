"""Synthetic contract tests for enterprise forecast delivery artifacts."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pytest
from pydantic import BaseModel, ValidationError

import aquaops.forecasting as forecasting
from aquaops.forecasting import contracts


UTC = timezone.utc
ORIGIN = datetime(2026, 8, 12, 0, 0, tzinfo=UTC)


def _forecast_point_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "forecast_run_id": "run-20260812-001",
        "plant_alias": "plant_demo_001",
        "metric_id": "metric_t001_column_001",
        "unit": "mg/L",
        "origin_time_utc": ORIGIN,
        "target_time_utc": ORIGIN + timedelta(hours=24),
        "horizon_hours": 24,
        "point_estimate": 2.0,
        "lower_bound": 1.0,
        "upper_bound": 3.0,
        "interval_method": "heuristic",
        "interval_level": None,
        "status": "ready",
        "data_quality_flag": "pass",
        "model_name": "persistence",
        "model_version": "baseline-v1",
        "feature_version": "latest-observation-v1",
        "training_cutoff_utc": ORIGIN,
        "data_snapshot_sha256": "a" * 64,
        "created_at_utc": ORIGIN + timedelta(seconds=1),
    }


def _run_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "forecast_run_id": "run-20260812-001",
        "generated_at_utc": ORIGIN + timedelta(seconds=1),
        "model_kind": "baseline",
        "model_name": "persistence",
        "model_version": "baseline-v1",
        "feature_version": "latest-observation-v1",
        "model_artifact_sha256": None,
        "code_revision": "b" * 40,
        "data_snapshot_sha256": "a" * 64,
        "feature_spec_sha256": "c" * 64,
        "training_start_utc": ORIGIN - timedelta(days=365),
        "training_cutoff_utc": ORIGIN,
        "calibration_start_utc": None,
        "calibration_end_utc": None,
        "forecast_horizons_hours": (24,),
        "cadence_seconds": 3600,
        "business_timezone": "Asia/Shanghai",
        "validation_status": "not_evaluated",
        "drift_status": "not_evaluated",
        "forecast_points_sha256": "d" * 64,
    }


def _forecast_metric_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "forecast_run_id": "run-20260812-001",
        "plant_alias": "plant_demo_001",
        "metric_id": "metric_t001_column_001",
        "horizon_hours": 24,
        "evaluation_scope": "timecv",
        "fold_id": "fold-01",
        "condition": "all",
        "sample_count": 100,
        "mae": 0.2,
        "rmse": 0.3,
        "r2": 0.8,
        "wape": 0.1,
        "smape": 0.12,
        "persistence_skill": 0.2,
        "interval_coverage": None,
        "interval_mean_width": None,
        "evaluated_at_utc": ORIGIN,
    }


def test_enterprise_forecast_contract_package_exists() -> None:
    assert importlib.util.find_spec("aquaops.forecasting") is not None


def test_enterprise_forecast_contract_module_exists() -> None:
    assert importlib.util.find_spec("aquaops.forecasting.contracts") is not None


def test_enterprise_forecast_contract_exports_v1_artifact_models() -> None:
    assert hasattr(contracts, "ForecastPointV1")
    assert hasattr(contracts, "ForecastRunManifestV1")
    assert hasattr(contracts, "ForecastMetricV1")
    assert hasattr(contracts, "ForecastDeliveryV1")
    assert hasattr(contracts, "forecast_point_arrow_schema_v1")
    assert hasattr(contracts, "forecast_metric_arrow_schema_v1")


def test_forecasting_package_exposes_only_the_versioned_contract_api() -> None:
    assert forecasting.__all__ == [
        "ForecastDeliveryV1",
        "ForecastMetricV1",
        "ForecastPointV1",
        "ForecastRunManifestV1",
        "forecast_metric_arrow_schema_v1",
        "forecast_point_arrow_schema_v1",
    ]


def test_forecast_point_v1_carries_time_lineage_and_safe_baseline_semantics() -> None:
    point = contracts.ForecastPointV1(**_forecast_point_payload())

    assert point.target_time_utc - point.origin_time_utc == timedelta(hours=24)
    assert point.interval_method == "heuristic"
    assert point.interval_level is None
    assert point.model_name == "persistence"
    assert point.data_snapshot_sha256 == "a" * 64


def test_forecast_point_v1_has_exact_parquet_arrow_schema() -> None:
    schema = contracts.forecast_point_arrow_schema_v1()

    assert schema.names == list(_forecast_point_payload())
    assert str(schema.field("origin_time_utc").type) == "timestamp[us, tz=UTC]"
    assert str(schema.field("target_time_utc").type) == "timestamp[us, tz=UTC]"
    assert schema.metadata == {b"aquaops.schema": b"forecast-point-v1"}


def test_forecast_point_v1_materializes_as_one_long_form_arrow_row() -> None:
    point = contracts.ForecastPointV1(**_forecast_point_payload())

    table = pa.Table.from_pylist(
        [point.model_dump()], schema=contracts.forecast_point_arrow_schema_v1()
    )

    assert table.num_rows == 1
    assert table.column("horizon_hours").to_pylist() == [24]
    assert table.column("target_time_utc").to_pylist() == [ORIGIN + timedelta(hours=24)]


@pytest.mark.parametrize(
    "changes",
    [
        {"target_time_utc": ORIGIN + timedelta(hours=23)},
        {"origin_time_utc": ORIGIN.replace(tzinfo=None)},
        {"point_estimate": float("inf")},
        {"data_snapshot_sha256": "not-a-hash"},
        {"lower_bound": 3.0, "point_estimate": 2.0},
        {"upper_bound": None},
        {"interval_method": "heuristic", "interval_level": 0.95},
        {"interval_method": "split_conformal", "interval_level": None},
        {"status": "insufficient_history"},
        {"training_cutoff_utc": ORIGIN + timedelta(seconds=1)},
        {"created_at_utc": ORIGIN + timedelta(hours=24)},
        {"metric_id": "../../raw.csv"},
        {"unit": "mg/L\nsecret"},
        {"status": "ready", "data_quality_flag": "fail"},
    ],
)
def test_forecast_point_v1_rejects_ambiguous_or_unsafe_rows(
    changes: dict[str, object],
) -> None:
    payload = _forecast_point_payload()
    payload.update(changes)

    with pytest.raises(ValidationError):
        contracts.ForecastPointV1(**payload)


def test_forecast_point_v1_allows_a_ready_point_without_an_interval() -> None:
    payload = _forecast_point_payload()
    payload.update(
        lower_bound=None,
        upper_bound=None,
        interval_method="none",
        interval_level=None,
    )

    point = contracts.ForecastPointV1(**payload)

    assert point.point_estimate == 2.0
    assert point.interval_method == "none"


def test_forecast_point_v1_json_schema_is_closed_and_versioned() -> None:
    schema = contracts.ForecastPointV1.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == 1


def test_run_manifest_v1_binds_model_data_features_and_delivery_artifact() -> None:
    manifest = contracts.ForecastRunManifestV1(**_run_manifest_payload())

    assert manifest.model_kind == "baseline"
    assert manifest.forecast_horizons_hours == (24,)
    assert manifest.data_snapshot_sha256 == "a" * 64
    assert manifest.forecast_points_sha256 == "d" * 64
    assert (
        contracts.ForecastRunManifestV1.model_validate_json(manifest.model_dump_json())
        == manifest
    )


def test_forecast_metric_v1_is_horizon_fold_and_condition_specific() -> None:
    metric = contracts.ForecastMetricV1(**_forecast_metric_payload())

    assert metric.evaluation_scope == "timecv"
    assert metric.fold_id == "fold-01"
    assert metric.condition == "all"
    assert metric.persistence_skill == 0.2


def test_forecast_metric_v1_has_exact_parquet_arrow_schema() -> None:
    schema = contracts.forecast_metric_arrow_schema_v1()

    assert schema.names == list(_forecast_metric_payload())
    assert str(schema.field("evaluated_at_utc").type) == "timestamp[us, tz=UTC]"
    assert schema.metadata == {b"aquaops.schema": b"forecast-metric-v1"}


@pytest.mark.parametrize(
    "changes",
    [
        {"model_kind": "learned", "model_artifact_sha256": None},
        {"model_kind": "baseline", "model_artifact_sha256": "e" * 64},
        {"forecast_horizons_hours": (24, 6)},
        {"forecast_horizons_hours": (24, 24)},
        {"forecast_horizons_hours": (0,)},
        {"calibration_start_utc": ORIGIN - timedelta(days=1)},
        {
            "calibration_start_utc": ORIGIN - timedelta(days=1),
            "calibration_end_utc": ORIGIN + timedelta(seconds=1),
        },
        {"generated_at_utc": ORIGIN - timedelta(seconds=1)},
        {"business_timezone": "../../local"},
        {"training_start_utc": (ORIGIN - timedelta(days=365)).replace(tzinfo=None)},
    ],
)
def test_run_manifest_v1_rejects_incomplete_or_inconsistent_lineage(
    changes: dict[str, object],
) -> None:
    payload = _run_manifest_payload()
    payload.update(changes)

    with pytest.raises(ValidationError):
        contracts.ForecastRunManifestV1(**payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"mae": -0.1},
        {"mae": 0.4, "rmse": 0.3},
        {"r2": float("nan")},
        {"fold_id": None},
        {"interval_coverage": 0.95, "interval_mean_width": None},
        {"interval_coverage": 1.01, "interval_mean_width": 0.5},
        {"interval_coverage": 0.95, "interval_mean_width": -0.1},
        {"metric_id": "../../raw.csv"},
        {"evaluated_at_utc": ORIGIN.replace(tzinfo=None)},
    ],
)
def test_forecast_metric_v1_rejects_invalid_or_untraceable_metrics(
    changes: dict[str, object],
) -> None:
    payload = _forecast_metric_payload()
    payload.update(changes)

    with pytest.raises(ValidationError):
        contracts.ForecastMetricV1(**payload)


def _evaluated_manifest() -> contracts.ForecastRunManifestV1:
    payload = _run_manifest_payload()
    payload["validation_status"] = "passed"
    return contracts.ForecastRunManifestV1(**payload)


def test_forecast_delivery_v1_closes_lineage_across_all_three_artifacts() -> None:
    delivery = contracts.ForecastDeliveryV1(
        manifest=_evaluated_manifest(),
        points=(contracts.ForecastPointV1(**_forecast_point_payload()),),
        metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
    )

    assert delivery.manifest.forecast_run_id == delivery.points[0].forecast_run_id
    assert delivery.points[0].horizon_hours == delivery.metrics[0].horizon_hours


@pytest.mark.parametrize(
    ("artifact", "changes"),
    [
        ("point", {"forecast_run_id": "run-other"}),
        ("point", {"model_version": "other-v1"}),
        ("point", {"data_snapshot_sha256": "e" * 64}),
        ("point", {"created_at_utc": ORIGIN + timedelta(seconds=2)}),
        ("metric", {"forecast_run_id": "run-other"}),
        ("metric", {"horizon_hours": 6}),
        ("metric", {"metric_id": "metric_other"}),
    ],
)
def test_forecast_delivery_v1_rejects_cross_artifact_drift(
    artifact: str,
    changes: dict[str, object],
) -> None:
    point_payload = _forecast_point_payload()
    metric_payload = _forecast_metric_payload()
    if artifact == "point":
        point_payload.update(changes)
    else:
        metric_payload.update(changes)

    with pytest.raises(ValidationError):
        contracts.ForecastDeliveryV1(
            manifest=_evaluated_manifest(),
            points=(contracts.ForecastPointV1(**point_payload),),
            metrics=(contracts.ForecastMetricV1(**metric_payload),),
        )


def test_forecast_delivery_v1_rejects_duplicate_prediction_keys() -> None:
    point = contracts.ForecastPointV1(**_forecast_point_payload())

    with pytest.raises(ValidationError):
        contracts.ForecastDeliveryV1(
            manifest=_evaluated_manifest(),
            points=(point, point),
            metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
        )


def test_delivery_rejects_persistence_disguised_as_a_calibrated_interval() -> None:
    point_payload = _forecast_point_payload()
    point_payload.update(interval_method="split_conformal", interval_level=0.95)

    with pytest.raises(ValidationError):
        contracts.ForecastDeliveryV1(
            manifest=_evaluated_manifest(),
            points=(contracts.ForecastPointV1(**point_payload),),
            metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
        )


def test_delivery_rejects_feature_version_drift() -> None:
    manifest_payload = _run_manifest_payload()
    manifest_payload.update(
        validation_status="passed",
        feature_version="other-feature-v1",
    )

    with pytest.raises(ValidationError):
        contracts.ForecastDeliveryV1(
            manifest=contracts.ForecastRunManifestV1(**manifest_payload),
            points=(contracts.ForecastPointV1(**_forecast_point_payload()),),
            metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
        )


def test_delivery_rejects_a_declared_horizon_without_a_point() -> None:
    manifest_payload = _run_manifest_payload()
    manifest_payload.update(
        validation_status="passed",
        forecast_horizons_hours=(6, 24),
    )

    with pytest.raises(ValidationError):
        contracts.ForecastDeliveryV1(
            manifest=contracts.ForecastRunManifestV1(**manifest_payload),
            points=(contracts.ForecastPointV1(**_forecast_point_payload()),),
            metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
        )


def test_delivery_rejects_unit_drift_for_one_plant_metric() -> None:
    first_payload = _forecast_point_payload()
    second_payload = _forecast_point_payload()
    second_payload.update(
        origin_time_utc=ORIGIN + timedelta(seconds=1),
        target_time_utc=ORIGIN + timedelta(hours=24, seconds=1),
        created_at_utc=ORIGIN + timedelta(seconds=1),
        unit="g/L",
    )

    with pytest.raises(ValidationError):
        contracts.ForecastDeliveryV1(
            manifest=_evaluated_manifest(),
            points=(
                contracts.ForecastPointV1(**first_payload),
                contracts.ForecastPointV1(**second_payload),
            ),
            metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
        )


def test_run_manifest_rejects_a_syntactically_valid_unknown_timezone() -> None:
    payload = _run_manifest_payload()
    payload["business_timezone"] = "Mars/Olympus"

    with pytest.raises(ValidationError):
        contracts.ForecastRunManifestV1(**payload)


def test_conformal_run_requires_a_post_training_calibration_window() -> None:
    manifest_payload = _run_manifest_payload()
    manifest_payload.update(
        model_kind="learned",
        model_name="ridge",
        model_version="ridge-v1",
        model_artifact_sha256="e" * 64,
        validation_status="passed",
    )
    point_payload = _forecast_point_payload()
    point_payload.update(
        model_name="ridge",
        model_version="ridge-v1",
        interval_method="split_conformal",
        interval_level=0.95,
    )

    with pytest.raises(ValidationError):
        contracts.ForecastDeliveryV1(
            manifest=contracts.ForecastRunManifestV1(**manifest_payload),
            points=(contracts.ForecastPointV1(**point_payload),),
            metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
        )


def test_learned_conformal_delivery_accepts_an_independent_calibration_window() -> None:
    manifest_payload = _run_manifest_payload()
    manifest_payload.update(
        model_kind="learned",
        model_name="ridge",
        model_version="ridge-v1",
        model_artifact_sha256="e" * 64,
        validation_status="passed",
        training_cutoff_utc=ORIGIN - timedelta(hours=2),
        calibration_start_utc=ORIGIN - timedelta(hours=1),
        calibration_end_utc=ORIGIN - timedelta(minutes=30),
    )
    point_payload = _forecast_point_payload()
    point_payload.update(
        model_name="ridge",
        model_version="ridge-v1",
        interval_method="split_conformal",
        interval_level=0.95,
        training_cutoff_utc=ORIGIN - timedelta(hours=2),
    )

    delivery = contracts.ForecastDeliveryV1(
        manifest=contracts.ForecastRunManifestV1(**manifest_payload),
        points=(contracts.ForecastPointV1(**point_payload),),
        metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
    )

    assert delivery.points[0].interval_level == 0.95


def test_delivery_revalidates_a_model_constructed_forecast_point() -> None:
    forged_payload = _forecast_point_payload()
    forged_payload["point_estimate"] = float("inf")
    forged = BaseModel.model_construct.__func__(
        contracts.ForecastPointV1,
        **forged_payload,
    )

    with pytest.raises(ValidationError):
        contracts.ForecastDeliveryV1(
            manifest=_evaluated_manifest(),
            points=(forged,),
            metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
        )


def test_delivery_rejects_calibration_after_the_forecast_origin() -> None:
    manifest_payload = _run_manifest_payload()
    manifest_payload.update(
        model_kind="learned",
        model_name="ridge",
        model_version="ridge-v1",
        model_artifact_sha256="e" * 64,
        validation_status="passed",
        calibration_start_utc=ORIGIN + timedelta(microseconds=1),
        calibration_end_utc=ORIGIN + timedelta(microseconds=2),
    )
    point_payload = _forecast_point_payload()
    point_payload.update(
        model_name="ridge",
        model_version="ridge-v1",
        interval_method="split_conformal",
        interval_level=0.95,
    )

    with pytest.raises(ValidationError):
        contracts.ForecastDeliveryV1(
            manifest=contracts.ForecastRunManifestV1(**manifest_payload),
            points=(contracts.ForecastPointV1(**point_payload),),
            metrics=(contracts.ForecastMetricV1(**_forecast_metric_payload()),),
        )


def test_delivery_revalidates_field_only_forgery_for_each_nested_model() -> None:
    valid_manifest = _evaluated_manifest()
    valid_point = contracts.ForecastPointV1(**_forecast_point_payload())
    valid_metric = contracts.ForecastMetricV1(**_forecast_metric_payload())
    forged_point_payload = _forecast_point_payload()
    forged_point_payload["schema_version"] = 2
    forged_manifest_payload = _run_manifest_payload()
    forged_manifest_payload.update(
        validation_status="passed",
        business_timezone="Mars/Olympus",
    )
    forged_metric_payload = _forecast_metric_payload()
    forged_metric_payload["schema_version"] = 2

    forged_cases = (
        (
            valid_manifest,
            BaseModel.model_construct.__func__(
                contracts.ForecastPointV1, **forged_point_payload
            ),
            valid_metric,
        ),
        (
            BaseModel.model_construct.__func__(
                contracts.ForecastRunManifestV1, **forged_manifest_payload
            ),
            valid_point,
            valid_metric,
        ),
        (
            valid_manifest,
            valid_point,
            BaseModel.model_construct.__func__(
                contracts.ForecastMetricV1, **forged_metric_payload
            ),
        ),
    )

    for manifest, point, metric in forged_cases:
        with pytest.raises(ValidationError):
            contracts.ForecastDeliveryV1(
                manifest=manifest,
                points=(point,),
                metrics=(metric,),
            )


def test_forecast_point_rejects_spreadsheet_formula_unit() -> None:
    payload = _forecast_point_payload()
    payload["unit"] = "=1+1"

    with pytest.raises(ValidationError):
        contracts.ForecastPointV1(**payload)
