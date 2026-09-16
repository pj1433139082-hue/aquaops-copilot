# Enterprise forecast output contract v1

Status: `PLAN_FREEZE approved 2026-08-12`

This contract separates machine-readable forecast delivery from the existing
private Operations Agent summary. It defines delivery fields and validation
rules only. It does not train a model, expose private data, add an API, or turn
the current persistence baseline into a production-accuracy claim.

## Delivery artifacts

One forecast delivery consists of three versioned artifacts:

| Artifact | Primary format | Purpose |
|---|---|---|
| `forecast-points-v1.parquet` | Long-form Parquet | One independently produced horizon prediction per row |
| `forecast-run-manifest-v1.json` | JSON | Run, model, data, feature and time lineage |
| `forecast-metrics-v1.parquet` | Long-form Parquet | Per-horizon, per-fold and per-condition evaluation |

Pydantic models provide strict JSON Schema through `model_json_schema()`. The
two Parquet tables use the Arrow schemas returned by
`forecast_point_arrow_schema_v1()` and `forecast_metric_arrow_schema_v1()`.
All schemas reject unknown fields.

## Forecast point contract

`ForecastPointV1` records:

- `schema_version`, fixed to `1`;
- `forecast_run_id`, `plant_alias`, `metric_id` and `unit`;
- `origin_time_utc`, `target_time_utc` and `horizon_hours`;
- `point_estimate`, optional bounds and exact interval semantics;
- delivery status and data-quality disposition;
- model, feature, training-cutoff and data-snapshot lineage; and
- `created_at_utc`.

The contract requires:

- UTC-aware timestamps;
- `target_time_utc - origin_time_utc == horizon_hours`;
- training data ending no later than the forecast origin;
- creation after the origin and before the target;
- finite values and ordered, complete bounds;
- no estimate when status is unavailable; and
- a failed quality flag to use `data_quality_blocked` rather than `ready`.

The idempotency key is the tuple
`(forecast_run_id, plant_alias, metric_id, origin_time_utc, target_time_utc)`.
Duplicate keys are rejected by the delivery gate.

### Interval semantics

`interval_method="heuristic"` may carry bounds but must leave
`interval_level=null`. It therefore cannot be presented as a calibrated 90% or
95% interval. `split_conformal`, `rolling_conformal`, and `quantile` require an
explicit level between zero and one. `interval_method="none"` carries no
bounds.

The existing AquaOps persistence summary maps to a single `h=24` row with
`interval_method="heuristic"`. It must not be expanded into 24 hourly rows.
Additional horizons may be emitted only after each horizon has its own aligned
target, leakage audit, TimeCV and evaluation evidence.

## Run manifest contract

`ForecastRunManifestV1` binds all delivered rows to:

- model kind, name, version and optional learned-model artifact hash;
- code revision, data snapshot hash, feature version and feature-specification hash;
- training and optional calibration windows;
- sorted unique forecast horizons and source cadence;
- the business timezone while stored timestamps remain UTC;
- validation and drift status; and
- the SHA-256 of the forecast-points artifact.

A learned model requires an artifact hash. A deterministic baseline must not
invent one. Calibration timestamps must either both be absent or form an
independent post-training window: `training_cutoff < calibration_start <=
calibration_end <= forecast_origin <= generated_at`. Conformal delivery requires
that window, and the delivery gate checks it against every forecast origin.
`business_timezone` must be a real IANA timezone; all stored event timestamps
remain UTC.

## Evaluation metric contract

`ForecastMetricV1` records metrics by plant, anonymous metric, horizon,
evaluation scope, fold and condition. It includes sample count, MAE, RMSE,
optional R-squared, WAPE, sMAPE, persistence skill, interval coverage and mean
interval width.

TimeCV rows require a fold identifier. Interval coverage and width must appear
together. Metrics contain no raw observations or prediction arrays.

## Cross-artifact delivery gate

`ForecastDeliveryV1` rejects a delivery when any point or metric has a different
run ID, model version, feature version, data snapshot, training cutoff,
generation time or horizon from its manifest. Every plant/metric/origin group
must deliver the full declared horizon set, including explicit unavailable rows
when evidence is insufficient. Units cannot change for the same plant/metric
inside one run. Metrics must correspond to a delivered plant/metric/horizon
key. Duplicate prediction and metric keys are rejected.

This gate should run before writing an API response, database batch, Parquet
dataset or human CSV export.

## Synthetic example

```json
{
  "schema_version": 1,
  "forecast_run_id": "run-20260812-001",
  "plant_alias": "plant_demo_001",
  "metric_id": "metric_t001_column_001",
  "unit": "mg/L",
  "origin_time_utc": "2026-08-12T00:00:00Z",
  "target_time_utc": "2026-08-13T00:00:00Z",
  "horizon_hours": 24,
  "point_estimate": 2.0,
  "lower_bound": 1.0,
  "upper_bound": 3.0,
  "interval_method": "heuristic",
  "interval_level": null,
  "status": "ready",
  "data_quality_flag": "pass",
  "model_name": "persistence",
  "model_version": "baseline-v1",
  "feature_version": "latest-observation-v1",
  "training_cutoff_utc": "2026-08-12T00:00:00Z",
  "data_snapshot_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "created_at_utc": "2026-08-12T00:00:01Z"
}
```

The example is synthetic and contains no private plant, metric, path or source
identity.

## Storage and serving boundary

- Parquet is the batch/warehouse source of truth for forecast points and
  evaluation metrics.
- A database or JSON API may serve validated rows without changing their field
  semantics.
- CSV is an optional human exchange format, not the authoritative artifact.
- Contract units use a restricted non-formula grammar. A future CSV writer must
  still neutralize spreadsheet formula prefixes for every exported text field.
- Excel is not a system of record.
- Partitioning may use forecast date and scoped plant alias, but partition keys
  must not replace the timestamps and IDs inside each row.
- The private Agent continues to consume only verified aggregate evidence and
  render its fixed ABC response. It does not receive these full delivery rows.

## Current implementation boundary

The v1 contract is implemented in `src/aquaops/forecasting/contracts.py`. The
cross-artifact gate revalidates nested models, including objects constructed by
bypassing normal Pydantic initialization. It is tested only with synthetic
values in `tests/forecasting/test_contracts.py`.
There is not yet a production writer, database migration, public API or learned
water-quality model connected to it. Those are separate reviewed tasks.
