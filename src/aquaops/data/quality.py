"""Aggregate-only quality profiling for approved public water datasets.

This module deliberately rejects paths that resolve under ``data/private`` and
never returns source rows. It is intended for the public Co-UDlabs WWTP
Langmatt CSV acquired through the project's allowlisted acquisition workflow.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ID = "co-udlabs-wwtp-lpicm-2025"
PRIVATE_DATA_ROOT = PROJECT_ROOT / "data" / "private"
PUBLIC_SOURCE_ROOT = PROJECT_ROOT / "data" / "public" / SOURCE_ID
DERIVED_SOURCE_ROOT = PROJECT_ROOT / "data" / "public" / "derived" / SOURCE_ID
_REQUIRED_COLUMNS = ("date_vec", "q", "cond", "nh4")
_NUMERIC_COLUMNS = ("q", "cond", "nh4")
_PROFILE_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_fields",
        "row_count",
        "timestamp_column",
        "start",
        "end",
        "timestamp_order_violations",
        "duplicate_timestamp_count",
        "median_interval_seconds",
        "numeric_columns",
    }
)
_NUMERIC_SUMMARY_FIELDS = frozenset(
    {
        "non_null_count",
        "null_count",
        "invalid_numeric_count",
        "min",
        "max",
        "mean",
    }
)


class PublicDataQualityError(ValueError):
    """Raised when public-data profiling would be unsafe or invalid."""


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError as error:
        raise PublicDataQualityError("could not resolve data path") from error


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_private_path(path: Path) -> Path:
    resolved_path = _resolve_path(path)
    if _is_within(resolved_path, _resolve_path(PRIVATE_DATA_ROOT)):
        raise PublicDataQualityError("private data paths are not supported")
    return resolved_path


def _parse_timestamp(value: str, row_number: int) -> datetime:
    candidate = value.strip()
    if not candidate:
        raise PublicDataQualityError(f"invalid timestamp at row {row_number}")
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        timestamp = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise PublicDataQualityError(
            f"invalid timestamp at row {row_number}"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _format_timestamp(timestamp: datetime) -> str:
    normalized = timestamp.astimezone(timezone.utc).isoformat()
    return f"{normalized[:-6]}Z"


def _empty_numeric_summary() -> dict[str, int | float | None]:
    return {
        "non_null_count": 0,
        "null_count": 0,
        "invalid_numeric_count": 0,
        "min": None,
        "max": None,
        "mean": None,
    }


def _validate_output_path(output_path: Path, input_path: Path | None) -> Path:
    resolved_output = _reject_private_path(output_path)
    if input_path is not None and resolved_output == _resolve_path(input_path):
        raise PublicDataQualityError("profile output must not overwrite the input")

    derived_source_root = _resolve_path(DERIVED_SOURCE_ROOT)
    if not _is_within(resolved_output, derived_source_root):
        raise PublicDataQualityError(
            "profile output must be within the approved derived public source directory"
        )
    if resolved_output.suffix.lower() == ".csv":
        raise PublicDataQualityError("profile output must not be a CSV")
    return resolved_output


def _schema_error(message: str) -> PublicDataQualityError:
    return PublicDataQualityError(f"profile schema is invalid: {message}")


def _require_exact_fields(
    value: Mapping[object, object], expected_fields: frozenset[str], context: str
) -> None:
    actual_fields = set(value)
    unexpected_fields = actual_fields - expected_fields
    missing_fields = expected_fields - actual_fields
    if unexpected_fields:
        rendered_fields = ", ".join(sorted(map(str, unexpected_fields)))
        raise _schema_error(f"{context} has unexpected fields: {rendered_fields}")
    if missing_fields:
        rendered_fields = ", ".join(sorted(missing_fields))
        raise _schema_error(f"{context} is missing fields: {rendered_fields}")


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_float_or_none(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isfinite(value))


def _is_canonical_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    candidate = f"{value[:-1]}+00:00"
    try:
        timestamp = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return value == _format_timestamp(timestamp)


def _validate_profile_schema(profile: Mapping[str, object]) -> None:
    """Accept only the aggregate profile shape produced by this module."""
    _require_exact_fields(profile, _PROFILE_TOP_LEVEL_FIELDS, "profile")

    schema_fields = profile["schema_fields"]
    if (
        not isinstance(schema_fields, list)
        or not all(isinstance(field, str) for field in schema_fields)
        or any(column not in schema_fields for column in _REQUIRED_COLUMNS)
    ):
        raise _schema_error("schema_fields must contain the required string columns")

    row_count = profile["row_count"]
    if not _is_nonnegative_int(row_count) or row_count == 0:
        raise _schema_error("row_count must be a positive integer")
    if profile["timestamp_column"] != "date_vec":
        raise _schema_error("timestamp_column must be date_vec")
    if not _is_canonical_utc_timestamp(
        profile["start"]
    ) or not _is_canonical_utc_timestamp(profile["end"]):
        raise _schema_error(
            "start and end must be canonical UTC timestamps ending in Z"
        )

    for field in ("timestamp_order_violations", "duplicate_timestamp_count"):
        if not _is_nonnegative_int(profile[field]):
            raise _schema_error(f"{field} must be a non-negative integer")
    if profile["timestamp_order_violations"] > row_count - 1:
        raise _schema_error("timestamp_order_violations exceeds the row count")
    if profile["duplicate_timestamp_count"] >= row_count:
        raise _schema_error("duplicate_timestamp_count exceeds the row count")

    median_interval = profile["median_interval_seconds"]
    if not _is_finite_float_or_none(median_interval) or (
        median_interval is not None and median_interval <= 0
    ):
        raise _schema_error("median_interval_seconds must be a positive float or null")
    if row_count < 2 and median_interval is not None:
        raise _schema_error("median_interval_seconds requires at least two rows")

    numeric_columns = profile["numeric_columns"]
    if not isinstance(numeric_columns, Mapping):
        raise _schema_error("numeric_columns must be an object")
    _require_exact_fields(
        numeric_columns, frozenset(_NUMERIC_COLUMNS), "numeric_columns"
    )
    for column in _NUMERIC_COLUMNS:
        summary = numeric_columns[column]
        if not isinstance(summary, Mapping):
            raise _schema_error(f"numeric_columns.{column} must be an object")
        _require_exact_fields(
            summary, _NUMERIC_SUMMARY_FIELDS, f"numeric_columns.{column}"
        )

        for field in ("non_null_count", "null_count", "invalid_numeric_count"):
            if not _is_nonnegative_int(summary[field]):
                raise _schema_error(
                    f"numeric_columns.{column}.{field} must be a non-negative integer"
                )
        if summary["invalid_numeric_count"] != 0:
            raise _schema_error(
                f"numeric_columns.{column}.invalid_numeric_count must be zero"
            )
        if summary["non_null_count"] + summary["null_count"] != row_count:
            raise _schema_error(f"numeric_columns.{column} counts must equal row_count")

        statistics_values = (summary["min"], summary["max"], summary["mean"])
        if not all(_is_finite_float_or_none(value) for value in statistics_values):
            raise _schema_error(
                f"numeric_columns.{column} statistics must be finite floats or null"
            )
        if summary["non_null_count"] == 0:
            if any(value is not None for value in statistics_values):
                raise _schema_error(
                    f"numeric_columns.{column} statistics must be null without values"
                )
        elif any(value is None for value in statistics_values):
            raise _schema_error(
                f"numeric_columns.{column} statistics must be present with values"
            )
        elif not (summary["min"] <= summary["mean"] <= summary["max"]):
            raise _schema_error(
                f"numeric_columns.{column} mean must fall between min and max"
            )


def profile_wwtp_langmatt_csv(csv_path: Path) -> dict[str, object]:
    """Return a non-sensitive aggregate profile for a public WWTP Langmatt CSV.

    The function validates all timestamps and numeric values before returning a
    result. Non-empty invalid numeric cells are counted internally and cause a
    safe failure instead of producing a partial profile.
    """
    resolved_csv_path = _reject_private_path(csv_path)
    if not _is_within(resolved_csv_path, _resolve_path(PUBLIC_SOURCE_ROOT)):
        raise PublicDataQualityError(
            "public source input must be within the approved public source directory"
        )
    if resolved_csv_path.suffix.lower() != ".csv":
        raise PublicDataQualityError("public source input must be a CSV")
    if not resolved_csv_path.is_file():
        raise PublicDataQualityError("public source CSV does not exist")

    numeric_values: dict[str, list[float]] = {column: [] for column in _NUMERIC_COLUMNS}
    numeric_summaries = {
        column: _empty_numeric_summary() for column in _NUMERIC_COLUMNS
    }
    timestamps: list[datetime] = []
    row_count = 0

    try:
        with resolved_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            schema_fields = reader.fieldnames
            if schema_fields is None:
                raise PublicDataQualityError("public source CSV has no header")
            missing_columns = [
                column for column in _REQUIRED_COLUMNS if column not in schema_fields
            ]
            if missing_columns:
                raise PublicDataQualityError(
                    f"missing required columns: {', '.join(missing_columns)}"
                )

            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                timestamp = _parse_timestamp(row.get("date_vec") or "", row_number)
                timestamps.append(timestamp)

                for column in _NUMERIC_COLUMNS:
                    raw_value = row.get(column)
                    summary = numeric_summaries[column]
                    if raw_value is None or not raw_value.strip():
                        summary["null_count"] += 1
                        continue
                    try:
                        numeric_value = float(raw_value)
                    except ValueError:
                        summary["invalid_numeric_count"] += 1
                        continue
                    if not math.isfinite(numeric_value):
                        summary["invalid_numeric_count"] += 1
                        continue
                    numeric_values[column].append(numeric_value)
    except PublicDataQualityError:
        raise
    except UnicodeDecodeError as error:
        raise PublicDataQualityError("public source CSV is not valid UTF-8") from error
    except csv.Error as error:
        raise PublicDataQualityError("public source CSV could not be parsed") from error
    except OSError as error:
        raise PublicDataQualityError("public source CSV could not be read") from error

    if row_count == 0:
        raise PublicDataQualityError("public source CSV has no data rows")

    invalid_columns = [
        f"{column}({numeric_summaries[column]['invalid_numeric_count']})"
        for column in _NUMERIC_COLUMNS
        if numeric_summaries[column]["invalid_numeric_count"]
    ]
    if invalid_columns:
        raise PublicDataQualityError(
            f"invalid numeric values: {', '.join(invalid_columns)}"
        )

    for column, values in numeric_values.items():
        summary = numeric_summaries[column]
        summary["non_null_count"] = len(values)
        if values:
            summary["min"] = min(values)
            summary["max"] = max(values)
            summary["mean"] = statistics.fmean(values)

    timestamp_order_violations = sum(
        current <= previous for previous, current in zip(timestamps, timestamps[1:])
    )
    positive_intervals = [
        (current - previous).total_seconds()
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    median_interval_seconds = (
        float(statistics.median(positive_intervals)) if positive_intervals else None
    )

    return {
        "schema_fields": schema_fields,
        "row_count": row_count,
        "timestamp_column": "date_vec",
        "start": _format_timestamp(min(timestamps)),
        "end": _format_timestamp(max(timestamps)),
        "timestamp_order_violations": timestamp_order_violations,
        "duplicate_timestamp_count": len(timestamps) - len(set(timestamps)),
        "median_interval_seconds": median_interval_seconds,
        "numeric_columns": numeric_summaries,
    }


def _replace_profile_file(temporary_path: Path, output_path: Path) -> None:
    temporary_path.replace(output_path)


def _write_profile_atomically(profile: Mapping[str, object], output_path: Path) -> None:
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(profile, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_profile_file(temporary_path, output_path)
        temporary_path = None
    except (OSError, TypeError, ValueError) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise PublicDataQualityError(
            "could not write public quality profile"
        ) from error


def write_profile(
    profile: Mapping[str, object], output_path: Path, *, input_path: Path | None = None
) -> None:
    """Atomically write a safe aggregate profile under the derived source path."""
    resolved_output_path = _validate_output_path(output_path, input_path)
    _validate_profile_schema(profile)
    _write_profile_atomically(profile, resolved_output_path)
