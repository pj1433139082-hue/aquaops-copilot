"""Aggregate-only access to the fixed, verified public WWTP history."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from aquaops.data.public_history import (
    PUBLIC_SOURCE_ID,
    PUBLIC_SOURCE_URL,
    PUBLIC_SOURCE_VERSION,
    PublicHistoryError,
    _INDICATOR_UNITS,
    _load_verified_public_observations,
)


class PublicWindowError(ValueError):
    """Raised when a public historical aggregate cannot be safely produced."""


@dataclass(frozen=True)
class PublicWindowSummary:
    """A non-identifying aggregate for one approved historical time window."""

    source_id: str
    source_url: str
    source_version: str
    indicator: str
    unit: str
    start: str
    end: str
    row_count: int
    null_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None


def _format_utc(timestamp: datetime) -> str:
    normalized = timestamp.astimezone(timezone.utc).isoformat()
    return f"{normalized[:-6]}Z"


def _parse_canonical_end_at(end_at: str) -> datetime:
    if not isinstance(end_at, str) or not end_at.endswith("Z"):
        raise PublicWindowError("end_at must be canonical UTC Z")
    try:
        timestamp = datetime.fromisoformat(f"{end_at[:-1]}+00:00")
    except ValueError as error:
        raise PublicWindowError("end_at must be canonical UTC Z") from error
    if timestamp.tzinfo is None or _format_utc(timestamp) != end_at:
        raise PublicWindowError("end_at must be canonical UTC Z")
    return timestamp.astimezone(timezone.utc)


def _validate_query(indicator: str, hours: int, end_at: str) -> tuple[str, datetime]:
    if not isinstance(indicator, str) or indicator not in _INDICATOR_UNITS:
        raise PublicWindowError("unsupported public indicator")
    if type(hours) is not int or not 1 <= hours <= 168:
        raise PublicWindowError("hours must be an integer from 1 to 168")
    return indicator, _parse_canonical_end_at(end_at)


def summarize_public_window(
    indicator: str, hours: int, end_at: str
) -> PublicWindowSummary:
    """Summarize `(end_at - hours, end_at]` from the fixed public CSV only."""

    selected_indicator, end = _validate_query(indicator, hours, end_at)
    start = end - timedelta(hours=hours)
    try:
        observations = _load_verified_public_observations()
    except PublicHistoryError as error:
        raise PublicWindowError(str(error)) from None

    row_count = 0
    null_count = 0
    non_null_count = 0
    numeric_total = 0.0
    minimum: float | None = None
    maximum: float | None = None
    for observation in observations:
        if not start < observation.timestamp <= end:
            continue
        row_count += 1
        selected_value = observation.values[selected_indicator]
        if selected_value is None:
            null_count += 1
            continue
        non_null_count += 1
        numeric_total += selected_value
        if not math.isfinite(numeric_total):
            raise PublicWindowError("public window statistics overflow")
        minimum = selected_value if minimum is None else min(minimum, selected_value)
        maximum = selected_value if maximum is None else max(maximum, selected_value)

    if row_count == 0:
        raise PublicWindowError("no public observations cover the requested window")

    mean = numeric_total / non_null_count if non_null_count else None
    if mean is not None and not math.isfinite(mean):
        raise PublicWindowError("public window statistics overflow")
    return PublicWindowSummary(
        source_id=PUBLIC_SOURCE_ID,
        source_url=PUBLIC_SOURCE_URL,
        source_version=PUBLIC_SOURCE_VERSION,
        indicator=selected_indicator,
        unit=_INDICATOR_UNITS[selected_indicator],
        start=_format_utc(start),
        end=_format_utc(end),
        row_count=row_count,
        null_count=null_count,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
    )
