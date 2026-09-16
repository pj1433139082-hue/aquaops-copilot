from __future__ import annotations

from math import isfinite
from typing import Literal


LatencyBucket = Literal["under_100ms", "under_500ms", "under_2s", "over_2s"]


def latency_bucket_for(elapsed_milliseconds: float) -> LatencyBucket:
    if not isfinite(elapsed_milliseconds) or elapsed_milliseconds < 0:
        raise ValueError("latency must be finite and non-negative")
    if elapsed_milliseconds < 100:
        return "under_100ms"
    if elapsed_milliseconds < 500:
        return "under_500ms"
    if elapsed_milliseconds < 2_000:
        return "under_2s"
    return "over_2s"


def event_payload(
    event: str,
    *,
    request_id: str,
    run_id: str | None = None,
    data_version: str | None = None,
    model_version: str | None = None,
    tool_name: str | None = None,
    latency_bucket: LatencyBucket | None = None,
    retry_count: int | None = None,
    error_class: str | None = None,
) -> dict[str, str | int]:
    payload: dict[str, str | int] = {"event": event, "request_id": request_id}
    optional = {
        "run_id": run_id,
        "data_version": data_version,
        "model_version": model_version,
        "tool_name": tool_name,
        "latency_bucket": latency_bucket,
        "error_class": error_class,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    if retry_count is not None:
        payload["retry_count"] = retry_count
    return payload
