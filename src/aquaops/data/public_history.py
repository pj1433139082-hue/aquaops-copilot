"""Safely load the one fixed, approved public WWTP history source.

This internal module never accepts a caller-provided path.  It validates the
controlled layout and registered hash before parsing bytes, then returns only
immutable typed observations for other data-layer code to aggregate.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from aquaops.rag.source_registry import load_public_source_registry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_SOURCE_ID = "co-udlabs-wwtp-lpicm-2025"
PUBLIC_SOURCE_ROOT = PROJECT_ROOT / "data" / "public" / PUBLIC_SOURCE_ID
PUBLIC_CSV_PATH = PUBLIC_SOURCE_ROOT / "WWTP_Langmatt" / "data" / "WWTP_Langmatt.csv"
PUBLIC_SOURCE_URL = "https://zenodo.org/records/15285089"
PUBLIC_SOURCE_VERSION = "v1.0.0 (2025-04-26)"

_PUBLIC_SOURCE_RELATIVE_PATH = Path("data") / "public" / PUBLIC_SOURCE_ID
_CSV_RELATIVE_PATH = Path("WWTP_Langmatt") / "data" / "WWTP_Langmatt.csv"
_INDICATOR_UNITS = {
    "nh4": "mg/L",
    "cond": "µS/cm",
    "q": "L/s",
}
_REQUIRED_COLUMNS = ("date_vec", *_INDICATOR_UNITS)
_OBSERVATION_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?$"
)


class PublicHistoryError(ValueError):
    """Raised when verified public historical observations are unavailable."""


@dataclass(frozen=True)
class PublicObservation:
    """An immutable parsed observation for the fixed approved public source."""

    timestamp: datetime
    values: Mapping[str, float | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError as error:
        raise PublicHistoryError(
            "could not resolve approved public CSV path"
        ) from error


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _expected_public_root() -> Path:
    return PROJECT_ROOT / "data" / "public"


def _expected_source_root() -> Path:
    return PROJECT_ROOT / _PUBLIC_SOURCE_RELATIVE_PATH


def _expected_csv_path() -> Path:
    return _expected_source_root() / _CSV_RELATIVE_PATH


def _reject_symbolic_link_components(*components: Path) -> None:
    for component in components:
        if component.is_symlink():
            raise PublicHistoryError(
                "approved public paths must not use symbolic links"
            )


def _validated_public_csv_path() -> Path:
    public_root = _expected_public_root()
    expected_source_root = _expected_source_root()
    expected_csv_path = _expected_csv_path()

    if PUBLIC_SOURCE_ROOT != expected_source_root:
        raise PublicHistoryError(
            "approved public paths must match the controlled public source layout"
        )
    if not _is_within(PUBLIC_CSV_PATH, PUBLIC_SOURCE_ROOT):
        raise PublicHistoryError(
            "public CSV path must remain within the approved public source root"
        )
    if PUBLIC_CSV_PATH != expected_csv_path:
        raise PublicHistoryError(
            "approved public paths must match the controlled public source layout"
        )
    _reject_symbolic_link_components(
        public_root.parent,
        public_root,
        expected_source_root,
        expected_source_root / _CSV_RELATIVE_PATH.parts[0],
        expected_source_root / _CSV_RELATIVE_PATH.parent,
        expected_csv_path,
    )

    resolved_public_root = _resolve(public_root)
    resolved_source_root = _resolve(expected_source_root)
    resolved_csv_path = _resolve(expected_csv_path)
    if not _is_within(resolved_source_root, resolved_public_root) or not _is_within(
        resolved_csv_path, resolved_public_root
    ):
        raise PublicHistoryError(
            "approved public paths resolve outside the controlled public root"
        )
    if not _is_within(resolved_csv_path, resolved_source_root):
        raise PublicHistoryError(
            "public CSV path must remain within the approved public source root"
        )
    if resolved_csv_path.suffix.lower() != ".csv":
        raise PublicHistoryError("approved public source must be a .csv file")
    if not resolved_csv_path.is_file():
        raise PublicHistoryError("approved public CSV file does not exist")
    return resolved_csv_path


def _validated_public_registry_path() -> Path:
    public_root = _expected_public_root()
    registry_path = public_root / "sources.json"
    _reject_symbolic_link_components(public_root.parent, public_root, registry_path)

    resolved_public_root = _resolve(public_root)
    resolved_registry_path = _resolve(registry_path)
    if (
        not _is_within(resolved_registry_path, resolved_public_root)
        or not resolved_registry_path.is_file()
    ):
        raise PublicHistoryError(
            "approved public source integrity metadata is unavailable"
        )
    return resolved_registry_path


def _is_lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_registered_csv_sha256() -> str:
    registry_path = _validated_public_registry_path()
    try:
        sources = load_public_source_registry(registry_path)
    except (OSError, UnicodeError, ValueError) as error:
        raise PublicHistoryError(
            "approved public source integrity metadata is unavailable"
        ) from error

    source = next(
        (candidate for candidate in sources if candidate.source_id == PUBLIC_SOURCE_ID),
        None,
    )
    if source is None:
        raise PublicHistoryError(
            "approved public source integrity metadata is unavailable"
        )
    registered_sha256 = source.verified_file_sha256.get(_CSV_RELATIVE_PATH.as_posix())
    if not _is_lowercase_sha256(registered_sha256):
        raise PublicHistoryError(
            "approved public source integrity metadata is unavailable"
        )
    return registered_sha256


def _read_verified_csv_bytes(csv_path: Path) -> bytes:
    registered_sha256 = _load_registered_csv_sha256()
    if not _is_lowercase_sha256(registered_sha256):
        raise PublicHistoryError(
            "approved public source integrity metadata is unavailable"
        )
    try:
        source_bytes = csv_path.read_bytes()
    except OSError as error:
        raise PublicHistoryError("could not read approved public CSV") from error
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if not hmac.compare_digest(actual_sha256, registered_sha256):
        raise PublicHistoryError("approved public CSV integrity check failed")
    return source_bytes


def _parse_observation_timestamp(value: object, row_number: int) -> datetime:
    if not isinstance(value, str) or not _OBSERVATION_TIMESTAMP_PATTERN.fullmatch(
        value
    ):
        raise PublicHistoryError(f"invalid timestamp at row {row_number}")
    candidate = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise PublicHistoryError(f"invalid timestamp at row {row_number}") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        # The approved WWTP_Langmatt README declares naive date_vec timestamps UTC.
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _parse_optional_numeric(
    value: object, indicator: str, row_number: int
) -> float | None:
    if value == "":
        return None
    if not isinstance(value, str):
        raise PublicHistoryError(
            f"invalid numeric value for {indicator} at row {row_number}"
        )
    try:
        parsed = float(value)
    except ValueError as error:
        raise PublicHistoryError(
            f"invalid numeric value for {indicator} at row {row_number}"
        ) from error
    if not math.isfinite(parsed):
        raise PublicHistoryError(
            f"invalid numeric value for {indicator} at row {row_number}"
        )
    return parsed


def _validate_headers(fieldnames: list[str] | None) -> None:
    actual_headers = set(fieldnames or [])
    missing_columns = [
        column for column in _REQUIRED_COLUMNS if column not in actual_headers
    ]
    if missing_columns:
        raise PublicHistoryError(
            f"missing required columns: {', '.join(missing_columns)}"
        )


def _load_verified_public_observations() -> tuple[PublicObservation, ...]:
    """Load immutable observations from the fixed public CSV after hash verification."""

    csv_path = _validated_public_csv_path()
    try:
        verified_csv_text = _read_verified_csv_bytes(csv_path).decode("utf-8-sig")
        with io.StringIO(verified_csv_text, newline="") as source_file:
            reader = csv.DictReader(source_file)
            _validate_headers(reader.fieldnames)
            observations: list[PublicObservation] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise PublicHistoryError(f"invalid CSV row at row {row_number}")
                timestamp = _parse_observation_timestamp(
                    row.get("date_vec"), row_number
                )
                values = {
                    indicator: _parse_optional_numeric(
                        row.get(indicator), indicator, row_number
                    )
                    for indicator in _INDICATOR_UNITS
                }
                observations.append(
                    PublicObservation(
                        timestamp=timestamp,
                        values=values,
                    )
                )
    except PublicHistoryError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise PublicHistoryError("could not read approved public CSV") from error
    return tuple(observations)
