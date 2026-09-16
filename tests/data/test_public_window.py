from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

import aquaops.data.public_history as public_history
import aquaops.data.public_window as public_window_module
from aquaops.data.public_window import PublicWindowError, summarize_public_window


SOURCE_ID = "co-udlabs-wwtp-lpicm-2025"


@pytest.fixture
def approved_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source_root = tmp_path / "data" / "public" / SOURCE_ID
    csv_path = source_root / "WWTP_Langmatt" / "data" / "WWTP_Langmatt.csv"
    monkeypatch.setattr(public_history, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(public_history, "PUBLIC_SOURCE_ROOT", source_root)
    monkeypatch.setattr(public_history, "PUBLIC_CSV_PATH", csv_path)
    monkeypatch.setattr(
        public_history,
        "_load_registered_csv_sha256",
        lambda: hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    )
    return source_root


def write_csv(path: Path, body: str, *, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding=encoding, newline="")
    return path


def source_csv_path(source_root: Path) -> Path:
    return source_root / "WWTP_Langmatt" / "data" / "WWTP_Langmatt.csv"


def test_summarizes_window_without_returning_raw_observations(
    approved_source_root: Path,
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n"
        "2025-01-01T00:00:00Z,0.1,100,10\n"
        "2025-01-01T00:30:00Z,0.2,101,11\n"
        "2025-01-01T01:00:00Z,0.4,102,12\n"
        "2025-01-01T02:00:00Z,,103,13\n",
    )

    summary = summarize_public_window(
        indicator="nh4", hours=2, end_at="2025-01-01T02:00:00Z"
    )

    assert asdict(summary) == {
        "source_id": SOURCE_ID,
        "source_url": "https://zenodo.org/records/15285089",
        "source_version": "v1.0.0 (2025-04-26)",
        "indicator": "nh4",
        "unit": "mg/L",
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-01T02:00:00Z",
        "row_count": 3,
        "null_count": 1,
        "minimum": 0.2,
        "maximum": 0.4,
        "mean": 0.30000000000000004,
    }
    assert not {"values", "rows", "timestamps", "samples"} & set(asdict(summary))


def test_aggregates_shared_history_loader_without_reading_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = (
        public_history.PublicObservation(
            timestamp=datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc),
            values=MappingProxyType({"nh4": 0.2, "cond": 100.0, "q": 10.0}),
        ),
        public_history.PublicObservation(
            timestamp=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            values=MappingProxyType({"nh4": None, "cond": 101.0, "q": 11.0}),
        ),
    )
    calls = 0

    def load_history() -> tuple[public_history.PublicObservation, ...]:
        nonlocal calls
        calls += 1
        return observations

    monkeypatch.setattr(
        public_window_module, "_load_verified_public_observations", load_history
    )

    summary = summarize_public_window("nh4", 1, "2025-01-01T01:00:00Z")

    assert calls == 1
    assert (summary.row_count, summary.null_count, summary.minimum, summary.mean) == (
        2,
        1,
        0.2,
        0.2,
    )


@pytest.mark.parametrize(
    ("hours", "end_at", "expected_start", "expected_count"),
    [
        (1, "2025-01-01T01:00:00Z", "2025-01-01T00:00:00Z", 2),
        (168, "2025-01-08T00:00:00Z", "2025-01-01T00:00:00Z", 3),
    ],
)
def test_applies_open_start_closed_end_window_boundaries(
    approved_source_root: Path,
    hours: int,
    end_at: str,
    expected_start: str,
    expected_count: int,
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n"
        "2025-01-01T00:00:00Z,0.2,100,10\n"
        "2025-01-01T00:00:01Z,0.3,101,11\n"
        "2025-01-01T01:00:00Z,0.4,102,12\n"
        "2025-01-08T00:00:00Z,0.5,103,13\n",
    )

    summary = summarize_public_window("nh4", hours, end_at)

    assert summary.start == expected_start
    assert summary.end == end_at
    assert summary.row_count == expected_count


@pytest.mark.parametrize(
    ("indicator", "hours", "end_at", "message"),
    [
        ("temperature", 1, "2025-01-01T00:00:00Z", "unsupported public indicator"),
        ("nh4", 0, "2025-01-01T00:00:00Z", "hours must be an integer from 1 to 168"),
        ("nh4", 169, "2025-01-01T00:00:00Z", "hours must be an integer from 1 to 168"),
        ("nh4", True, "2025-01-01T00:00:00Z", "hours must be an integer from 1 to 168"),
        ("nh4", 1, "2025-01-01T00:00:00+00:00", "end_at must be canonical UTC Z"),
        ("nh4", 1, "2025-01-01T00:00:00", "end_at must be canonical UTC Z"),
        ("nh4", 1, "not-a-timestamp", "end_at must be canonical UTC Z"),
    ],
)
def test_rejects_invalid_query_inputs(
    approved_source_root: Path,
    indicator: str,
    hours: object,
    end_at: str,
    message: str,
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )

    with pytest.raises(PublicWindowError, match=message):
        summarize_public_window(indicator, hours, end_at)  # type: ignore[arg-type]


def test_reads_utf8_bom_csv(approved_source_root: Path) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
        encoding="utf-8-sig",
    )

    summary = summarize_public_window("q", 1, "2025-01-01T00:00:00Z")

    assert summary.row_count == 1
    assert summary.mean == 10.0


def test_loads_registered_hash_from_temp_project_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "temporary-project"
    source_root = project_root / "data" / "public" / SOURCE_ID
    csv_path = source_csv_path(source_root)
    csv_bytes = (
        "date_vec,nh4,cond,q\n"
        "2025-01-01T00:00:00Z,0.2,100,10\n"
        "2025-01-01T00:30:00Z,0.4,101,11\n"
    ).encode("utf-8-sig")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_bytes)
    (project_root / "data" / "public" / "sources.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "source_id": SOURCE_ID,
                        "title": "Synthetic approved source",
                        "source_url": "https://zenodo.org/records/15285089",
                        "source_kind": "dataset",
                        "license_name": "CC0-1.0",
                        "license_status": "verified",
                        "source_version": "v1.0.0 (2025-04-26)",
                        "access_mode": "local_download_only",
                        "repository_commit_allowed": False,
                        "attribution": "Synthetic test attribution",
                        "license_evidence": "test evidence",
                        "download_url": "https://example.invalid/archive.zip",
                        "archive_md5": "0123456789abcdef0123456789abcdef",
                        "permitted_paths": ["WWTP_Langmatt/**"],
                        "verified_file_sha256": {
                            "WWTP_Langmatt/data/WWTP_Langmatt.csv": (
                                hashlib.sha256(csv_bytes).hexdigest()
                            )
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(public_history, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(public_history, "PUBLIC_SOURCE_ROOT", source_root)
    monkeypatch.setattr(public_history, "PUBLIC_CSV_PATH", csv_path)

    summary = summarize_public_window("nh4", 1, "2025-01-01T00:30:00Z")

    assert summary.row_count == 2
    assert summary.minimum == 0.2
    assert summary.maximum == 0.4
    assert summary.mean == pytest.approx(0.3)


def test_rejects_public_csv_when_registered_content_hash_does_not_match(
    approved_source_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    monkeypatch.setattr(
        public_history,
        "_load_registered_csv_sha256",
        lambda: "0" * 64,
    )

    with pytest.raises(PublicWindowError, match="integrity check failed"):
        summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")


@pytest.mark.parametrize("registered_hash", [None, "not-a-sha256"])
def test_rejects_missing_or_invalid_registered_csv_hash(
    approved_source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered_hash: object,
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    monkeypatch.setattr(
        public_history,
        "_load_registered_csv_sha256",
        lambda: registered_hash,
    )

    with pytest.raises(PublicWindowError, match="integrity metadata is unavailable"):
        summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")


def test_rejects_window_without_public_observations(
    approved_source_root: Path,
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )

    with pytest.raises(
        PublicWindowError, match="no public observations cover the requested window"
    ):
        summarize_public_window("nh4", 1, "2025-01-02T00:00:00Z")


def test_rejects_missing_required_columns(approved_source_root: Path) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,q\n2025-01-01T00:00:00Z,0.2,10\n",
    )

    with pytest.raises(PublicWindowError, match="missing required columns: cond"):
        summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")


@pytest.mark.parametrize(
    "timestamp",
    ["2025-01-01 00:00:00", "2025-01-01T00:00:00"],
)
def test_interprets_documented_utc_naive_csv_timestamps(
    approved_source_root: Path, timestamp: str
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        f"date_vec,nh4,cond,q\n{timestamp},0.2,100,10\n",
    )

    summary = summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")

    assert summary.row_count == 1
    assert summary.start == "2024-12-31T23:00:00Z"
    assert summary.end == "2025-01-01T00:00:00Z"


@pytest.mark.parametrize(
    "timestamp",
    [
        "2025-13-40T00:00:00",
        "not-a-timestamp",
        "2025-01-01",
        "2025-01-01T00:00",
        "2025-01-01T00",
        "2025-01-01T00:00:00.1234567Z",
        "",
    ],
)
def test_rejects_invalid_csv_timestamps_without_echoing_values(
    approved_source_root: Path, timestamp: str
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        f"date_vec,nh4,cond,q\n{timestamp},0.2,100,10\n",
    )

    with pytest.raises(PublicWindowError, match="invalid timestamp at row 2") as error:
        summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")

    if timestamp:
        assert timestamp not in str(error.value)


def test_normalizes_offset_csv_timestamps_to_utc(approved_source_root: Path) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n2025-01-01T08:00:00+08:00,0.2,100,10\n",
    )

    summary = summarize_public_window("cond", 1, "2025-01-01T00:00:00Z")

    assert summary.row_count == 1
    assert summary.mean == 100.0


def test_accepts_fractional_second_observation_timestamp(
    approved_source_root: Path,
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n2020-06-11T23:55:00.123Z,0.2,100,10\n",
    )

    summary = summarize_public_window("nh4", 1, "2020-06-11T23:55:00.123000Z")

    assert summary.row_count == 1
    assert summary.start == "2020-06-11T22:55:00.123000Z"
    assert summary.end == "2020-06-11T23:55:00.123000Z"


def test_accepts_fractional_second_offset_observation_timestamp(
    approved_source_root: Path,
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n2020-06-12T07:55:00.123+08:00,0.2,100,10\n",
    )

    summary = summarize_public_window("nh4", 1, "2020-06-11T23:55:00.123000Z")

    assert summary.row_count == 1
    assert summary.start == "2020-06-11T22:55:00.123000Z"
    assert summary.end == "2020-06-11T23:55:00.123000Z"


def test_rejects_nonempty_invalid_numeric_values_without_echoing_values(
    approved_source_root: Path,
) -> None:
    invalid_value = "not-a-number"
    write_csv(
        source_csv_path(approved_source_root),
        f"date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,{invalid_value},100,10\n",
    )

    with pytest.raises(
        PublicWindowError, match="invalid numeric value for nh4 at row 2"
    ) as error:
        summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")

    assert invalid_value not in str(error.value)


@pytest.mark.parametrize("unsafe_location", ["private", "outside"])
def test_rejects_private_or_outside_resolved_csv_path(
    approved_source_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_location: str,
) -> None:
    if unsafe_location == "private":
        unsafe_path = tmp_path / "data" / "private" / "leak.csv"
    else:
        unsafe_path = tmp_path / "outside" / "data.csv"
    write_csv(
        unsafe_path,
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    monkeypatch.setattr(public_history, "PUBLIC_CSV_PATH", unsafe_path)

    with pytest.raises(
        PublicWindowError,
        match="public CSV path must remain within the approved public source root",
    ):
        summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")


def test_rejects_source_path_not_bound_to_project_public_layout(
    approved_source_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unregistered_source_root = tmp_path / "alternate-public" / SOURCE_ID
    unregistered_csv_path = source_csv_path(unregistered_source_root)
    write_csv(
        unregistered_csv_path,
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    monkeypatch.setattr(public_history, "PUBLIC_SOURCE_ROOT", unregistered_source_root)
    monkeypatch.setattr(public_history, "PUBLIC_CSV_PATH", unregistered_csv_path)

    with pytest.raises(
        PublicWindowError, match="must match the controlled public source layout"
    ):
        summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")


def test_rejects_symbolic_linked_registered_source_root(
    approved_source_root: Path, tmp_path: Path
) -> None:
    outside_source_root = tmp_path / "outside" / SOURCE_ID
    write_csv(
        source_csv_path(outside_source_root),
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    approved_source_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        approved_source_root.symlink_to(outside_source_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Windows does not permit test directory symlinks: {error}")

    with pytest.raises(PublicWindowError, match="must not use symbolic links"):
        summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")


def test_rejects_missing_registered_public_csv(approved_source_root: Path) -> None:
    approved_source_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(
        PublicWindowError, match="approved public CSV file does not exist"
    ):
        summarize_public_window("nh4", 1, "2025-01-01T00:00:00Z")


def test_returns_none_statistics_when_selected_indicator_is_all_null(
    approved_source_root: Path,
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n"
        "2025-01-01T00:00:00Z,,100,10\n"
        "2025-01-01T00:30:00Z,,101,11\n",
    )

    summary = summarize_public_window("nh4", 1, "2025-01-01T00:30:00Z")

    assert summary.row_count == 2
    assert summary.null_count == 2
    assert (summary.minimum, summary.maximum, summary.mean) == (None, None, None)


def test_rejects_finite_values_when_window_statistics_overflow(
    approved_source_root: Path,
) -> None:
    write_csv(
        source_csv_path(approved_source_root),
        "date_vec,nh4,cond,q\n"
        "2025-01-01T00:00:00Z,1e308,100,10\n"
        "2025-01-01T00:30:00Z,1e308,101,11\n",
    )

    with pytest.raises(PublicWindowError, match="public window statistics overflow"):
        summarize_public_window("nh4", 1, "2025-01-01T00:30:00Z")
