import hashlib
import json
from pathlib import Path

import pytest

import aquaops.data.public_history as public_history


SOURCE_ID = "co-udlabs-wwtp-lpicm-2025"
CSV_RELATIVE_PATH = Path("WWTP_Langmatt") / "data" / "WWTP_Langmatt.csv"


def write_csv(path: Path, body: str, *, encoding: str = "utf-8") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding=encoding, newline="")
    return path.read_bytes()


def write_registry(
    project_root: Path, csv_bytes: bytes, *, sha256: str | None = None
) -> None:
    registry_path = project_root / "data" / "public" / "sources.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
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
                            CSV_RELATIVE_PATH.as_posix(): sha256
                            or hashlib.sha256(csv_bytes).hexdigest()
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def approved_history_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    project_root = tmp_path / "temporary-project"
    source_root = project_root / "data" / "public" / SOURCE_ID
    csv_path = source_root / CSV_RELATIVE_PATH
    monkeypatch.setattr(public_history, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(public_history, "PUBLIC_SOURCE_ROOT", source_root)
    monkeypatch.setattr(public_history, "PUBLIC_CSV_PATH", csv_path)
    return project_root, csv_path


def test_loads_verified_observations_as_immutable_utc_records(
    approved_history_project: tuple[Path, Path],
) -> None:
    project_root, csv_path = approved_history_project
    csv_bytes = write_csv(
        csv_path,
        "date_vec,nh4,cond,q\n"
        "2025-01-01 00:00:00,0.2,100,10\n"
        "2025-01-01T08:00:00+08:00,,101,11\n",
        encoding="utf-8-sig",
    )
    write_registry(project_root, csv_bytes)

    observations = public_history._load_verified_public_observations()

    assert isinstance(observations, tuple)
    assert len(observations) == 2
    assert observations[0].timestamp.isoformat() == "2025-01-01T00:00:00+00:00"
    assert observations[1].timestamp.isoformat() == "2025-01-01T00:00:00+00:00"
    assert observations[0].values == {"nh4": 0.2, "cond": 100.0, "q": 10.0}
    assert observations[1].values["nh4"] is None
    assert not hasattr(observations[0], "raw_row")
    with pytest.raises(TypeError):
        observations[0].values["nh4"] = 99.0  # type: ignore[index]


def test_rejects_hash_mismatch_before_csv_parser_executes(
    approved_history_project: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, csv_path = approved_history_project
    csv_bytes = write_csv(
        csv_path,
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    write_registry(project_root, csv_bytes, sha256="0" * 64)

    def parser_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("CSV parser must not run after a hash mismatch")

    monkeypatch.setattr(public_history.csv, "DictReader", parser_must_not_run)

    with pytest.raises(
        public_history.PublicHistoryError, match="integrity check failed"
    ):
        public_history._load_verified_public_observations()


@pytest.mark.parametrize(
    "timestamp",
    [
        "2025-01-01 00:00:00",
        "2025-01-01T00:00:00",
        "2025-01-01T00:00:00Z",
        "2025-01-01T08:00:00+08:00",
        "2025-01-01T00:00:00.1Z",
        "2025-01-01T00:00:00.123456Z",
    ],
)
def test_accepts_documented_timestamp_forms(
    approved_history_project: tuple[Path, Path], timestamp: str
) -> None:
    project_root, csv_path = approved_history_project
    csv_bytes = write_csv(csv_path, f"date_vec,nh4,cond,q\n{timestamp},0.2,100,10\n")
    write_registry(project_root, csv_bytes)

    observations = public_history._load_verified_public_observations()

    assert len(observations) == 1
    assert observations[0].timestamp.tzinfo is not None


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
def test_rejects_unsafe_timestamp_forms_without_echoing_values(
    approved_history_project: tuple[Path, Path], timestamp: str
) -> None:
    project_root, csv_path = approved_history_project
    csv_bytes = write_csv(csv_path, f"date_vec,nh4,cond,q\n{timestamp},0.2,100,10\n")
    write_registry(project_root, csv_bytes)

    with pytest.raises(
        public_history.PublicHistoryError, match="invalid timestamp at row 2"
    ) as error:
        public_history._load_verified_public_observations()

    if timestamp:
        assert timestamp not in str(error.value)


def test_validates_timestamp_before_other_row_values(
    approved_history_project: tuple[Path, Path],
) -> None:
    project_root, csv_path = approved_history_project
    csv_bytes = write_csv(
        csv_path,
        "date_vec,nh4,cond,q\nnot-a-timestamp,not-a-number,100,10\n",
    )
    write_registry(project_root, csv_bytes)

    with pytest.raises(
        public_history.PublicHistoryError, match="invalid timestamp at row 2"
    ):
        public_history._load_verified_public_observations()


def test_rejects_symbolic_linked_source_root(
    approved_history_project: tuple[Path, Path],
) -> None:
    project_root, csv_path = approved_history_project
    outside_source_root = project_root / "outside" / SOURCE_ID
    outside_csv_path = outside_source_root / CSV_RELATIVE_PATH
    csv_bytes = write_csv(
        outside_csv_path,
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    write_registry(project_root, csv_bytes)
    source_root = csv_path.parents[2]
    source_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_root.symlink_to(outside_source_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Windows does not permit test directory symlinks: {error}")

    with pytest.raises(
        public_history.PublicHistoryError, match="must not use symbolic links"
    ):
        public_history._load_verified_public_observations()


def test_rejects_symbolic_linked_source_registry(
    approved_history_project: tuple[Path, Path],
) -> None:
    project_root, csv_path = approved_history_project
    csv_bytes = write_csv(
        csv_path,
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    write_registry(project_root, csv_bytes)
    registry_path = project_root / "data" / "public" / "sources.json"
    outside_registry_path = project_root / "outside" / "sources.json"
    outside_registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.replace(outside_registry_path)
    try:
        registry_path.symlink_to(outside_registry_path)
    except OSError as error:
        pytest.skip(f"Windows does not permit test file symlinks: {error}")

    with pytest.raises(
        public_history.PublicHistoryError, match="must not use symbolic links"
    ):
        public_history._load_verified_public_observations()


def test_rejects_registry_path_marked_as_symbolic_link(
    approved_history_project: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, csv_path = approved_history_project
    csv_bytes = write_csv(
        csv_path,
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    write_registry(project_root, csv_bytes)
    registry_path = project_root / "data" / "public" / "sources.json"
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == registry_path or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    with pytest.raises(
        public_history.PublicHistoryError, match="must not use symbolic links"
    ):
        public_history._load_verified_public_observations()


@pytest.mark.parametrize(
    "registry_parent_relative_path",
    [Path("data"), Path("data") / "public"],
    ids=["data_directory", "public_directory"],
)
def test_rejects_symbolic_linked_source_registry_parent_components(
    approved_history_project: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    registry_parent_relative_path: Path,
) -> None:
    project_root, csv_path = approved_history_project
    csv_bytes = write_csv(
        csv_path,
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    write_registry(project_root, csv_bytes)
    symbolic_component = project_root / registry_parent_relative_path
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == symbolic_component or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    with pytest.raises(
        public_history.PublicHistoryError, match="must not use symbolic links"
    ):
        public_history._load_verified_public_observations()


def test_rejects_csv_path_outside_fixed_project_layout(
    approved_history_project: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, _ = approved_history_project
    outside_csv_path = project_root / "outside" / "data.csv"
    write_csv(
        outside_csv_path,
        "date_vec,nh4,cond,q\n2025-01-01T00:00:00Z,0.2,100,10\n",
    )
    monkeypatch.setattr(public_history, "PUBLIC_CSV_PATH", outside_csv_path)

    with pytest.raises(
        public_history.PublicHistoryError,
        match="public CSV path must remain within the approved public source root",
    ):
        public_history._load_verified_public_observations()
