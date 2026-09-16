import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from zipfile import ZipFile, ZipInfo

import pytest

import aquaops.rag.acquire as acquire_module
from aquaops.rag.acquire import (
    PublicDataAcquisitionError,
    acquire_public_source,
    calculate_md5,
    extract_verified_archive,
)
from aquaops.rag.source_registry import PublicSource


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_SCRIPT = PROJECT_ROOT / "scripts" / "acquire_public_source.py"


def make_zip(path: Path, members: dict[str, str]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def make_zip_with_unsafe_member(path: Path, unsafe_member: str) -> Path:
    with ZipFile(path, "w") as archive:
        archive.writestr("WWTP_Langmatt/data/WWTP_Langmatt.csv", "allowed")
        member = ZipInfo("placeholder")
        member.filename = unsafe_member
        archive.writestr(member, "unsafe")
    return path


def make_local_source(archive_md5: str) -> PublicSource:
    return PublicSource.model_validate(
        {
            "source_id": "local-source",
            "title": "Local Source",
            "source_url": "https://example.invalid/source",
            "source_kind": "dataset",
            "license_name": "CC0-1.0",
            "license_status": "verified",
            "source_version": "v1",
            "access_mode": "local_download_only",
            "repository_commit_allowed": False,
            "attribution": "example attribution",
            "license_evidence": "common/license.txt",
            "download_url": "https://example.invalid/archive.zip",
            "archive_md5": archive_md5,
            "permitted_paths": ["WWTP_Langmatt/**"],
            "verified_file_sha256": {
                "WWTP_Langmatt/data/WWTP_Langmatt.csv": "0" * 64,
            },
        }
    )


def write_registry(path: Path, source: dict[str, object]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "sources": [source]}), encoding="utf-8"
    )
    return path


def local_source_payload(archive_md5: str) -> dict[str, object]:
    return {
        "source_id": "local-source",
        "title": "Local Source",
        "source_url": "https://example.invalid/source",
        "source_kind": "dataset",
        "license_name": "CC0-1.0",
        "license_status": "verified",
        "source_version": "v1",
        "access_mode": "local_download_only",
        "repository_commit_allowed": False,
        "attribution": "example attribution",
        "license_evidence": "common/license.txt",
        "download_url": "https://example.invalid/archive.zip",
        "archive_md5": archive_md5,
        "permitted_paths": ["WWTP_Langmatt/**"],
        "verified_file_sha256": {
            "WWTP_Langmatt/data/WWTP_Langmatt.csv": "0" * 64,
        },
    }


def copy_archive_downloader(archive: Path) -> Callable[[str, Path], None]:
    def download(_: str, destination: Path) -> None:
        destination.write_bytes(archive.read_bytes())

    return download


def load_cli_module() -> object:
    module_name = "aquaops_acquire_cli_test"
    spec = importlib.util.spec_from_file_location(module_name, CLI_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_calculate_md5_reads_file_in_chunks(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"aquaops-public-data")

    assert calculate_md5(path) == hashlib.md5(b"aquaops-public-data").hexdigest()


def test_extracts_only_registry_whitelisted_members(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "source.zip",
        {
            "WWTP_Langmatt/data/WWTP_Langmatt.csv": "timestamp,nh4\n",
            "WWTP_Langmatt/documentation/README.md": "public documentation",
            "Rainfall_Lenzburg/data/rain.csv": "not permitted",
            "common/license.txt": "not permitted",
        },
    )
    destination = tmp_path / "acquired"
    source = make_local_source(calculate_md5(archive))

    extracted = extract_verified_archive(archive, destination, source)

    assert extracted == [
        Path("WWTP_Langmatt/data/WWTP_Langmatt.csv"),
        Path("WWTP_Langmatt/documentation/README.md"),
    ]
    assert (destination / "WWTP_Langmatt/data/WWTP_Langmatt.csv").is_file()
    assert (destination / "WWTP_Langmatt/documentation/README.md").is_file()
    assert not (destination / "Rainfall_Lenzburg").exists()
    assert not (destination / "common").exists()


@pytest.mark.parametrize(
    "unsafe_member",
    [
        "../outside.csv",
        "WWTP_Langmatt\\outside.csv",
        "/WWTP_Langmatt/outside.csv",
        "WWTP_Langmatt/./outside.csv",
        "C:/outside.csv",
    ],
)
def test_rejects_any_unsafe_archive_member_before_writing_target(
    tmp_path: Path, unsafe_member: str
) -> None:
    archive = make_zip_with_unsafe_member(tmp_path / "unsafe.zip", unsafe_member)
    destination = tmp_path / "acquired"
    source = make_local_source(calculate_md5(archive))

    with pytest.raises(PublicDataAcquisitionError, match="unsafe archive member"):
        extract_verified_archive(archive, destination, source)

    assert not destination.exists()


def test_rejects_nonempty_destination(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "source.zip",
        {"WWTP_Langmatt/data/WWTP_Langmatt.csv": "allowed"},
    )
    destination = tmp_path / "acquired"
    destination.mkdir()
    existing = destination / "existing.txt"
    existing.write_text("retain", encoding="utf-8")
    source = make_local_source(calculate_md5(archive))

    with pytest.raises(PublicDataAcquisitionError, match="empty destination"):
        extract_verified_archive(archive, destination, source)

    assert existing.read_text(encoding="utf-8") == "retain"


def test_hash_mismatch_never_extracts_downloaded_archive(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "source.zip",
        {"WWTP_Langmatt/data/WWTP_Langmatt.csv": "allowed"},
    )
    registry_path = write_registry(
        tmp_path / "sources.json", local_source_payload("0" * 32)
    )
    destination = tmp_path / "acquired"

    with pytest.raises(PublicDataAcquisitionError, match="checksum mismatch"):
        acquire_public_source(
            registry_path,
            "local-source",
            destination,
            downloader=copy_archive_downloader(archive),
        )

    assert not destination.exists()


def test_downloader_failure_is_wrapped_without_writing_destination(
    tmp_path: Path,
) -> None:
    registry_path = write_registry(
        tmp_path / "sources.json", local_source_payload("0" * 32)
    )
    destination = tmp_path / "acquired"

    def fail_download(_: str, __: Path) -> None:
        raise URLError("offline")

    with pytest.raises(PublicDataAcquisitionError, match="download failed"):
        acquire_public_source(
            registry_path,
            "local-source",
            destination,
            downloader=fail_download,
        )

    assert not destination.exists()


def test_downloader_that_does_not_create_archive_is_wrapped(tmp_path: Path) -> None:
    registry_path = write_registry(
        tmp_path / "sources.json", local_source_payload("0" * 32)
    )
    destination = tmp_path / "acquired"

    def no_output(_: str, __: Path) -> None:
        return None

    with pytest.raises(PublicDataAcquisitionError, match="download failed"):
        acquire_public_source(
            registry_path,
            "local-source",
            destination,
            downloader=no_output,
        )

    assert not destination.exists()


def test_reference_only_source_is_rejected_without_downloading(tmp_path: Path) -> None:
    source = local_source_payload("0" * 32)
    source["access_mode"] = "reference_only"
    source["download_url"] = None
    source["archive_md5"] = None
    source["license_evidence"] = None
    registry_path = write_registry(tmp_path / "sources.json", source)
    destination = tmp_path / "acquired"

    def fail_download(_: str, __: Path) -> None:
        raise AssertionError("reference-only source must not invoke downloader")

    with pytest.raises(PublicDataAcquisitionError, match="not eligible"):
        acquire_public_source(
            registry_path,
            "local-source",
            destination,
            downloader=fail_download,
        )

    assert not destination.exists()


def test_writes_public_provenance_receipt_after_verified_acquisition(
    tmp_path: Path,
) -> None:
    archive = make_zip(
        tmp_path / "source.zip",
        {
            "WWTP_Langmatt/documentation/README.md": "allowed documentation",
            "Rainfall_Lenzburg/data/rain.csv": "not permitted",
            "common/license.txt": "not permitted",
            "WWTP_Langmatt/data/WWTP_Langmatt.csv": "allowed data",
        },
    )
    expected_md5 = calculate_md5(archive)
    registry_path = write_registry(
        tmp_path / "sources.json", local_source_payload(expected_md5)
    )
    destination = tmp_path / "acquired"

    receipt = acquire_public_source(
        registry_path,
        "local-source",
        destination,
        downloader=copy_archive_downloader(archive),
    )

    receipt_path = destination / ".acquisition.json"
    assert receipt_path.is_file()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert receipt == {
        "source_id": "local-source",
        "source_version": "v1",
        "download_url": "https://example.invalid/archive.zip",
        "archive_md5": expected_md5,
        "license_evidence": "common/license.txt",
        "permitted_paths": ["WWTP_Langmatt/**"],
        "extracted_paths": [
            "WWTP_Langmatt/data/WWTP_Langmatt.csv",
            "WWTP_Langmatt/documentation/README.md",
        ],
    }
    assert (destination / "WWTP_Langmatt/data/WWTP_Langmatt.csv").is_file()
    assert (destination / "WWTP_Langmatt/documentation/README.md").is_file()
    assert all("Rainfall_Lenzburg" not in path for path in receipt["extracted_paths"])
    assert all("common" not in path for path in receipt["extracted_paths"])


def test_retries_transient_permission_error_during_atomic_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = make_zip(
        tmp_path / "source.zip",
        {"WWTP_Langmatt/data/WWTP_Langmatt.csv": "allowed data"},
    )
    registry_path = write_registry(
        tmp_path / "sources.json", local_source_payload(calculate_md5(archive))
    )
    destination = tmp_path / "acquired"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(self: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if self.name.startswith("aquaops-acquire-") and attempts <= 2:
            raise PermissionError(5, "access denied")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    receipt = acquire_public_source(
        registry_path,
        "local-source",
        destination,
        downloader=copy_archive_downloader(archive),
    )

    assert attempts == 3
    assert receipt["extracted_paths"] == ["WWTP_Langmatt/data/WWTP_Langmatt.csv"]
    assert (destination / "WWTP_Langmatt/data/WWTP_Langmatt.csv").is_file()


def test_receipt_write_failure_does_not_commit_extracted_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = make_zip(
        tmp_path / "source.zip",
        {"WWTP_Langmatt/data/WWTP_Langmatt.csv": "allowed"},
    )
    registry_path = write_registry(
        tmp_path / "sources.json", local_source_payload(calculate_md5(archive))
    )
    destination = tmp_path / "acquired"
    original_write_text = Path.write_text

    def fail_receipt_write(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == ".acquisition.json":
            raise OSError("receipt storage unavailable")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_receipt_write)

    with pytest.raises(PublicDataAcquisitionError, match="acquisition receipt"):
        acquire_public_source(
            registry_path,
            "local-source",
            destination,
            downloader=copy_archive_downloader(archive),
        )

    assert not destination.exists()


def test_commit_failure_does_not_leave_data_in_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = make_zip(
        tmp_path / "source.zip",
        {"WWTP_Langmatt/data/WWTP_Langmatt.csv": "allowed"},
    )
    registry_path = write_registry(
        tmp_path / "sources.json", local_source_payload(calculate_md5(archive))
    )
    destination = tmp_path / "acquired"

    def fail_commit(_: Path, __: Path) -> None:
        raise OSError("rename unavailable")

    monkeypatch.setattr(acquire_module, "_commit_staged_directory", fail_commit)

    with pytest.raises(PublicDataAcquisitionError, match="commit staged public data"):
        acquire_public_source(
            registry_path,
            "local-source",
            destination,
            downloader=copy_archive_downloader(archive),
        )

    assert not destination.exists()


def test_acquisition_rejects_existing_empty_destination(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "source.zip",
        {"WWTP_Langmatt/data/WWTP_Langmatt.csv": "allowed"},
    )
    registry_path = write_registry(
        tmp_path / "sources.json", local_source_payload(calculate_md5(archive))
    )
    destination = tmp_path / "acquired"
    destination.mkdir()

    with pytest.raises(PublicDataAcquisitionError, match="must not already exist"):
        acquire_public_source(
            registry_path,
            "local-source",
            destination,
            downloader=copy_archive_downloader(archive),
        )

    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_cli_resolves_relative_destination_from_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = load_cli_module()
    received: dict[str, object] = {}

    def fake_acquire(
        registry_path: Path, source_id: str, destination: Path
    ) -> dict[str, object]:
        received.update(
            {
                "registry_path": registry_path,
                "source_id": source_id,
                "destination": destination,
            }
        )
        return {"source_id": source_id}

    monkeypatch.setattr(acquire_module, "acquire_public_source", fake_acquire)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acquire_public_source.py",
            "local-source",
            "--destination",
            "output/public-data",
        ],
    )

    assert cli.main() == 0
    assert received == {
        "registry_path": PROJECT_ROOT / "data" / "public" / "sources.json",
        "source_id": "local-source",
        "destination": PROJECT_ROOT / "output" / "public-data",
    }
