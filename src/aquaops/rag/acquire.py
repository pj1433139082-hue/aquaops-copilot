"""Controlled acquisition of registered public datasets.

The registry, rather than a caller-supplied URL, defines every allowed download.
Archives are checksum-verified and only registry-approved member paths are extracted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import stat
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile, ZipInfo

from aquaops.rag.source_registry import PublicSource, load_public_source_registry


_HASH_BLOCK_SIZE = 1024 * 1024
_ATOMIC_COMMIT_ATTEMPTS = 4
_ATOMIC_COMMIT_RETRY_DELAY_SECONDS = 0.05
Downloader = Callable[[str, Path], None]


class PublicDataAcquisitionError(ValueError):
    """Raised when a public-data acquisition violates a registry safety boundary."""


def calculate_md5(path: Path) -> str:
    """Return a file's lowercase MD5 digest while reading it in bounded chunks."""
    try:
        digest = hashlib.md5(usedforsecurity=False)
    except TypeError:
        digest = hashlib.md5()

    with path.open("rb") as file_handle:
        while chunk := file_handle.read(_HASH_BLOCK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_empty_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise PublicDataAcquisitionError("destination must not be a symbolic link")
    if destination.exists() and not destination.is_dir():
        raise PublicDataAcquisitionError("destination must be a directory")
    if destination.exists() and any(destination.iterdir()):
        raise PublicDataAcquisitionError("destination must be an empty destination")


def _validate_new_acquisition_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise PublicDataAcquisitionError("destination must not be a symbolic link")
    if destination.exists():
        raise PublicDataAcquisitionError(
            "destination must not already exist for an atomic acquisition"
        )


def _validated_archive_name(member: ZipInfo) -> str:
    # On Windows, ZipFile normalizes ``filename`` backslashes to slashes. Keep
    # the original central-directory name so a malicious backslash is rejected.
    name = member.orig_filename
    if not name or "\\" in name or name.startswith("/") or "\x00" in name:
        raise PublicDataAcquisitionError(f"unsafe archive member: {name!r}")

    name_without_directory_suffix = name[:-1] if member.is_dir() else name
    parts = name_without_directory_suffix.split("/")
    if (
        not name_without_directory_suffix
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
    ):
        raise PublicDataAcquisitionError(f"unsafe archive member: {name!r}")

    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise PublicDataAcquisitionError(f"unsafe archive member: {name!r}")

    return name_without_directory_suffix


def _is_permitted_member(name: str, permitted_paths: list[str]) -> bool:
    for permitted_path in permitted_paths:
        if permitted_path.endswith("/**"):
            prefix = permitted_path[:-3]
            if name.startswith(f"{prefix}/"):
                return True
        elif name == permitted_path:
            return True
    return False


def _create_sibling_staging_directory(destination: Path, prefix: str) -> Path:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=destination.parent))
    except OSError as error:
        raise PublicDataAcquisitionError(
            "could not create public-data staging directory"
        ) from error


def _extract_allowed_members(
    archive_path: Path, staging_directory: Path, source: PublicSource
) -> list[Path]:
    try:
        with ZipFile(archive_path) as archive:
            members = [
                (member, _validated_archive_name(member))
                for member in archive.infolist()
            ]
            permitted_members = [
                (member, name)
                for member, name in members
                if not member.is_dir()
                and _is_permitted_member(name, source.permitted_paths)
            ]
            if not permitted_members:
                raise PublicDataAcquisitionError(
                    "archive contains no members permitted by the registry"
                )

            extracted_paths: list[Path] = []
            for member, name in permitted_members:
                relative_path = Path(*name.split("/"))
                output_path = staging_directory / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.open(member) as archive_member,
                    output_path.open("wb") as output_file,
                ):
                    shutil.copyfileobj(archive_member, output_file)
                extracted_paths.append(relative_path)
            return extracted_paths
    except PublicDataAcquisitionError:
        raise
    except BadZipFile as error:
        raise PublicDataAcquisitionError(
            "downloaded archive is not a valid ZIP"
        ) from error
    except Exception as error:
        raise PublicDataAcquisitionError(
            "could not extract verified archive"
        ) from error


def extract_verified_archive(
    archive_path: Path, destination: Path, source: PublicSource
) -> list[Path]:
    """Extract only the source allowlist after validating every ZIP member.

    This low-level helper preserves support for an empty existing directory. The
    end-to-end acquisition path uses a stricter new-destination policy so its final
    commit can be one directory rename after both data and receipt are complete.
    """
    _validate_empty_destination(destination)
    staging_directory = _create_sibling_staging_directory(
        destination, "aquaops-extract-"
    )
    try:
        extracted_paths = _extract_allowed_members(
            archive_path, staging_directory, source
        )
        _validate_empty_destination(destination)
        try:
            destination.mkdir(parents=True, exist_ok=True)
            for staged_path in staging_directory.iterdir():
                staged_path.replace(destination / staged_path.name)
        except OSError as error:
            raise PublicDataAcquisitionError(
                "could not materialize verified archive"
            ) from error
        return extracted_paths
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)


def _download_archive(download_url: str, destination: Path) -> None:
    try:
        with (
            urlopen(download_url, timeout=30) as response,
            destination.open("wb") as output_file,
        ):
            shutil.copyfileobj(response, output_file)
    except OSError as error:
        raise PublicDataAcquisitionError("download failed") from error


def _load_eligible_source(registry_path: Path, source_id: str) -> PublicSource:
    sources = load_public_source_registry(registry_path)
    source = next((item for item in sources if item.source_id == source_id), None)
    if source is None:
        raise PublicDataAcquisitionError(f"unknown registered source: {source_id}")
    if source.access_mode != "local_download_only":
        raise PublicDataAcquisitionError(
            f"source is not eligible for local acquisition: {source_id}"
        )
    if (
        source.download_url is None
        or source.archive_md5 is None
        or source.license_evidence is None
        or not source.license_evidence.strip()
        or not source.permitted_paths
    ):
        raise PublicDataAcquisitionError(
            f"source lacks verified acquisition metadata: {source_id}"
        )
    return source


def _receipt_for(
    source: PublicSource, extracted_paths: list[Path]
) -> dict[str, object]:
    if (
        source.download_url is None
        or source.archive_md5 is None
        or source.license_evidence is None
    ):
        raise PublicDataAcquisitionError(
            f"source lacks verified acquisition metadata: {source.source_id}"
        )
    return {
        "source_id": source.source_id,
        "source_version": source.source_version,
        "download_url": str(source.download_url),
        "archive_md5": source.archive_md5,
        "license_evidence": source.license_evidence,
        "permitted_paths": source.permitted_paths,
        "extracted_paths": sorted(path.as_posix() for path in extracted_paths),
    }


def _write_receipt(staging_directory: Path, receipt: dict[str, object]) -> None:
    try:
        (staging_directory / ".acquisition.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as error:
        raise PublicDataAcquisitionError(
            "could not write acquisition receipt"
        ) from error


def _commit_staged_directory(staging_directory: Path, destination: Path) -> None:
    if destination.exists():
        raise PublicDataAcquisitionError(
            "destination appeared before staged public data could be committed"
        )
    last_error: PermissionError | None = None
    for attempt in range(_ATOMIC_COMMIT_ATTEMPTS):
        if destination.exists():
            raise PublicDataAcquisitionError(
                "destination appeared before staged public data could be committed"
            )
        try:
            staging_directory.replace(destination)
            return
        except PermissionError as error:
            last_error = error
            if attempt + 1 < _ATOMIC_COMMIT_ATTEMPTS:
                time.sleep(_ATOMIC_COMMIT_RETRY_DELAY_SECONDS * (attempt + 1))
        except OSError as error:
            raise PublicDataAcquisitionError(
                "could not commit staged public data"
            ) from error
    raise PublicDataAcquisitionError("could not commit staged public data") from last_error


def acquire_public_source(
    registry_path: Path,
    source_id: str,
    destination: Path,
    *,
    downloader: Downloader | None = None,
) -> dict[str, object]:
    """Download and atomically acquire one registered public source.

    The destination must not exist. This lets a sibling staging directory be renamed
    into place only after allowlisted data and its provenance receipt are both ready.
    """
    source = _load_eligible_source(registry_path, source_id)
    _validate_new_acquisition_destination(destination)
    transfer = downloader or _download_archive

    try:
        with tempfile.TemporaryDirectory(
            prefix="aquaops-download-"
        ) as temporary_directory:
            archive_path = Path(temporary_directory) / "source.zip"
            try:
                transfer(str(source.download_url), archive_path)
                if not archive_path.is_file():
                    raise FileNotFoundError("downloader did not create the archive")
                actual_md5 = calculate_md5(archive_path)
            except PublicDataAcquisitionError:
                raise
            except Exception as error:
                raise PublicDataAcquisitionError(
                    "download failed or downloaded archive could not be read"
                ) from error

            if not hmac.compare_digest(actual_md5, source.archive_md5):
                raise PublicDataAcquisitionError(
                    "downloaded archive checksum mismatch; extraction refused"
                )

            staging_directory = _create_sibling_staging_directory(
                destination, "aquaops-acquire-"
            )
            try:
                extracted_paths = _extract_allowed_members(
                    archive_path, staging_directory, source
                )
                receipt = _receipt_for(source, extracted_paths)
                _write_receipt(staging_directory, receipt)
                try:
                    _commit_staged_directory(staging_directory, destination)
                except PublicDataAcquisitionError:
                    raise
                except OSError as error:
                    raise PublicDataAcquisitionError(
                        "could not commit staged public data"
                    ) from error
                return receipt
            finally:
                shutil.rmtree(staging_directory, ignore_errors=True)
    except PublicDataAcquisitionError:
        raise
    except OSError as error:
        raise PublicDataAcquisitionError(
            "download failed or temporary archive could not be read"
        ) from error
