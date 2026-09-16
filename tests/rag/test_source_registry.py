import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from aquaops.rag.source_registry import load_public_source_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ATTRIBUTION = (
    "Rieckermann, J. (2025): Electrical conductivity and ammonium monitoring "
    "data from WWTP Langmatt inlet, Lenzburg (Version 1) [Data set]. Zenodo. "
    "https://doi.org/10.5281/zenodo.15285089"
)
EXPECTED_SOURCE_IDS = {
    "co-udlabs-wwtp-lpicm-2025",
    "iwa-benchmark-simulation-models",
    "dhi-waterbench-inflow",
    "mee-gb-18918-2025-amendment",
    "epa-nutrient-control-design-manual",
}
EXPECTED_WWTP_CSV_SHA256 = (
    "5757cd15831d8d2e742a49e130d07d771460a9aa0d9993062116c08bd225eb02"
)


def test_gitignore_excludes_private_and_acquired_public_data() -> None:
    ignored_paths = (
        "data/private/example.csv",
        "data/public/raw/example.csv",
        "data/public/derived/example.csv",
        "data/public/co-udlabs-wwtp-lpicm-2025/sample.csv",
        "src/aquaops/rag/__pycache__/source_registry.cpython-312.pyc",
    )
    retained_paths = (
        "data/public/README.md",
        "data/public/sources.json",
        "src/aquaops/rag/source_registry.py",
        "tests/rag/test_source_registry.py",
    )

    git_root = next(
        (
            candidate
            for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents)
            if (candidate / ".git").exists()
        ),
        None,
    )

    if git_root is not None:

        def is_ignored(path: str) -> bool:
            result = subprocess.run(
                ["git", "check-ignore", "--no-index", "--quiet", "--", path],
                cwd=PROJECT_ROOT,
                check=False,
            )
            return result.returncode == 0

    else:
        # An extracted public staging tree deliberately has no Git metadata.
        # Keep the isolation check useful there by validating the explicit
        # denylist/allowlist entries in the copied .gitignore.
        patterns = {
            line.strip()
            for line in (PROJECT_ROOT / ".gitignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        assert "data/private/" in patterns
        assert "data/public/*" in patterns
        assert "!data/public/README.md" in patterns
        assert "!data/public/sources.json" in patterns
        assert "__pycache__/" in patterns

        def is_ignored(path: str) -> bool:
            normalized = path.replace("\\", "/")
            if normalized.startswith("data/private/"):
                return True
            if normalized.startswith("data/public/"):
                return normalized not in {
                    "data/public/README.md",
                    "data/public/sources.json",
                }
            return "/__pycache__/" in f"/{normalized}"

    for path in ignored_paths:
        assert is_ignored(path), f"{path} must be ignored"

    for path in retained_paths:
        assert not is_ignored(path), f"{path} must remain tracked"


def valid_local_download_source() -> dict[str, object]:
    return {
        "source_id": "local-source",
        "title": "Local Source",
        "source_url": "https://example.invalid/data",
        "source_kind": "dataset",
        "license_name": "CC-BY-4.0",
        "license_status": "verified",
        "source_version": "v1",
        "access_mode": "local_download_only",
        "repository_commit_allowed": False,
        "attribution": "example",
        "license_evidence": "common/license.txt",
        "download_url": "https://example.invalid/archive.zip",
        "archive_md5": "0123456789abcdef0123456789abcdef",
        "permitted_paths": ["WWTP_Langmatt/**"],
        "verified_file_sha256": {
            "WWTP_Langmatt/data/WWTP_Langmatt.csv": EXPECTED_WWTP_CSV_SHA256,
        },
    }


def write_registry(path: Path, source: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "sources": [source]}),
        encoding="utf-8",
    )


def test_loads_verified_co_udlabs_source() -> None:
    sources = load_public_source_registry(
        PROJECT_ROOT / "data" / "public" / "sources.json"
    )

    assert {source.source_id for source in sources} == EXPECTED_SOURCE_IDS

    source = next(
        source for source in sources if source.source_id == "co-udlabs-wwtp-lpicm-2025"
    )

    assert source.source_id == "co-udlabs-wwtp-lpicm-2025"
    assert source.title == "Co-UDlabs LPICM — WWTP Langmatt data within record 15285089"
    assert str(source.source_url) == "https://zenodo.org/records/15285089"
    assert source.source_kind == "dataset"
    assert source.license_name == "CC0-1.0 (WWTP_Langmatt/** only)"
    assert source.license_status == "verified"
    assert source.source_version == "v1.0.0 (2025-04-26)"
    assert source.access_mode == "local_download_only"
    assert source.repository_commit_allowed is False
    assert source.attribution == EXPECTED_ATTRIBUTION
    assert (
        source.license_evidence
        == "common/license.txt; WWTP_Langmatt/documentation/README_WWTP_Langmatt.md"
    )
    assert (
        str(source.download_url) == "https://zenodo.org/api/records/15285089/files/"
        "Co-UDlabs_WP6_T61_EAWAG_001_LPICM.zip/content"
    )
    assert source.archive_md5 == "11c5aea9d1a420752c1d174da635b389"
    assert source.permitted_paths == ["WWTP_Langmatt/**"]
    assert source.verified_file_sha256 == {
        "WWTP_Langmatt/data/WWTP_Langmatt.csv": EXPECTED_WWTP_CSV_SHA256,
    }

    reference_only_sources = [
        source for source in sources if source.source_id != "co-udlabs-wwtp-lpicm-2025"
    ]
    assert len(reference_only_sources) == 4
    assert all(
        source.access_mode == "reference_only" for source in reference_only_sources
    )
    assert all(
        source.license_status == "review_required" for source in reference_only_sources
    )
    assert all(
        source.repository_commit_allowed is False for source in reference_only_sources
    )


@pytest.mark.parametrize(
    (
        "source_id",
        "title",
        "source_url",
        "source_kind",
        "license_name",
        "source_version",
        "attribution",
    ),
    [
        (
            "iwa-benchmark-simulation-models",
            "IWA Benchmark Simulation Models (BSM1 and BSM2)",
            "https://github.com/wwtmodels/Benchmark-Simulation-Models",
            "simulation_model",
            "not_verified",
            "repository main, accessed 2026-07-19",
            "IWA Task Group on Benchmarking of Control Strategies for WWTPs via "
            "wwtmodels/Benchmark-Simulation-Models.",
        ),
        (
            "dhi-waterbench-inflow",
            "WaterBench — Inflow to a Wastewater Treatment Plant",
            "https://github.com/DHI/WaterBench-TimeSeries-WWTPINflow",
            "dataset",
            "not_verified",
            "v1.0, accessed 2026-07-19",
            "DHI WaterBench-TimeSeries-WWTPINflow; preserve the repository licence "
            "and source attribution before any use.",
        ),
        (
            "mee-gb-18918-2025-amendment",
            "GB 18918—2002 (including 2006 and 2025 amendments)",
            "https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/shjbh/swrwpfbz/200307/"
            "W020260206765128897192.pdf",
            "knowledge_document",
            "not_verified",
            "official consolidated electronic edition, accessed 2026-07-19",
            "Ministry of Ecology and Environment of the People's Republic of China, "
            "GB 18918—2002 including amendments.",
        ),
        (
            "epa-nutrient-control-design-manual",
            "EPA Nutrient Control Design Manual",
            "https://www.epa.gov/sites/default/files/2019-02/documents/"
            "nutrient-control-design-manual.pdf",
            "knowledge_document",
            "not_verified",
            "EPA/600/R-10/100, August 2010",
            "United States Environmental Protection Agency, Nutrient Control Design "
            "Manual, EPA/600/R-10/100.",
        ),
    ],
)
def test_loads_reference_only_sources_with_review_boundaries(
    source_id: str,
    title: str,
    source_url: str,
    source_kind: str,
    license_name: str,
    source_version: str,
    attribution: str,
) -> None:
    sources = load_public_source_registry(
        PROJECT_ROOT / "data" / "public" / "sources.json"
    )
    source = next(source for source in sources if source.source_id == source_id)

    assert source.title == title
    assert str(source.source_url) == source_url
    assert source.source_kind == source_kind
    assert source.license_name == license_name
    assert source.license_status == "review_required"
    assert source.source_version == source_version
    assert source.access_mode == "reference_only"
    assert source.repository_commit_allowed is False
    assert source.attribution == attribution
    assert source.permitted_paths == []
    assert source.license_evidence is None
    assert source.download_url is None
    assert source.archive_md5 is None


def test_waterbench_license_remains_unverified_pending_direct_review() -> None:
    sources = load_public_source_registry(
        PROJECT_ROOT / "data" / "public" / "sources.json"
    )
    source = next(
        source for source in sources if source.source_id == "dhi-waterbench-inflow"
    )

    assert source.license_name == "not_verified"
    assert source.license_status == "review_required"
    assert source.license_review_note is not None
    assert "CC-BY-NC-4.0" in source.license_review_note


def test_rejects_local_download_source_without_verified_license(tmp_path: Path) -> None:
    registry_path = tmp_path / "sources.json"
    registry_path.write_text(
        """
        {
          "schema_version": 1,
          "sources": [
            {
              "source_id": "bad-source",
              "title": "Bad Source",
              "source_url": "https://example.invalid/data",
              "source_kind": "dataset",
              "license_name": "not_verified",
              "license_status": "review_required",
              "source_version": "v1",
              "access_mode": "local_download_only",
              "repository_commit_allowed": false,
              "attribution": "example"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="verified license"):
        load_public_source_registry(registry_path)


def test_rejects_local_download_source_with_not_verified_license_name(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["license_name"] = "not_verified"
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match="verified license"):
        load_public_source_registry(registry_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("license_name", "", "verified license"),
        ("license_name", " \t ", "verified license"),
        ("license_evidence", "", "license_evidence"),
        ("license_evidence", " \t ", "license_evidence"),
    ],
)
def test_rejects_local_download_source_with_blank_license_metadata(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source[field] = value
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match=message):
        load_public_source_registry(registry_path)


def test_rejects_local_download_source_without_provenance_fields(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["license_evidence"] = None
    source["download_url"] = None
    source["archive_md5"] = None
    write_registry(registry_path, source)

    with pytest.raises(
        ValueError,
        match="download_url, archive_md5, and license_evidence",
    ):
        load_public_source_registry(registry_path)


def test_rejects_local_download_source_without_verified_file_hashes(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["verified_file_sha256"] = {}
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match="non-empty verified_file_sha256"):
        load_public_source_registry(registry_path)


def test_rejects_invalid_verified_file_sha256(tmp_path: Path) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["verified_file_sha256"] = {
        "WWTP_Langmatt/data/WWTP_Langmatt.csv": "A" * 64,
    }
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        load_public_source_registry(registry_path)


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("", "non-empty relative POSIX"),
        ("/WWTP_Langmatt/data/WWTP_Langmatt.csv", "relative POSIX"),
        ("../WWTP_Langmatt/data/WWTP_Langmatt.csv", "parent traversal"),
        ("WWTP_Langmatt\\data\\WWTP_Langmatt.csv", "relative POSIX"),
        ("C:/WWTP_Langmatt/data/WWTP_Langmatt.csv", "relative POSIX"),
        ("WWTP_Langmatt/*/WWTP_Langmatt.csv", "concrete file paths"),
        ("WWTP_Langmatt/**", "concrete file paths"),
    ],
)
def test_rejects_unsafe_verified_file_sha256_paths(
    tmp_path: Path, relative_path: str, message: str
) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["verified_file_sha256"] = {
        relative_path: EXPECTED_WWTP_CSV_SHA256,
    }
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match=message):
        load_public_source_registry(registry_path)


@pytest.mark.parametrize("sha256", ["a" * 63, "a" * 65])
def test_rejects_verified_file_sha256_with_wrong_length(
    tmp_path: Path, sha256: str
) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["verified_file_sha256"] = {
        "WWTP_Langmatt/data/WWTP_Langmatt.csv": sha256,
    }
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        load_public_source_registry(registry_path)


def test_rejects_verified_file_hash_outside_permitted_paths(tmp_path: Path) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["verified_file_sha256"] = {
        "Rainfall_Lenzburg/data/rainfall.csv": EXPECTED_WWTP_CSV_SHA256,
    }
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match="covered by permitted_paths"):
        load_public_source_registry(registry_path)


def test_rejects_local_download_source_with_invalid_archive_md5(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["archive_md5"] = "not-an-md5"
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        load_public_source_registry(registry_path)


def test_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    registry_path = tmp_path / "sources.json"
    registry_path.write_text(
        """
        {
          "schema_version": 1,
          "sources": [
            {
              "source_id": "duplicate-source",
              "title": "First Source",
              "source_url": "https://example.invalid/first",
              "source_kind": "knowledge_document",
              "license_name": "CC-BY-4.0",
              "license_status": "verified",
              "source_version": "v1",
              "access_mode": "reference_only",
              "repository_commit_allowed": false,
              "attribution": "example"
            },
            {
              "source_id": "duplicate-source",
              "title": "Second Source",
              "source_url": "https://example.invalid/second",
              "source_kind": "simulation_model",
              "license_name": "CC-BY-4.0",
              "license_status": "verified",
              "source_version": "v2",
              "access_mode": "reference_only",
              "repository_commit_allowed": false,
              "attribution": "example"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_id values must be unique"):
        load_public_source_registry(registry_path)


@pytest.mark.parametrize(
    ("permitted_paths", "message"),
    [
        ([], "permitted_paths"),
        (["../Rainfall_Lenzburg/**"], "parent traversal"),
    ],
)
def test_rejects_unsafe_local_download_paths(
    tmp_path: Path, permitted_paths: list[str], message: str
) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["permitted_paths"] = permitted_paths
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match=message):
        load_public_source_registry(registry_path)


@pytest.mark.parametrize(
    ("permitted_paths", "message"),
    [
        (["/WWTP_Langmatt/**"], "relative POSIX"),
        (["WWTP_Langmatt/./data.csv"], "parent traversal"),
        (["C:/WWTP_Langmatt/data.csv"], "relative POSIX"),
        (["*"], "full-path wildcards"),
        (["**"], "full-path wildcards"),
    ],
)
def test_rejects_non_archive_permitted_paths(
    tmp_path: Path, permitted_paths: list[str], message: str
) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["permitted_paths"] = permitted_paths
    write_registry(registry_path, source)

    with pytest.raises(ValueError, match=message):
        load_public_source_registry(registry_path)


def test_rejects_repository_commits_for_raw_data(tmp_path: Path) -> None:
    registry_path = tmp_path / "sources.json"
    source = valid_local_download_source()
    source["repository_commit_allowed"] = True
    write_registry(registry_path, source)

    with pytest.raises(ValidationError):
        load_public_source_registry(registry_path)
