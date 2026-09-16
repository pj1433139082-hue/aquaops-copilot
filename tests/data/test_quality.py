import importlib.util
import json
import sys
from pathlib import Path

import pytest

import aquaops.data.quality as quality_module
from aquaops.data.quality import (
    PublicDataQualityError,
    profile_wwtp_langmatt_csv,
    write_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_SCRIPT = PROJECT_ROOT / "scripts" / "profile_public_source.py"
SOURCE_ID = "co-udlabs-wwtp-lpicm-2025"


@pytest.fixture
def approved_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source_root = tmp_path / "data" / "public" / SOURCE_ID
    monkeypatch.setattr(quality_module, "PUBLIC_SOURCE_ROOT", source_root)
    monkeypatch.setattr(
        quality_module,
        "DERIVED_SOURCE_ROOT",
        tmp_path / "data" / "public" / "derived" / SOURCE_ID,
    )
    monkeypatch.setattr(
        quality_module, "PRIVATE_DATA_ROOT", tmp_path / "data" / "private"
    )
    return source_root


def write_csv(path: Path, body: str, *, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding=encoding, newline="")
    return path


def valid_csv_body() -> str:
    return (
        "date_vec,q,cond,nh4\n"
        "2025-01-01T00:00:00Z,10,100,0.2\n"
        "2025-01-01T00:05:00Z,,110,0.3\n"
        "2025-01-01T00:10:00Z,12,120,\n"
        "2025-01-01T00:15:00Z,14,130,0.5\n"
    )


def generated_profile(source_root: Path) -> dict[str, object]:
    return profile_wwtp_langmatt_csv(
        write_csv(source_root / "WWTP_Langmatt.csv", valid_csv_body())
    )


def load_cli_module() -> object:
    module_name = "aquaops_profile_public_source_cli_test"
    spec = importlib.util.spec_from_file_location(module_name, CLI_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_profiles_expected_schema_and_aggregate_statistics(
    approved_source_root: Path,
) -> None:
    profile = profile_wwtp_langmatt_csv(
        write_csv(approved_source_root / "WWTP_Langmatt.csv", valid_csv_body())
    )

    assert profile["schema_fields"] == ["date_vec", "q", "cond", "nh4"]
    assert profile["row_count"] == 4
    assert profile["timestamp_column"] == "date_vec"
    assert profile["start"] == "2025-01-01T00:00:00Z"
    assert profile["end"] == "2025-01-01T00:15:00Z"
    assert profile["timestamp_order_violations"] == 0
    assert profile["duplicate_timestamp_count"] == 0
    assert profile["median_interval_seconds"] == 300.0
    assert profile["numeric_columns"]["q"] == {
        "non_null_count": 3,
        "null_count": 1,
        "invalid_numeric_count": 0,
        "min": 10.0,
        "max": 14.0,
        "mean": 12.0,
    }
    assert profile["numeric_columns"]["cond"]["mean"] == 115.0
    assert profile["numeric_columns"]["nh4"] == {
        "non_null_count": 3,
        "null_count": 1,
        "invalid_numeric_count": 0,
        "min": 0.2,
        "max": 0.5,
        "mean": pytest.approx(1 / 3),
    }


def test_accepts_utf8_bom_headers(approved_source_root: Path) -> None:
    profile = profile_wwtp_langmatt_csv(
        write_csv(
            approved_source_root / "WWTP_Langmatt.csv",
            valid_csv_body(),
            encoding="utf-8-sig",
        )
    )

    assert profile["schema_fields"][0] == "date_vec"
    assert profile["row_count"] == 4


def test_interprets_naive_timestamps_as_utc(approved_source_root: Path) -> None:
    profile = profile_wwtp_langmatt_csv(
        write_csv(
            approved_source_root / "WWTP_Langmatt.csv",
            "date_vec,q,cond,nh4\n"
            "2025-01-01T00:00:00,10,100,0.2\n"
            "2025-01-01T00:05:00,11,101,0.3\n",
        )
    )

    assert profile["start"] == "2025-01-01T00:00:00Z"
    assert profile["end"] == "2025-01-01T00:05:00Z"
    assert profile["median_interval_seconds"] == 300.0


def test_normalizes_offset_timestamps_to_utc(approved_source_root: Path) -> None:
    profile = profile_wwtp_langmatt_csv(
        write_csv(
            approved_source_root / "WWTP_Langmatt.csv",
            "date_vec,q,cond,nh4\n"
            "2025-01-01T08:00:00+08:00,10,100,0.2\n"
            "2025-01-01T09:00:00+08:00,11,101,0.3\n",
        )
    )

    assert profile["start"] == "2025-01-01T00:00:00Z"
    assert profile["end"] == "2025-01-01T01:00:00Z"
    assert profile["median_interval_seconds"] == 3600.0


def test_normalizes_mixed_z_offset_and_naive_timestamps_to_utc(
    approved_source_root: Path,
) -> None:
    profile = profile_wwtp_langmatt_csv(
        write_csv(
            approved_source_root / "WWTP_Langmatt.csv",
            "date_vec,q,cond,nh4\n"
            "2025-01-01T00:10:00Z,10,100,0.2\n"
            "2025-01-01T08:00:00+08:00,11,101,0.3\n"
            "2025-01-01T00:05:00,12,102,0.4\n",
        )
    )

    assert profile["start"] == "2025-01-01T00:00:00Z"
    assert profile["end"] == "2025-01-01T00:10:00Z"
    assert profile["timestamp_order_violations"] == 1
    assert profile["median_interval_seconds"] == 300.0


def test_rejects_missing_required_columns(approved_source_root: Path) -> None:
    path = write_csv(
        approved_source_root / "WWTP_Langmatt.csv",
        "date_vec,q,cond\n2025-01-01T00:00:00Z,10,100\n",
    )

    with pytest.raises(PublicDataQualityError, match="missing required columns: nh4"):
        profile_wwtp_langmatt_csv(path)


def test_rejects_invalid_numeric_values_after_counting_them(
    approved_source_root: Path,
) -> None:
    path = write_csv(
        approved_source_root / "WWTP_Langmatt.csv",
        "date_vec,q,cond,nh4\n2025-01-01T00:00:00Z,invalid,100,0.2\n",
    )

    with pytest.raises(PublicDataQualityError, match=r"invalid numeric values: q\(1\)"):
        profile_wwtp_langmatt_csv(path)


def test_rejects_invalid_timestamps_without_echoing_source_values(
    approved_source_root: Path,
) -> None:
    path = write_csv(
        approved_source_root / "WWTP_Langmatt.csv",
        "date_vec,q,cond,nh4\nnot-a-time,10,100,0.2\n",
    )

    with pytest.raises(PublicDataQualityError, match="invalid timestamp at row 2"):
        profile_wwtp_langmatt_csv(path)


def test_reports_duplicate_and_out_of_order_timestamps_without_reordering(
    approved_source_root: Path,
) -> None:
    path = write_csv(
        approved_source_root / "WWTP_Langmatt.csv",
        "date_vec,q,cond,nh4\n"
        "2025-01-01T00:05:00Z,10,100,0.2\n"
        "2025-01-01T00:00:00Z,11,101,0.3\n"
        "2025-01-01T00:00:00Z,12,102,0.4\n"
        "2025-01-01T00:10:00Z,13,103,0.5\n",
    )

    profile = profile_wwtp_langmatt_csv(path)

    assert profile["timestamp_order_violations"] == 2
    assert profile["duplicate_timestamp_count"] == 1
    assert profile["median_interval_seconds"] == 600.0
    assert profile["start"] == "2025-01-01T00:00:00Z"
    assert profile["end"] == "2025-01-01T00:10:00Z"


def test_rejects_private_input_before_reading(
    tmp_path: Path, approved_source_root: Path
) -> None:
    private_path = tmp_path / "data" / "private" / "water-plant.csv"

    with pytest.raises(PublicDataQualityError, match="private data"):
        profile_wwtp_langmatt_csv(private_path)


def test_rejects_empty_public_csv(approved_source_root: Path) -> None:
    path = write_csv(
        approved_source_root / "WWTP_Langmatt.csv", "date_vec,q,cond,nh4\n"
    )

    with pytest.raises(PublicDataQualityError, match="no data rows"):
        profile_wwtp_langmatt_csv(path)


def test_rejects_other_public_source_input(
    tmp_path: Path, approved_source_root: Path
) -> None:
    other_source_csv = write_csv(
        tmp_path / "data" / "public" / "other-public-source" / "source.csv",
        valid_csv_body(),
    )

    with pytest.raises(
        PublicDataQualityError, match="approved public source directory"
    ):
        profile_wwtp_langmatt_csv(other_source_csv)


def test_write_profile_rejects_private_outside_and_csv_targets(
    tmp_path: Path, approved_source_root: Path
) -> None:
    profile = {"row_count": 4}

    with pytest.raises(PublicDataQualityError, match="private data"):
        write_profile(profile, tmp_path / "data" / "private" / "profile.json")
    with pytest.raises(PublicDataQualityError, match="derived public source directory"):
        write_profile(profile, tmp_path / "outside.json")
    with pytest.raises(PublicDataQualityError, match="derived public source directory"):
        write_profile(
            profile,
            approved_source_root / "WWTP_Langmatt.csv",
        )


@pytest.mark.parametrize(
    "output_path",
    [
        Path("data/public/other-public-source/profile.json"),
        Path("data/public/derived/other-public-source/profile.json"),
    ],
)
def test_write_profile_rejects_cross_source_targets(
    tmp_path: Path, approved_source_root: Path, output_path: Path
) -> None:
    with pytest.raises(PublicDataQualityError, match="derived public source directory"):
        write_profile({"row_count": 4}, tmp_path / output_path)


def test_write_profile_explicitly_rejects_overwriting_input(
    approved_source_root: Path,
) -> None:
    source_csv = approved_source_root / "WWTP_Langmatt.csv"

    with pytest.raises(PublicDataQualityError, match="must not overwrite the input"):
        write_profile({"row_count": 4}, source_csv, input_path=source_csv)


def test_write_profile_creates_stable_public_json(
    tmp_path: Path, approved_source_root: Path
) -> None:
    output = tmp_path / "data" / "public" / "derived" / SOURCE_ID / "profile.json"
    profile = generated_profile(approved_source_root)

    write_profile(profile, output)

    assert output.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(output.read_text(encoding="utf-8")) == profile
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize(
    "timestamp_field,timestamp_value",
    [
        ("start", "2025-01-01T00:00:00"),
        ("end", "2025-01-01T08:05:00+08:00"),
    ],
)
def test_write_profile_rejects_noncanonical_utc_timestamps(
    tmp_path: Path,
    approved_source_root: Path,
    timestamp_field: str,
    timestamp_value: str,
) -> None:
    profile = generated_profile(approved_source_root)
    profile[timestamp_field] = timestamp_value
    output = tmp_path / "data" / "public" / "derived" / SOURCE_ID / "profile.json"

    with pytest.raises(PublicDataQualityError, match="canonical UTC timestamps"):
        write_profile(profile, output)

    assert not output.exists()


def test_write_profile_rejects_unaggregated_raw_rows(
    tmp_path: Path, approved_source_root: Path
) -> None:
    profile = generated_profile(approved_source_root)
    profile["raw_rows"] = []
    output = tmp_path / "data" / "public" / "derived" / SOURCE_ID / "profile.json"

    with pytest.raises(PublicDataQualityError, match="profile schema.*raw_rows"):
        write_profile(profile, output)

    assert not output.exists()


@pytest.mark.parametrize("source_filename", ["README.md", ".acquisition.json"])
def test_write_profile_rejects_source_tree_files_even_if_they_exist(
    approved_source_root: Path, source_filename: str
) -> None:
    source_file = approved_source_root / source_filename
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("source-owned", encoding="utf-8")

    with pytest.raises(PublicDataQualityError, match="derived public source directory"):
        write_profile(generated_profile(approved_source_root), source_file)

    assert source_file.read_text(encoding="utf-8") == "source-owned"


def test_write_profile_cleans_temporary_file_when_atomic_replace_fails(
    tmp_path: Path, approved_source_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "data" / "public" / "derived" / SOURCE_ID / "profile.json"

    def fail_replace(_: Path, __: Path) -> None:
        raise OSError("replace unavailable")

    monkeypatch.setattr(
        quality_module, "_replace_profile_file", fail_replace, raising=False
    )

    with pytest.raises(PublicDataQualityError, match="could not write"):
        write_profile(generated_profile(approved_source_root), output)

    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_write_profile_rejects_nested_sample_values(
    tmp_path: Path, approved_source_root: Path
) -> None:
    profile = generated_profile(approved_source_root)
    numeric_columns = profile["numeric_columns"]
    assert isinstance(numeric_columns, dict)
    q_summary = numeric_columns["q"]
    assert isinstance(q_summary, dict)
    q_summary["sample_values"] = [10.0]
    output = tmp_path / "data" / "public" / "derived" / SOURCE_ID / "profile.json"

    with pytest.raises(PublicDataQualityError, match="profile schema.*sample_values"):
        write_profile(profile, output)

    assert not output.exists()


def test_cli_profiles_valid_relative_public_source_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = load_cli_module()
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    source_directory = tmp_path / "data" / "public" / SOURCE_ID
    monkeypatch.setattr(quality_module, "PUBLIC_SOURCE_ROOT", source_directory)
    monkeypatch.setattr(
        quality_module,
        "DERIVED_SOURCE_ROOT",
        tmp_path / "data" / "public" / "derived" / SOURCE_ID,
    )
    monkeypatch.setattr(
        quality_module, "PRIVATE_DATA_ROOT", tmp_path / "data" / "private"
    )
    write_csv(
        source_directory / "WWTP_Langmatt" / "data" / "WWTP_Langmatt.csv",
        valid_csv_body(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_public_source.py",
            SOURCE_ID,
            "--source-dir",
            f"data/public/{SOURCE_ID}",
        ],
    )

    assert cli.main() == 0
    assert (
        json.loads(
            (
                tmp_path
                / "data"
                / "public"
                / "derived"
                / SOURCE_ID
                / ".quality-profile.json"
            ).read_text(encoding="utf-8")
        )["row_count"]
        == 4
    )


def test_cli_rejects_unknown_source_id_and_private_source_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = load_cli_module()
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        quality_module, "PUBLIC_SOURCE_ROOT", tmp_path / "data" / "public" / SOURCE_ID
    )
    monkeypatch.setattr(
        quality_module,
        "DERIVED_SOURCE_ROOT",
        tmp_path / "data" / "public" / "derived" / SOURCE_ID,
    )
    monkeypatch.setattr(
        quality_module, "PRIVATE_DATA_ROOT", tmp_path / "data" / "private"
    )

    monkeypatch.setattr(sys, "argv", ["profile_public_source.py", "not-registered"])
    with pytest.raises(SystemExit) as unknown_source:
        cli.main()
    assert unknown_source.value.code == 2

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_public_source.py",
            SOURCE_ID,
            "--source-dir",
            "data/private",
        ],
    )
    assert cli.main() == 2


def test_cli_rejects_csv_that_resolves_outside_registered_source_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = load_cli_module()
    source_directory = tmp_path / "data" / "public" / SOURCE_ID
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(quality_module, "PUBLIC_SOURCE_ROOT", source_directory)
    monkeypatch.setattr(
        quality_module,
        "DERIVED_SOURCE_ROOT",
        tmp_path / "data" / "public" / "derived" / SOURCE_ID,
    )
    monkeypatch.setattr(
        quality_module, "PRIVATE_DATA_ROOT", tmp_path / "data" / "private"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_public_source.py",
            SOURCE_ID,
            "--csv",
            f"data/public/{SOURCE_ID}/../other-public-source/source.csv",
        ],
    )

    assert cli.main() == 2
