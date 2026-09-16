"""Create an aggregate-only local quality profile for the approved public source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "co-udlabs-wwtp-lpicm-2025"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an aggregate quality profile for the approved public source."
    )
    parser.add_argument("source_id", choices=[SOURCE_ID])
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Public source directory, relative to the project root by default",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV path inside --source-dir, relative to the project root by default",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output under data/public, relative to the project root by default",
    )
    return parser.parse_args()


def _resolve_from_project_root(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main() -> int:
    from aquaops.data.quality import (
        PublicDataQualityError,
        profile_wwtp_langmatt_csv,
        write_profile,
    )

    args = parse_args()
    expected_source_directory = (
        PROJECT_ROOT / "data" / "public" / args.source_id
    ).resolve()
    source_directory = (
        expected_source_directory
        if args.source_dir is None
        else _resolve_from_project_root(args.source_dir)
    )
    if source_directory != expected_source_directory:
        print(
            "Profiling refused: source directory must be the registered public source directory",
            file=sys.stderr,
        )
        return 2

    csv_path = (
        source_directory / "WWTP_Langmatt" / "data" / "WWTP_Langmatt.csv"
        if args.csv is None
        else _resolve_from_project_root(args.csv)
    )
    if not _is_within(csv_path, source_directory):
        print("Profiling refused: CSV must be inside the public source directory", file=sys.stderr)
        return 2

    output_path = (
        PROJECT_ROOT
        / "data"
        / "public"
        / "derived"
        / args.source_id
        / ".quality-profile.json"
        if args.output is None
        else _resolve_from_project_root(args.output)
    )
    try:
        profile = profile_wwtp_langmatt_csv(csv_path)
        write_profile(profile, output_path, input_path=csv_path)
    except PublicDataQualityError as error:
        print(f"Profiling refused: {error}", file=sys.stderr)
        return 2

    print(f"Wrote aggregate public quality profile to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
