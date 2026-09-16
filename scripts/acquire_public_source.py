"""Acquire one registry-approved public dataset without accepting arbitrary URLs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one checksum-verified source from data/public/sources.json."
    )
    parser.add_argument("source_id", help="A source_id registered in sources.json")
    parser.add_argument(
        "--destination",
        type=Path,
        help="New, non-existent directory for registry-permitted archive members",
    )
    return parser.parse_args()


def main() -> int:
    from aquaops.rag.acquire import PublicDataAcquisitionError, acquire_public_source

    args = parse_args()
    if args.destination is None:
        destination = PROJECT_ROOT / "data" / "public" / args.source_id
    elif args.destination.is_absolute():
        destination = args.destination
    else:
        destination = PROJECT_ROOT / args.destination
    try:
        receipt = acquire_public_source(
            PROJECT_ROOT / "data" / "public" / "sources.json",
            args.source_id,
            destination,
        )
    except PublicDataAcquisitionError as error:
        print(f"Acquisition refused: {error}", file=sys.stderr)
        return 2

    print(f"Acquired {receipt['source_id']} into {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
