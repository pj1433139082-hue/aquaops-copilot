"""Prepare the fixed public-only AquaOps RAG runtime."""

import sys

from aquaops.public_demo_cli import main


if __name__ == "__main__":
    raise SystemExit(main(("prepare", *sys.argv[1:])))
