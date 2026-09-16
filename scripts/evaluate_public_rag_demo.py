"""Evaluate all fixed public-only AquaOps RAG variants."""

import sys

from aquaops.public_demo_cli import main


if __name__ == "__main__":
    raise SystemExit(main(("evaluate", *sys.argv[1:])))
