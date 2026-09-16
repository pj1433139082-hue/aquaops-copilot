"""Smoke-test the public-only Agent, API, and MCP entry points."""

import sys

from aquaops.public_demo_cli import main


if __name__ == "__main__":
    raise SystemExit(main(("smoke", *sys.argv[1:])))
