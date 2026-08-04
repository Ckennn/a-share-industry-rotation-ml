#!/usr/bin/env python3
"""Build compact public figures and a Markdown summary from saved outputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from industry_rotation.reporting import build_report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    build_report(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
