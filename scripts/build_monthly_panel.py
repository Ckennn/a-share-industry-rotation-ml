#!/usr/bin/env python3
"""Build the monthly research panel from user-supplied quote files or SQLite."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from industry_rotation.config import load_config
from industry_rotation.data import load_sqlite_table
from industry_rotation.features import build_monthly_panel


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--database", type=Path)
    source.add_argument("--industry-csv", type=Path)
    parser.add_argument("--hs300-csv", type=Path)
    parser.add_argument("--industry-table", default="sw_index_quotes")
    parser.add_argument("--hs300-table", default="hs300_quotes")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    if args.database:
        industry = load_sqlite_table(args.database, args.industry_table)
        hs300 = load_sqlite_table(args.database, args.hs300_table)
    else:
        if args.hs300_csv is None:
            raise ValueError("--hs300-csv is required with --industry-csv")
        industry = pd.read_csv(args.industry_csv, dtype={"index_code": str})
        hs300 = pd.read_csv(args.hs300_csv)
    config = load_config(args.config)
    panel = build_monthly_panel(industry, hs300, config.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
