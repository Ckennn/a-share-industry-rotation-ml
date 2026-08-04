#!/usr/bin/env python3
"""Run prior-only monthly selection, refitting, prediction, and audit."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from industry_rotation.config import load_config
from industry_rotation.validation import run_walk_forward


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    panel = pd.read_csv(
        args.panel,
        dtype={
            "index_code": str,
            "decision_date": str,
            "decision_month": str,
            "next_month_start": str,
            "next_month_end": str,
        },
    )
    result = run_walk_forward(panel, config)
    from industry_rotation.reporting import write_experiment_result

    write_experiment_result(result, args.output, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
