#!/usr/bin/env python3
"""Optionally fetch public quote inputs through AKShare into SQLite."""

from __future__ import annotations

import argparse
import sqlite3
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from industry_rotation.data import normalise_industry_quotes, normalise_market_quotes
from industry_rotation.samples import continuity_codes


def _retry(call: Callable[[], pd.DataFrame], retries: int, delay_seconds: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return call()
        except Exception as error:  # network and supplier errors vary by AKShare version
            last_error = error
            if attempt + 1 < retries:
                time.sleep(delay_seconds)
    raise RuntimeError(f"AKShare request failed after {retries} attempts") from last_error


def fetch_sw_quotes(
    start_date: str,
    end_date: str,
    retries: int = 3,
    delay_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the continuity sample and HS300; importing this module never calls the network."""

    import akshare as ak

    industry_frames: list[pd.DataFrame] = []
    for code in continuity_codes():
        raw = _retry(
            lambda code=code: ak.index_hist_sw(symbol=code, period="day"),
            retries,
            delay_seconds,
        )
        raw = raw.copy()
        raw["index_code"] = code
        normalised = normalise_industry_quotes(raw)
        dates = (normalised["date"] >= pd.Timestamp(start_date)) & (
            normalised["date"] <= pd.Timestamp(end_date)
        )
        industry_frames.append(normalised.loc[dates])
    industry = pd.concat(industry_frames, ignore_index=True)
    hs300_raw = _retry(
        lambda: ak.stock_zh_index_daily_em(
            symbol="sh000300", start_date=start_date, end_date=end_date
        ),
        retries,
        delay_seconds,
    )
    hs300 = normalise_market_quotes(hs300_raw)
    dates = (hs300["date"] >= pd.Timestamp(start_date)) & (hs300["date"] <= pd.Timestamp(end_date))
    return industry, hs300.loc[dates].reset_index(drop=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--start-date", default="20150101")
    parser.add_argument("--end-date", default="20260630")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.database.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.database}; pass --overwrite")
    args.database.parent.mkdir(parents=True, exist_ok=True)
    industry, hs300 = fetch_sw_quotes(
        args.start_date, args.end_date, args.retries, args.delay_seconds
    )
    with sqlite3.connect(args.database) as connection:
        industry.to_sql("sw_index_quotes", connection, if_exists="replace", index=False)
        hs300.to_sql("hs300_quotes", connection, if_exists="replace", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
