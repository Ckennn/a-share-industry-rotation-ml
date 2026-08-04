#!/usr/bin/env python3
"""Generate a deterministic, non-empirical monthly panel for public tests."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from industry_rotation.config import SplitSpec
from industry_rotation.features import BASELINE_FEATURES, CONSTANT_COMPATIBILITY_FEATURES
from industry_rotation.samples import continuity_codes, continuity_names


def generate_synthetic_panel(seed: int = 42) -> pd.DataFrame:
    """Create 27 industries by 137 decision months from generated latent factors."""

    rng = np.random.default_rng(seed)
    calendar = pd.date_range("2015-01-31", "2026-06-30", freq="ME")
    decisions = calendar[:-1]
    split = SplitSpec("20221231", "20231231", "20260531")
    codes = continuity_codes()
    names = continuity_names()
    industry_effect = rng.normal(0.0, 0.012, size=len(codes))
    rows: list[dict[str, object]] = []

    for month_number, decision_date in enumerate(decisions):
        holding_end = calendar[month_number + 1]
        market_state = 0.012 * np.sin(month_number / 7.0) + float(rng.normal(0.0, 0.008))
        market_volatility = 0.012 + 0.004 * (1.0 + np.cos(month_number / 11.0))
        hs300_next = market_state + float(rng.normal(0.0, 0.018))
        hs_features = {
            "hs300_ret_1m": market_state + float(rng.normal(0.0, 0.01)),
            "hs300_ret_3m": 2.5 * market_state + float(rng.normal(0.0, 0.015)),
            "hs300_vol_1m": market_volatility,
            "hs300_vol_3m": market_volatility * 1.15,
        }
        for code_number, code in enumerate(codes):
            latent = industry_effect[code_number] + 0.015 * np.sin(
                month_number / 5.0 + code_number / 4.0
            )
            ret_1m = market_state + latent + float(rng.normal(0.0, 0.025))
            ret_3m = 2.2 * latent + 1.8 * market_state + float(rng.normal(0.0, 0.035))
            vol_1m = abs(market_volatility + float(rng.normal(0.0, 0.004)))
            vol_3m = abs(1.15 * market_volatility + float(rng.normal(0.0, 0.004)))
            features: dict[str, float] = {
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "ret_6m": 1.6 * ret_3m + float(rng.normal(0.0, 0.05)),
                "ret_12m": 2.4 * ret_3m + float(rng.normal(0.0, 0.07)),
                "vol_1m": vol_1m,
                "vol_3m": vol_3m,
                "drawdown_1m": -abs(float(rng.normal(0.025 + vol_1m, 0.015))),
                "avg_turnover_1m": float(np.exp(18.0 + 0.3 * latent + rng.normal(0.0, 0.2))),
                **hs_features,
                "industry_minus_hs300_1m": ret_1m - hs_features["hs300_ret_1m"],
                "industry_minus_hs300_3m": ret_3m - hs_features["hs300_ret_3m"],
            }
            for constant in CONSTANT_COMPATIBILITY_FEATURES:
                features[constant] = 0.0
            excess_signal = (
                0.16 * features["industry_minus_hs300_1m"]
                + 0.09 * features["industry_minus_hs300_3m"]
                - 0.08 * features["vol_1m"]
                + 0.04 * np.tanh(ret_1m / max(vol_1m, 1e-6))
            )
            next_excess = excess_signal + float(rng.normal(0.0, 0.035))
            next_return = hs300_next + next_excess
            rows.append(
                {
                    "decision_date": decision_date.strftime("%Y%m%d"),
                    "decision_month": decision_date.strftime("%Y%m"),
                    "next_month_start": holding_end.replace(day=1).strftime("%Y%m%d"),
                    "next_month_end": holding_end.strftime("%Y%m%d"),
                    "split": split.label(decision_date.strftime("%Y%m%d")),
                    "index_code": code,
                    "index_name": names[code],
                    "decision_close": 100.0,
                    **{feature: features[feature] for feature in BASELINE_FEATURES},
                    "next_month_return": next_return,
                    "next_month_hs300_return": hs300_next,
                    "next_month_excess_return": next_excess,
                    "source": "synthetic_test_fixture",
                }
            )
    return pd.DataFrame(rows).sort_values(["decision_date", "index_code"]).reset_index(drop=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/synthetic_panel.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_panel(args.seed).to_csv(args.output, index=False, float_format="%.12g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
