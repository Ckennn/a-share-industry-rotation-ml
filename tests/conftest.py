from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from industry_rotation.config import ExperimentConfig, load_config
from industry_rotation.features import BASELINE_FEATURES
from industry_rotation.samples import continuity_codes


@pytest.fixture
def smoke_config() -> ExperimentConfig:
    return load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")


@pytest.fixture
def monthly_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    months = pd.date_range("2015-01-30", "2026-06-30", freq="ME")
    rows: list[dict[str, object]] = []
    for month_number, decision_date in enumerate(months[:-1]):
        market = 0.01 * np.sin(month_number / 8.0)
        for code_number, code in enumerate(continuity_codes()):
            features = {feature: float(rng.normal(0.0, 0.05)) for feature in BASELINE_FEATURES}
            for constant in (
                "avg_turnover_rate_1m",
                "positive_stock_ratio_1m",
                "coverage_ratio",
                "coverage_ratio_1m",
            ):
                features[constant] = 0.0
            next_return = market + 0.002 * (code_number % 5) + float(rng.normal(0.0, 0.03))
            hs300_return = market + float(rng.normal(0.0, 0.01))
            rows.append(
                {
                    "decision_date": decision_date.strftime("%Y%m%d"),
                    "decision_month": decision_date.strftime("%Y%m"),
                    "next_month_start": (decision_date + pd.offsets.MonthBegin()).strftime("%Y%m%d"),
                    "next_month_end": (decision_date + pd.offsets.MonthEnd()).strftime("%Y%m%d"),
                    "index_code": code,
                    "index_name": code,
                    "next_month_return": next_return,
                    "next_month_hs300_return": hs300_return,
                    "next_month_excess_return": next_return - hs300_return,
                    **features,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def prediction_month(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    result = monthly_panel[monthly_panel.decision_date == monthly_panel.decision_date.min()].copy()
    result["predicted_score"] = np.arange(len(result), dtype=float)
    return result


@pytest.fixture
def training_frame(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    return monthly_panel[monthly_panel.decision_date <= "20221231"].copy()


@pytest.fixture
def daily_bundle() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2014-01-02", periods=560)
    industry_rows: list[dict[str, object]] = []
    for code_number, code in enumerate(continuity_codes()):
        daily_return = 0.0002 + 0.00001 * code_number + 0.001 * np.sin(np.arange(len(dates)) / 17.0)
        close = 100.0 * np.cumprod(1.0 + daily_return)
        for date, value in zip(dates, close, strict=True):
            industry_rows.append(
                {
                    "index_code": code,
                    "index_name": code,
                    "date": date,
                    "open": value,
                    "high": value * 1.01,
                    "low": value * 0.99,
                    "close": value,
                    "volume": 1_000_000.0 + code_number,
                    "amount": 100_000_000.0 + code_number,
                }
            )
    hs_return = 0.0002 + 0.0008 * np.sin(np.arange(len(dates)) / 19.0)
    hs_close = 3_000.0 * np.cumprod(1.0 + hs_return)
    hs300 = pd.DataFrame(
        {
            "date": dates,
            "open": hs_close,
            "high": hs_close * 1.01,
            "low": hs_close * 0.99,
            "close": hs_close,
            "volume": 1_000_000_000.0,
            "amount": 100_000_000_000.0,
        }
    )
    return pd.DataFrame(industry_rows), hs300
