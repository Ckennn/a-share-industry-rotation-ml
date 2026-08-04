import pandas as pd
import pytest

from industry_rotation.config import SplitSpec
from industry_rotation.features import (
    BASELINE_FEATURES,
    CONSTANT_COMPATIBILITY_FEATURES,
    WINDOW_1M,
    WINDOW_3M,
    WINDOW_6M,
    WINDOW_12M,
    build_monthly_panel,
)


def test_feature_registry_matches_audited_contract() -> None:
    assert len(BASELINE_FEATURES) == 18
    assert set(CONSTANT_COMPATIBILITY_FEATURES) == {
        "avg_turnover_rate_1m",
        "positive_stock_ratio_1m",
        "coverage_ratio",
        "coverage_ratio_1m",
    }
    assert (WINDOW_1M, WINDOW_3M, WINDOW_6M, WINDOW_12M) == (21, 63, 126, 252)


def test_next_month_label_is_shifted_forward(
    daily_bundle: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    panel = build_monthly_panel(*daily_bundle, split=SplitSpec("20221231", "20231231", "20260531"))
    rows = panel.query("index_code == '801010'").sort_values("decision_date").iloc[:2]
    expected = rows.iloc[1]["decision_close"] / rows.iloc[0]["decision_close"] - 1.0
    assert rows.iloc[0]["next_month_return"] == pytest.approx(expected)


def test_future_quote_does_not_change_prior_features(
    daily_bundle: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    split = SplitSpec("20221231", "20231231", "20260531")
    industry, hs300 = daily_bundle
    before = build_monthly_panel(industry, hs300, split=split)
    next_date = (
        pd.Timestamp(industry["date"].max())
        + pd.offsets.MonthEnd(0)
        + pd.offsets.BDay(1)
    )
    industry_future = industry.groupby("index_code", as_index=False).tail(1).copy()
    industry_future["date"] = next_date
    industry_future["close"] *= 50.0
    hs300_future = hs300.tail(1).copy()
    hs300_future["date"] = next_date
    hs300_future["close"] *= 50.0
    after = build_monthly_panel(
        pd.concat([industry, industry_future], ignore_index=True),
        pd.concat([hs300, hs300_future], ignore_index=True),
        split=split,
    )
    cutoff = before["decision_date"].max()
    pd.testing.assert_frame_equal(
        before[before.decision_date <= cutoff].reset_index(drop=True),
        after[after.decision_date <= cutoff].reset_index(drop=True),
    )
