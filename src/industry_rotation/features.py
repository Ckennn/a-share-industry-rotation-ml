"""Daily-to-monthly baseline features and forward labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SplitSpec
from .data import normalise_industry_quotes, normalise_market_quotes, validate_monthly_universe
from .samples import continuity_codes

WINDOW_1M = 21
WINDOW_3M = 63
WINDOW_6M = 126
WINDOW_12M = 252

BASELINE_FEATURES = (
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "ret_12m",
    "vol_1m",
    "vol_3m",
    "drawdown_1m",
    "avg_turnover_1m",
    "avg_turnover_rate_1m",
    "positive_stock_ratio_1m",
    "coverage_ratio",
    "coverage_ratio_1m",
    "hs300_ret_1m",
    "hs300_ret_3m",
    "hs300_vol_1m",
    "hs300_vol_3m",
    "industry_minus_hs300_1m",
    "industry_minus_hs300_3m",
)

CONSTANT_COMPATIBILITY_FEATURES = (
    "avg_turnover_rate_1m",
    "positive_stock_ratio_1m",
    "coverage_ratio",
    "coverage_ratio_1m",
)

VARYING_FEATURES = tuple(
    feature for feature in BASELINE_FEATURES if feature not in CONSTANT_COMPATIBILITY_FEATURES
)


def _max_drawdown(values: np.ndarray) -> float:
    running_max = np.maximum.accumulate(values)
    return float(np.min(values / running_max - 1.0))


def _industry_features(group: pd.DataFrame) -> pd.DataFrame:
    result = group.sort_values("date").copy()
    close = result["close"]
    daily_return = close.pct_change(fill_method=None)
    result["ret_1m"] = close / close.shift(WINDOW_1M) - 1.0
    result["ret_3m"] = close / close.shift(WINDOW_3M) - 1.0
    result["ret_6m"] = close / close.shift(WINDOW_6M) - 1.0
    result["ret_12m"] = close / close.shift(WINDOW_12M) - 1.0
    result["vol_1m"] = daily_return.rolling(WINDOW_1M, min_periods=WINDOW_1M).std()
    result["vol_3m"] = daily_return.rolling(WINDOW_3M, min_periods=WINDOW_3M).std()
    result["drawdown_1m"] = close.rolling(WINDOW_1M, min_periods=WINDOW_1M).apply(
        _max_drawdown, raw=True
    )
    result["avg_turnover_1m"] = result["amount"].rolling(
        WINDOW_1M, min_periods=WINDOW_1M
    ).mean()
    for column in CONSTANT_COMPATIBILITY_FEATURES:
        result[column] = 0.0
    return result


def _market_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("date").copy()
    close = result["close"]
    daily_return = close.pct_change(fill_method=None)
    result["hs300_ret_1m"] = close / close.shift(WINDOW_1M) - 1.0
    result["hs300_ret_3m"] = close / close.shift(WINDOW_3M) - 1.0
    result["hs300_vol_1m"] = daily_return.rolling(WINDOW_1M, min_periods=WINDOW_1M).std()
    result["hs300_vol_3m"] = daily_return.rolling(WINDOW_3M, min_periods=WINDOW_3M).std()
    return result


def build_monthly_panel(
    industry_daily: pd.DataFrame,
    hs300_daily: pd.DataFrame,
    split: SplitSpec,
) -> pd.DataFrame:
    """Build the audited 27-industry monthly panel without forward-filled features."""

    industry = normalise_industry_quotes(industry_daily)
    observed = set(industry["index_code"])
    if observed != set(continuity_codes()):
        raise ValueError("industry quotes must cover the registered 27-industry continuity sample")
    market = normalise_market_quotes(hs300_daily)

    industry = pd.concat(
        [_industry_features(group) for _, group in industry.groupby("index_code", sort=True)],
        ignore_index=True,
    )
    industry["decision_month"] = industry["date"].dt.strftime("%Y%m")
    monthly = (
        industry.sort_values(["index_code", "date"])
        .groupby(["index_code", "decision_month"], as_index=False, sort=True)
        .tail(1)
        .copy()
    )
    monthly = monthly.rename(columns={"close": "decision_close"})
    monthly["decision_date"] = monthly["date"].dt.strftime("%Y%m%d")
    validate_monthly_universe(monthly)

    monthly = monthly.sort_values(["index_code", "decision_date"]).reset_index(drop=True)
    monthly["next_month_return"] = (
        monthly.groupby("index_code")["decision_close"].shift(-1)
        / monthly["decision_close"]
        - 1.0
    )

    market = _market_features(market)
    market["decision_month"] = market["date"].dt.strftime("%Y%m")
    market_monthly = (
        market.sort_values("date").groupby("decision_month", as_index=False, sort=True).tail(1).copy()
    )
    market_monthly = market_monthly.rename(columns={"close": "hs300_close"})
    market_monthly["next_month_hs300_return"] = (
        market_monthly["hs300_close"].shift(-1) / market_monthly["hs300_close"] - 1.0
    )
    market_columns = [
        "decision_month",
        "hs300_close",
        "hs300_ret_1m",
        "hs300_ret_3m",
        "hs300_vol_1m",
        "hs300_vol_3m",
        "next_month_hs300_return",
    ]
    monthly = monthly.merge(market_monthly[market_columns], on="decision_month", how="left")
    monthly["next_month_excess_return"] = (
        monthly["next_month_return"] - monthly["next_month_hs300_return"]
    )
    monthly["industry_minus_hs300_1m"] = monthly["ret_1m"] - monthly["hs300_ret_1m"]
    monthly["industry_minus_hs300_3m"] = monthly["ret_3m"] - monthly["hs300_ret_3m"]

    daily_dates = industry_daily.copy()
    daily_dates["date"] = pd.to_datetime(daily_dates["date"])
    daily_dates["month"] = daily_dates["date"].dt.strftime("%Y%m")
    month_bounds = daily_dates.groupby("month")["date"].agg(["min", "max"]).sort_index()
    months = list(month_bounds.index)
    next_month = {months[index]: months[index + 1] for index in range(len(months) - 1)}
    monthly["next_month_start"] = monthly["decision_month"].map(
        {month: month_bounds.loc[value, "min"].strftime("%Y%m%d") for month, value in next_month.items()}
    )
    monthly["next_month_end"] = monthly["decision_month"].map(
        {month: month_bounds.loc[value, "max"].strftime("%Y%m%d") for month, value in next_month.items()}
    )
    monthly["split"] = monthly["decision_date"].map(split.label)
    monthly = monthly[
        (monthly["decision_date"] <= split.test_end)
        & monthly["next_month_return"].notna()
        & monthly["next_month_hs300_return"].notna()
    ].copy()
    columns = [
        "decision_date",
        "decision_month",
        "next_month_start",
        "next_month_end",
        "split",
        "index_code",
        "index_name",
        "decision_close",
        *BASELINE_FEATURES,
        "next_month_return",
        "next_month_hs300_return",
        "next_month_excess_return",
    ]
    return monthly.loc[:, columns].sort_values(["decision_date", "index_code"]).reset_index(drop=True)
