import numpy as np
import pandas as pd
import pytest

from industry_rotation.metrics import portfolio_statistics, rank_ic_for_month
from industry_rotation.portfolio import (
    backtest_scores,
    one_way_turnover,
    passive_monthly_returns,
    select_top_n,
)


def test_top3_weights_and_target_weight_turnover(prediction_month: pd.DataFrame) -> None:
    holdings = select_top_n(prediction_month, top_n=3)
    assert holdings["target_weight"].tolist() == pytest.approx([1 / 3] * 3)
    assert holdings["target_weight"].sum() == pytest.approx(1.0)
    assert one_way_turnover({}, dict(zip(holdings.index_code, holdings.target_weight))) == pytest.approx(1.0)


def test_subsequent_turnover_uses_target_weight_changes() -> None:
    previous = {"a": 0.5, "b": 0.5}
    current = {"b": 0.5, "c": 0.5}
    assert one_way_turnover(previous, current) == pytest.approx(0.5)


def test_rank_ic_is_nan_for_constant_scores(prediction_month: pd.DataFrame) -> None:
    prediction_month["predicted_score"] = 1.0
    assert np.isnan(rank_ic_for_month(prediction_month))


def test_backtest_applies_cost_and_reports_two_benchmarks(prediction_month: pd.DataFrame) -> None:
    result = backtest_scores(prediction_month, cost_rate=0.001, top_n=3)
    row = result.iloc[0]
    assert row["turnover"] == pytest.approx(1.0)
    assert row["transaction_cost"] == pytest.approx(0.001)
    assert row["net_return"] == pytest.approx(row["gross_return"] - 0.001)
    assert row["gross_selection_excess"] == pytest.approx(
        row["gross_return"] - row["industry_equal_weight_return"]
    )
    assert row["gross_active_return_hs300"] == pytest.approx(
        row["gross_return"] - row["hs300_return"]
    )


def test_portfolio_statistics_are_geometric_and_risk_adjusted() -> None:
    returns = pd.Series([0.10, -0.05, 0.02])
    stats = portfolio_statistics(returns)
    expected = float(np.prod(1.0 + returns) ** 4 - 1.0)
    assert stats["annualized_return"] == pytest.approx(expected)
    assert stats["annualized_volatility"] > 0
    assert stats["maximum_drawdown"] < 0


def test_passive_benchmarks_are_reported_without_strategy_turnover(
    prediction_month: pd.DataFrame,
) -> None:
    passive = passive_monthly_returns(prediction_month, cost_rate=0.001)
    assert passive["turnover"].eq(0.0).all()
    assert passive["transaction_cost"].eq(0.0).all()
    assert passive["net_return"].equals(passive["gross_return"])
