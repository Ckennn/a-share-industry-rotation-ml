"""Fixed Top3 portfolio accounting and public benchmark construction."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .metrics import portfolio_statistics


def select_top_n(frame: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if top_n < 1 or top_n > len(frame):
        raise ValueError("top_n must be between one and the cross-section size")
    result = frame.sort_values(
        ["predicted_score", "index_code"], ascending=[False, True]
    ).head(top_n).copy()
    result["target_weight"] = 1.0 / top_n
    return result


def one_way_turnover(
    previous_weights: Mapping[str, float], current_weights: Mapping[str, float]
) -> float:
    """Target-weight turnover; within-month pre-trade drift is not reconstructed."""

    if not previous_weights:
        return 1.0
    codes = set(previous_weights) | set(current_weights)
    return 0.5 * sum(
        abs(float(current_weights.get(code, 0.0)) - float(previous_weights.get(code, 0.0)))
        for code in codes
    )


def _backtest_one_strategy(
    predictions: pd.DataFrame,
    strategy: str,
    cost_rate: float,
    top_n: int,
) -> pd.DataFrame:
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for decision_date, month in predictions.groupby("decision_date", sort=True):
        holdings = select_top_n(month, top_n=top_n)
        current = dict(zip(holdings["index_code"], holdings["target_weight"], strict=True))
        turnover = one_way_turnover(previous, current)
        gross_return = float(
            np.dot(
                holdings["target_weight"].to_numpy(dtype=float),
                holdings["next_month_return"].to_numpy(dtype=float),
            )
        )
        industry_equal_weight = float(month["next_month_return"].mean())
        hs300 = float(month["next_month_hs300_return"].mean())
        transaction_cost = float(cost_rate * turnover)
        rows.append(
            {
                "decision_date": str(decision_date),
                "strategy": strategy,
                "gross_return": gross_return,
                "net_return": gross_return - transaction_cost,
                "industry_equal_weight_return": industry_equal_weight,
                "hs300_return": hs300,
                "gross_selection_excess": gross_return - industry_equal_weight,
                "gross_active_return_hs300": gross_return - hs300,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "holdings": ",".join(holdings["index_code"].astype(str)),
            }
        )
        previous = current
    return pd.DataFrame(rows)


def backtest_scores(
    predictions: pd.DataFrame,
    cost_rate: float,
    top_n: int = 3,
) -> pd.DataFrame:
    """Backtest score-ranked portfolios, separately for each model if present."""

    required = {
        "decision_date",
        "index_code",
        "predicted_score",
        "next_month_return",
        "next_month_hs300_return",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    if "model" not in predictions:
        return _backtest_one_strategy(predictions, "model", cost_rate, top_n)
    frames = [
        _backtest_one_strategy(group, str(model), cost_rate, top_n)
        for model, group in predictions.groupby("model", sort=True)
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def benchmark_score_frames(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create transparent momentum score frames on the model's common panel."""

    frames: dict[str, pd.DataFrame] = {}
    for strategy, feature in (("momentum_1m", "ret_1m"), ("momentum_3m", "ret_3m")):
        frame = panel.copy()
        frame["predicted_score"] = frame[feature]
        frame["model"] = strategy
        frames[strategy] = frame
    return frames


def passive_monthly_returns(panel: pd.DataFrame, cost_rate: float) -> pd.DataFrame:
    """Industry equal weight and HS300 on the same decision months."""

    rows: list[dict[str, object]] = []
    for decision_date, month in panel.groupby("decision_date", sort=True):
        for strategy, gross in (
            ("industry_equal_weight", float(month["next_month_return"].mean())),
            ("hs300", float(month["next_month_hs300_return"].mean())),
        ):
            turnover = 0.0
            cost = 0.0
            rows.append(
                {
                    "decision_date": str(decision_date),
                    "strategy": strategy,
                    "gross_return": gross,
                    "net_return": gross - cost,
                    "turnover": turnover,
                    "transaction_cost": cost,
                }
            )
    return pd.DataFrame(rows)


def simulate_random_top3(
    panel: pd.DataFrame,
    simulations: int,
    cost_rate: float,
    seed: int = 42,
    top_n: int = 3,
) -> pd.DataFrame:
    """Simulate random monthly Top3 portfolios with the same concentration and costs."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    grouped = [(date, group.copy()) for date, group in panel.groupby("decision_date", sort=True)]
    for simulation in range(simulations):
        previous: dict[str, float] = {}
        monthly: list[float] = []
        for _, month in grouped:
            selected = rng.choice(len(month), size=top_n, replace=False)
            holdings = month.iloc[selected]
            current = {str(code): 1.0 / top_n for code in holdings["index_code"]}
            turnover = one_way_turnover(previous, current)
            gross = float(holdings["next_month_return"].mean())
            monthly.append(gross - cost_rate * turnover)
            previous = current
        stats = portfolio_statistics(pd.Series(monthly))
        rows.append({"simulation": float(simulation), **stats})
    return pd.DataFrame(rows)
