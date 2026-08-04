"""Cross-sectional ranking and portfolio evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic_for_month(
    frame: pd.DataFrame,
    score_column: str = "predicted_score",
    target_column: str = "next_month_excess_return",
) -> float:
    """Spearman correlation between scores and realised cross-sectional returns."""

    valid = frame[[score_column, target_column]].dropna()
    if valid[score_column].nunique() < 2 or valid[target_column].nunique() < 2:
        return float("nan")
    return float(valid[score_column].rank(method="average").corr(valid[target_column].rank(method="average")))


def rank_ic_by_month(
    predictions: pd.DataFrame,
    score_column: str = "predicted_score",
    target_column: str = "next_month_excess_return",
) -> pd.Series:
    values = {
        str(decision_date): rank_ic_for_month(group, score_column, target_column)
        for decision_date, group in predictions.groupby("decision_date", sort=True)
    }
    return pd.Series(values, name="rank_ic", dtype=float)


def portfolio_statistics(monthly_returns: pd.Series) -> dict[str, float]:
    """Risk metrics from monthly arithmetic returns and zero risk-free rate."""

    values = pd.Series(monthly_returns, dtype=float).dropna()
    if values.empty:
        return {
            "months": 0.0,
            "cumulative_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "maximum_drawdown": float("nan"),
        }
    wealth = (1.0 + values).cumprod()
    cumulative = float(wealth.iloc[-1] - 1.0)
    annualized = float(wealth.iloc[-1] ** (12.0 / len(values)) - 1.0)
    monthly_std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    annualized_volatility = monthly_std * np.sqrt(12.0)
    sharpe = (
        float(values.mean() / monthly_std * np.sqrt(12.0))
        if np.isfinite(monthly_std) and monthly_std > 0
        else float("nan")
    )
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "months": float(len(values)),
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(drawdown.min()),
    }


def prediction_summary(predictions: pd.DataFrame, top_n: int = 3) -> dict[str, float]:
    """Summarise model ranking and Top3 selection on a common prediction frame."""

    rank_values = rank_ic_by_month(predictions)
    top = (
        predictions.sort_values(
            ["decision_date", "predicted_score", "index_code"],
            ascending=[True, False, True],
        )
        .groupby("decision_date", sort=True)
        .head(top_n)
    )
    mse = float(
        np.mean(
            np.square(
                predictions["next_month_excess_return"].to_numpy(dtype=float)
                - predictions["predicted_score"].to_numpy(dtype=float)
            )
        )
    )
    return {
        "top3_mean_target_return": float(top["next_month_excess_return"].mean()),
        "rank_ic_mean": float(rank_values.mean()) if rank_values.notna().any() else float("nan"),
        "mse": mse,
        "valid_rank_ic_months": float(rank_values.notna().sum()),
    }
