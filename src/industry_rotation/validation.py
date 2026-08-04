"""Fixed-split and prior-only walk-forward model comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .audit import assert_prior_only, leakage_flags
from .config import ExperimentConfig
from .features import BASELINE_FEATURES
from .metrics import prediction_summary
from .models import build_estimator
from .portfolio import (
    backtest_scores,
    benchmark_score_frames,
    passive_monthly_returns,
    simulate_random_top3,
)


@dataclass(frozen=True)
class SelectedConfig:
    model: str
    candidate_id: str
    params: Mapping[str, object]
    top3_mean_target_return: float
    rank_ic_mean: float
    mse: float


@dataclass(frozen=True)
class ExperimentResult:
    predictions: pd.DataFrame
    portfolio: pd.DataFrame
    audit: pd.DataFrame
    selections: pd.DataFrame
    random_top3: pd.DataFrame


def _prediction_frame(frame: pd.DataFrame, model: str, candidate_id: str, scores: np.ndarray) -> pd.DataFrame:
    columns = [
        "decision_date",
        "decision_month",
        "next_month_start",
        "next_month_end",
        "index_code",
        "index_name",
        "next_month_return",
        "next_month_hs300_return",
        "next_month_excess_return",
        "source",
    ]
    result = frame.loc[:, [column for column in columns if column in frame]].copy()
    result["model"] = model
    result["candidate_id"] = candidate_id
    result["predicted_score"] = scores
    result["predicted_rank"] = result.groupby("decision_date")["predicted_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return result


def _fit_predict(
    model_name: str,
    params: Mapping[str, object],
    train: pd.DataFrame,
    predict: pd.DataFrame,
    config: ExperimentConfig,
) -> np.ndarray:
    estimator = build_estimator(
        model_name,
        params,
        BASELINE_FEATURES,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    estimator.fit(train.loc[:, BASELINE_FEATURES], train["next_month_excess_return"].astype(float))
    return np.asarray(estimator.predict(predict.loc[:, BASELINE_FEATURES]), dtype=float)


def select_configuration(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: ExperimentConfig,
) -> SelectedConfig:
    """Select only from the versioned grid using validation data."""

    if model_name not in config.models:
        raise ValueError(f"model is not configured: {model_name}")
    evaluated: list[SelectedConfig] = []
    for candidate in config.models[model_name]:
        scores = _fit_predict(model_name, candidate.params, train, validation, config)
        predictions = _prediction_frame(validation, model_name, candidate.candidate_id, scores)
        metrics = prediction_summary(predictions, top_n=config.portfolio.top_n)
        evaluated.append(
            SelectedConfig(
                model=model_name,
                candidate_id=candidate.candidate_id,
                params=dict(candidate.params),
                top3_mean_target_return=float(metrics["top3_mean_target_return"]),
                rank_ic_mean=float(metrics["rank_ic_mean"]),
                mse=float(metrics["mse"]),
            )
        )

    def key(item: SelectedConfig) -> tuple[float, float, float]:
        top3 = item.top3_mean_target_return if np.isfinite(item.top3_mean_target_return) else -np.inf
        rank_ic = item.rank_ic_mean if np.isfinite(item.rank_ic_mean) else -np.inf
        return top3, rank_ic, -item.mse

    return max(evaluated, key=key)


def _selection_row(selected: SelectedConfig, decision_date: str | None = None) -> dict[str, object]:
    row = asdict(selected)
    row["params"] = dict(selected.params)
    if decision_date is not None:
        row["decision_date"] = decision_date
    return row


def _evaluate_with_benchmarks(
    panel: pd.DataFrame,
    model_predictions: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark_predictions = pd.concat(
        list(benchmark_score_frames(panel).values()), ignore_index=True
    )
    predictions = pd.concat([model_predictions, benchmark_predictions], ignore_index=True)
    ranked_portfolios = backtest_scores(
        predictions,
        cost_rate=config.portfolio.cost_rate,
        top_n=config.portfolio.top_n,
    )
    passive = passive_monthly_returns(panel, config.portfolio.cost_rate)
    portfolio = pd.concat([ranked_portfolios, passive], ignore_index=True, sort=False)
    random_top3 = simulate_random_top3(
        panel,
        simulations=config.random_top3_simulations,
        cost_rate=config.portfolio.cost_rate,
        seed=config.random_state,
        top_n=config.portfolio.top_n,
    )
    return predictions, portfolio, random_top3


def run_fixed_split(panel: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
    """Select on 2023, fit on training only, and evaluate the held-out test."""

    train = panel[panel["decision_date"] <= config.split.train_end].copy()
    validation = panel[
        (panel["decision_date"] > config.split.train_end)
        & (panel["decision_date"] <= config.split.validation_end)
    ].copy()
    test = panel[
        (panel["decision_date"] > config.split.validation_end)
        & (panel["decision_date"] <= config.split.test_end)
    ].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("fixed split requires non-empty training, validation, and test sets")

    prediction_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    first_test_date = str(test["decision_date"].min())
    for model_name in sorted(config.models):
        selected = select_configuration(model_name, train, validation, config)
        scores = _fit_predict(model_name, selected.params, train, test, config)
        prediction_frames.append(
            _prediction_frame(test, model_name, selected.candidate_id, scores)
        )
        selection_rows.append(_selection_row(selected))
        audit_rows.append(
            {
                "protocol": "fixed_split",
                "model": model_name,
                "decision_date": first_test_date,
                "training_start": str(train["decision_date"].min()),
                "training_end": str(train["decision_date"].max()),
                "validation_start": str(validation["decision_date"].min()),
                "validation_end": str(validation["decision_date"].max()),
                "config_validation_end": str(validation["decision_date"].max()),
                "target_realisation_end": str(validation["next_month_end"].max()),
                "refit_end": str(train["decision_date"].max()),
                "candidate_id": selected.candidate_id,
            }
        )
    model_predictions = pd.concat(prediction_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    audit["leakage_flag"] = leakage_flags(audit)
    assert_prior_only(audit)
    selections = pd.DataFrame(selection_rows).sort_values("model").reset_index(drop=True)
    predictions, portfolio, random_top3 = _evaluate_with_benchmarks(
        test, model_predictions, config
    )
    return ExperimentResult(predictions, portfolio, audit, selections, random_top3)


def run_walk_forward(panel: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
    """Repeat prior-only selection, refitting, and prediction each month."""

    dates = sorted(
        date
        for date in panel["decision_date"].astype(str).unique()
        if config.walk_forward_start <= date <= config.walk_forward_end
    )
    prediction_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    required_months = config.min_core_train_months + config.validation_months

    for decision_date in dates:
        predict = panel[panel["decision_date"] == decision_date].copy()
        prior = panel[
            (panel["decision_date"] < decision_date)
            & (panel["next_month_end"].astype(str) <= decision_date)
        ].copy()
        prior_months = sorted(prior["decision_date"].astype(str).unique())
        if len(prior_months) < required_months:
            continue
        validation_months = prior_months[-config.validation_months :]
        core_months = prior_months[: -config.validation_months]
        if len(core_months) < config.min_core_train_months:
            continue
        train_core = prior[prior["decision_date"].isin(core_months)].copy()
        validation = prior[prior["decision_date"].isin(validation_months)].copy()

        for model_name in sorted(config.models):
            selected = select_configuration(model_name, train_core, validation, config)
            scores = _fit_predict(model_name, selected.params, prior, predict, config)
            prediction_frames.append(
                _prediction_frame(predict, model_name, selected.candidate_id, scores)
            )
            selection_rows.append(_selection_row(selected, decision_date))
            audit_rows.append(
                {
                    "protocol": "clean_walk_forward",
                    "model": model_name,
                    "decision_date": decision_date,
                    "training_start": str(train_core["decision_date"].min()),
                    "training_end": str(train_core["decision_date"].max()),
                    "validation_start": str(validation["decision_date"].min()),
                    "validation_end": str(validation["decision_date"].max()),
                    "config_validation_end": str(validation["decision_date"].max()),
                    "target_realisation_end": str(prior["next_month_end"].max()),
                    "refit_end": str(prior["decision_date"].max()),
                    "candidate_id": selected.candidate_id,
                }
            )
    if not prediction_frames:
        raise ValueError("no walk-forward month has enough prior training and validation history")
    model_predictions = pd.concat(prediction_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    audit["leakage_flag"] = leakage_flags(audit)
    assert_prior_only(audit)
    selections = pd.DataFrame(selection_rows).sort_values(["decision_date", "model"]).reset_index(drop=True)
    evaluated_dates = set(model_predictions["decision_date"].astype(str))
    evaluation_panel = panel[panel["decision_date"].astype(str).isin(evaluated_dates)].copy()
    predictions, portfolio, random_top3 = _evaluate_with_benchmarks(
        evaluation_panel, model_predictions, config
    )
    return ExperimentResult(predictions, portfolio, audit, selections, random_top3)
