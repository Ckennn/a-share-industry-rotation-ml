import pandas as pd
import pytest

from industry_rotation.audit import LeakageError, assert_prior_only
from industry_rotation.config import ExperimentConfig
from industry_rotation.validation import run_fixed_split, run_walk_forward


def test_fixed_split_never_fits_on_validation_or_test(
    monthly_panel: pd.DataFrame,
    smoke_config: ExperimentConfig,
) -> None:
    result = run_fixed_split(monthly_panel, smoke_config)
    assert result.audit["training_end"].max() <= "20221231"
    assert result.audit["validation_end"].max() <= "20231231"
    assert result.audit["refit_end"].max() <= "20221231"
    assert result.predictions["decision_date"].min() > "20231231"


def test_fixed_split_includes_common_naive_and_passive_benchmarks(
    monthly_panel: pd.DataFrame,
    smoke_config: ExperimentConfig,
) -> None:
    result = run_fixed_split(monthly_panel, smoke_config)
    assert {
        "momentum_1m",
        "momentum_3m",
        "industry_equal_weight",
        "hs300",
    }.issubset(set(result.portfolio["strategy"]))
    assert len(result.random_top3) == smoke_config.random_top3_simulations


def test_fixed_split_test_labels_do_not_select_configuration(
    monthly_panel: pd.DataFrame,
    smoke_config: ExperimentConfig,
) -> None:
    first = run_fixed_split(monthly_panel, smoke_config)
    changed = monthly_panel.copy()
    test = changed["decision_date"] > smoke_config.split.validation_end
    changed.loc[test, "next_month_excess_return"] *= -100.0
    second = run_fixed_split(changed, smoke_config)
    pd.testing.assert_frame_equal(first.selections, second.selections)
    pd.testing.assert_series_equal(
        first.predictions["predicted_score"], second.predictions["predicted_score"]
    )


def test_walk_forward_boundaries_are_strictly_prior_only(
    monthly_panel: pd.DataFrame,
    smoke_config: ExperimentConfig,
) -> None:
    result = run_walk_forward(monthly_panel, smoke_config)
    assert not result.audit["leakage_flag"].any()
    assert (result.audit["training_end"] < result.audit["decision_date"]).all()
    assert (result.audit["validation_end"] < result.audit["decision_date"]).all()
    assert (result.audit["refit_end"] < result.audit["decision_date"]).all()
    assert (result.audit["target_realisation_end"] <= result.audit["decision_date"]).all()


def test_audit_rejects_equal_boundary() -> None:
    bad = pd.DataFrame({"decision_date": ["20240131"], "refit_end": ["20240131"]})
    with pytest.raises(LeakageError):
        assert_prior_only(bad)
