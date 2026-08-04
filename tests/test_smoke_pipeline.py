from pathlib import Path

import pandas as pd

from examples.generate_synthetic_panel import generate_synthetic_panel
from industry_rotation.config import ExperimentConfig
from industry_rotation.features import BASELINE_FEATURES
from industry_rotation.reporting import build_report, write_experiment_result
from industry_rotation.validation import run_fixed_split, run_walk_forward


def test_synthetic_panel_has_expected_shape(synthetic_panel: pd.DataFrame) -> None:
    assert synthetic_panel["index_code"].nunique() == 27
    assert synthetic_panel["decision_date"].nunique() == 137
    assert len(synthetic_panel) == 27 * 137
    assert synthetic_panel.groupby("decision_date")["index_code"].nunique().eq(27).all()
    assert set(BASELINE_FEATURES).issubset(synthetic_panel.columns)
    assert synthetic_panel["source"].eq("synthetic_test_fixture").all()


def test_smoke_fixed_and_walk_forward_write_audited_outputs(
    tmp_path: Path,
    smoke_config: ExperimentConfig,
    synthetic_panel: pd.DataFrame,
) -> None:
    fixed = run_fixed_split(synthetic_panel, smoke_config)
    walk = run_walk_forward(synthetic_panel, smoke_config)
    fixed_dir = tmp_path / "fixed"
    walk_dir = tmp_path / "walk"
    report_dir = tmp_path / "report"
    write_experiment_result(fixed, fixed_dir)
    write_experiment_result(walk, walk_dir)
    generated = build_report(walk_dir, report_dir)
    assert (fixed_dir / "predictions.csv").exists()
    assert (fixed_dir / "metrics.csv").exists()
    assert (walk_dir / "temporal_audit.csv").exists()
    assert (walk_dir / "run_manifest.json").exists()
    assert (report_dir / "summary.md") in generated
    assert (report_dir / "cumulative_returns.png").exists()
    assert not walk.audit["leakage_flag"].any()


def test_result_writer_refuses_nonempty_directory(
    tmp_path: Path,
    smoke_config: ExperimentConfig,
    synthetic_panel: pd.DataFrame,
) -> None:
    result = run_fixed_split(synthetic_panel, smoke_config)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("do not replace", encoding="utf-8")
    try:
        write_experiment_result(result, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("nonempty output directory must require overwrite=True")
