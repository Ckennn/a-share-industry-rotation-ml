"""Versioned CSV output and compact public reports."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import __version__
from .bootstrap import paired_block_interval
from .metrics import portfolio_statistics, rank_ic_by_month
from .validation import ExperimentResult

CANONICAL_BLOCK_MONTHS = 6
CANONICAL_BOOTSTRAP_SIMULATIONS = 2000
CANONICAL_SEED = 42

STRATEGY_COLORS = {
    "xgboost": "#1f77b4",
    "lightgbm": "#009E73",
    "random_forest": "#D55E00",
    "linear_lasso": "#7B2CBF",
    "momentum_1m": "#E69F00",
    "momentum_3m": "#CC79A7",
    "industry_equal_weight": "#6C757D",
    "hs300": "#222222",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite nonempty directory: {path}")
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _metrics_table(result: ExperimentResult) -> pd.DataFrame:
    rank_lookup = {
        str(model): rank_ic_by_month(group)
        for model, group in result.predictions.groupby("model", sort=True)
    }
    random_annualized = result.random_top3.get("annualized_return", pd.Series(dtype=float))
    rows: list[dict[str, object]] = []
    for strategy, group in result.portfolio.groupby("strategy", sort=True):
        stats = portfolio_statistics(group["net_return"])
        rank_values = rank_lookup.get(str(strategy), pd.Series(dtype=float))
        annualized = float(stats["annualized_return"])
        exceedances = int((random_annualized >= annualized).sum()) if not random_annualized.empty else 0
        rows.append(
            {
                "strategy": strategy,
                **stats,
                "rank_ic_mean": float(rank_values.mean()) if rank_values.notna().any() else np.nan,
                "valid_rank_ic_months": int(rank_values.notna().sum()),
                "mean_gross_selection_excess": float(group["gross_selection_excess"].mean())
                if "gross_selection_excess" in group and group["gross_selection_excess"].notna().any()
                else np.nan,
                "mean_gross_active_return_hs300": float(group["gross_active_return_hs300"].mean())
                if "gross_active_return_hs300" in group and group["gross_active_return_hs300"].notna().any()
                else np.nan,
                "mean_turnover": float(group["turnover"].mean()),
                "total_transaction_cost": float(group["transaction_cost"].sum()),
                "random_top3_exceedances": exceedances,
                "random_top3_simulations": int(len(random_annualized)),
                "random_top3_monte_carlo_p": (exceedances + 1) / (len(random_annualized) + 1)
                if not random_annualized.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_experiment_result(
    result: ExperimentResult,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """Write auditable result tables and a hash-bearing run manifest."""

    output_dir = Path(output_dir)
    _prepare_output(output_dir, overwrite)
    selections = result.selections.copy()
    if "params" in selections:
        selections["params"] = selections["params"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    tables = {
        "predictions.csv": result.predictions,
        "monthly_returns.csv": result.portfolio,
        "metrics.csv": _metrics_table(result),
        "selected_configs.csv": selections,
        "temporal_audit.csv": result.audit,
        "random_top3.csv": result.random_top3,
    }
    paths: list[Path] = []
    for filename, frame in tables.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        paths.append(path)
    manifest = {
        "software_version": __version__,
        "decision_start": str(result.predictions["decision_date"].min()),
        "decision_end": str(result.predictions["decision_date"].max()),
        "prediction_rows": int(len(result.predictions)),
        "models_and_ranked_benchmarks": sorted(result.predictions["model"].astype(str).unique()),
        "leakage_flags": int(result.audit["leakage_flag"].sum()),
        "files": {path.name: _sha256(path) for path in paths},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _plot_cumulative(monthly: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for strategy, group in monthly.groupby("strategy", sort=True):
        ordered = group.sort_values("decision_date")
        cumulative = (1.0 + ordered["net_return"].astype(float)).cumprod() - 1.0
        axis.plot(
            pd.to_datetime(ordered["decision_date"], format="%Y%m%d"),
            cumulative,
            label=str(strategy).replace("_", " ").title(),
            color=STRATEGY_COLORS.get(str(strategy), "#999999"),
            linewidth=2.0,
        )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_title("Net cumulative returns")
    axis.set_ylabel("Cumulative return")
    axis.set_xlabel("Decision month")
    axis.legend(ncol=2, frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _interval_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, group in predictions.groupby("model", sort=True):
        values = rank_ic_by_month(group)
        if not values.notna().any():
            continue
        interval = paired_block_interval(
            values,
            CANONICAL_BLOCK_MONTHS,
            CANONICAL_BOOTSTRAP_SIMULATIONS,
            CANONICAL_SEED,
        )
        rows.append({"strategy": strategy, **interval.__dict__})
    return pd.DataFrame(rows)


def _selection_interval_rows(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ranked = monthly[monthly["gross_selection_excess"].notna()] if "gross_selection_excess" in monthly else monthly.iloc[0:0]
    for strategy, group in ranked.groupby("strategy", sort=True):
        interval = paired_block_interval(
            group.sort_values("decision_date")["gross_selection_excess"],
            CANONICAL_BLOCK_MONTHS,
            CANONICAL_BOOTSTRAP_SIMULATIONS,
            CANONICAL_SEED,
        )
        rows.append({"strategy": strategy, **interval.__dict__})
    return pd.DataFrame(rows)


def _plot_intervals(frame: pd.DataFrame, title: str, ylabel: str, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    if not frame.empty:
        positions = np.arange(len(frame))
        means = frame["mean"].to_numpy(dtype=float)
        errors = np.vstack(
            [means - frame["lower"].to_numpy(dtype=float), frame["upper"].to_numpy(dtype=float) - means]
        )
        colors = [STRATEGY_COLORS.get(str(name), "#999999") for name in frame["strategy"]]
        axis.bar(positions, means, color=colors, alpha=0.85)
        axis.errorbar(positions, means, yerr=errors, fmt="none", ecolor="#222222", capsize=4)
        axis.set_xticks(positions, [str(name).replace("_", " ").title() for name in frame["strategy"]], rotation=25, ha="right")
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_report(output_dir: Path, report_dir: Path) -> list[Path]:
    """Build a compact Markdown and PNG report from already saved outputs."""

    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(output_dir / "predictions.csv", dtype={"decision_date": str, "index_code": str})
    monthly = pd.read_csv(output_dir / "monthly_returns.csv", dtype={"decision_date": str})
    metrics = pd.read_csv(output_dir / "metrics.csv")
    dataset_label = (
        str(predictions["source"].dropna().iloc[0])
        if "source" in predictions and predictions["source"].notna().any()
        else "user-supplied panel"
    )
    summary_path = report_dir / "summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Industry-Rotation Reproduction Summary",
                "",
                f"- Dataset: `{dataset_label}`",
                f"- Decision period: `{predictions['decision_date'].min()}` to `{predictions['decision_date'].max()}`",
                f"- Ranked methods: {', '.join(sorted(predictions['model'].astype(str).unique()))}",
                "- Portfolio rule: Top3 equal weight, one-month holding, 0.1% times one-way target-weight turnover.",
                "- Passive comparisons: Industry Equal Weight and HS300; random Top3 preserves concentration and timing.",
                "",
                "## Interpretation boundary",
                "",
                "Synthetic results verify software behavior only. User-data results remain conditional on the sample, validation window, and benchmark definition and are not investment advice.",
                "",
                "## Metrics",
                "",
                metrics.to_markdown(index=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cumulative_path = report_dir / "cumulative_returns.png"
    rank_path = report_dir / "rank_ic.png"
    selection_path = report_dir / "selection_excess.png"
    _plot_cumulative(monthly, cumulative_path)
    _plot_intervals(_interval_rows(predictions), "Mean monthly Rank IC", "Rank IC", rank_path)
    _plot_intervals(
        _selection_interval_rows(monthly),
        "Gross Top3 selection excess versus Industry Equal Weight",
        "Mean monthly return difference",
        selection_path,
    )
    generated = [summary_path, cumulative_path, rank_path, selection_path]
    manifest_path = report_dir / "file_manifest.json"
    source_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest_path.write_text(
        json.dumps(
            {
                "source_files": {path.name: _sha256(path) for path in source_files},
                "report_files": {path.name: _sha256(path) for path in generated},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated.append(manifest_path)
    return generated
