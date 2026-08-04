# A-share Industry Rotation ML

This repository provides a code-only reproduction of a controlled comparison of XGBoost, LightGBM, Random Forest, and Linear Lasso for monthly Chinese A-share industry ranking. The models estimate each industry's next-month return relative to the HS300. A fixed portfolio rule then ranks the 27 scores, holds the top three industries at equal weights for one month, and applies a cost of `0.001 x one-way target-weight turnover`.

The evidence motivating this software is conditional: model rankings can change across estimation windows and evaluation protocols. The repository is therefore designed to expose benchmarks and temporal audits, not to present a permanent winning strategy.

## Scope

Included:

- the 27-industry Shenwan continuity-sample registry;
- daily quote validation and monthly feature construction;
- the fixed train/validation/test protocol;
- prior-only monthly walk-forward selection and refitting;
- tree models, Linear Lasso, momentum, passive, and Random Top3 benchmarks;
- Rank IC, selection-excess, portfolio-risk, bootstrap, and leakage-audit utilities;
- a deterministic synthetic panel for tests and examples.

Excluded:

- dissertation text and presentation files;
- downloaded papers, databases, market data, trained models, and empirical outputs;
- post-hoc diagnostic specifications outside the core comparison protocol.

## Data Boundary

SWS Research is the publisher of the Shenwan index series. AKShare is an access interface, not the data publisher. This repository does not redistribute the historical quotes used in the study. Users must obtain data under the applicable provider and interface terms. See [docs/data_contract.md](docs/data_contract.md).

## Installation

Python 3.11 or newer is required.

```bash
git clone git@github.com:Ckennn/a-share-industry-rotation-ml.git
cd a-share-industry-rotation-ml
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

## Synthetic Smoke Workflow

The following single shell command regenerates the public fixture, runs both evaluation protocols, and builds the walk-forward report:

```bash
rm -rf /tmp/industry-rotation-smoke && .venv/bin/python examples/generate_synthetic_panel.py --output /tmp/synthetic_panel.csv && .venv/bin/python scripts/run_fixed_split.py --config configs/smoke.yaml --panel /tmp/synthetic_panel.csv --output /tmp/industry-rotation-smoke/fixed && .venv/bin/python scripts/run_walk_forward.py --config configs/smoke.yaml --panel /tmp/synthetic_panel.csv --output /tmp/industry-rotation-smoke/walk && .venv/bin/python scripts/build_report.py --input /tmp/industry-rotation-smoke/walk --output /tmp/industry-rotation-smoke/report
```

Synthetic results test the software and do not reproduce empirical performance.

## User-data Workflow

Users may either supply quote files that satisfy the public schema or use the optional AKShare downloader:

```bash
.venv/bin/python scripts/fetch_sw_quotes.py --database data/sw_quotes.sqlite --start-date 20150101 --end-date 20260630
.venv/bin/python scripts/build_monthly_panel.py --database data/sw_quotes.sqlite --config configs/fixed_split.yaml --output data/monthly_panel.csv
.venv/bin/python scripts/run_fixed_split.py --config configs/fixed_split.yaml --panel data/monthly_panel.csv --output outputs/fixed
.venv/bin/python scripts/run_walk_forward.py --config configs/walk_forward.yaml --panel data/monthly_panel.csv --output outputs/walk
.venv/bin/python scripts/build_report.py --input outputs/walk --output outputs/walk_report
```

The downloader is optional because supplier interfaces and data licences can change. Validate the retrieved fields and coverage before interpreting results.

## Outputs

Each experiment writes:

- `predictions.csv`
- `monthly_returns.csv`
- `metrics.csv`
- `selected_configs.csv`
- `temporal_audit.csv`
- `random_top3.csv`
- `run_manifest.json`

The report builder adds `summary.md`, three compact PNG figures, and a SHA-256 file manifest.

## Leakage Controls

The fixed split trains on `2015-01..2022-12`, selects configurations on `2023-01..2023-12`, and evaluates decisions from `2024-01..2026-05`. The selected static model is fitted on the training split only. The clean walk-forward begins in `2021-01`, reserves the latest 12 available prior months for validation, requires at least 60 earlier core-training months, refits on all labelled prior observations, and then predicts the current cross-section. Audit checks fail when training, validation, or refitting reaches the decision date, or when a target has not yet been realised.

## Limitations

The continuity sample is smaller than the current Shenwan first-level classification. Monthly decisions provide fewer independent time observations than the industry-month row count suggests. The baseline information set is narrow, the turnover calculation uses changes in target weights rather than within-month drifted weights, and repeated historical diagnostics cannot create a fresh prospective test set.

This software is provided for research and informational purposes only. It is not investment advice or a recommendation to trade any security.

## Citation and License

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The code is released under the [MIT License](LICENSE).
