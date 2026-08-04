# Reproducibility

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

## Public Synthetic Check

```bash
.venv/bin/python examples/generate_synthetic_panel.py --output /tmp/synthetic_panel.csv
.venv/bin/python scripts/run_fixed_split.py --config configs/smoke.yaml --panel /tmp/synthetic_panel.csv --output /tmp/industry-rotation/fixed
.venv/bin/python scripts/run_walk_forward.py --config configs/smoke.yaml --panel /tmp/synthetic_panel.csv --output /tmp/industry-rotation/walk
.venv/bin/python scripts/build_report.py --input /tmp/industry-rotation/walk --output /tmp/industry-rotation/report
```

The synthetic panel is generated from latent factors and random noise with seed 42. It verifies shape, time ordering, deterministic fitting, output generation, and leakage audits. It is not calibrated to reproduce dissertation returns.

## User-supplied Data

Prepare the daily industry and HS300 inputs described in `data_contract.md`, then run either CSV or SQLite construction.

```bash
.venv/bin/python scripts/build_monthly_panel.py --industry-csv path/to/industry.csv --hs300-csv path/to/hs300.csv --config configs/fixed_split.yaml --output data/monthly_panel.csv
.venv/bin/python scripts/run_fixed_split.py --config configs/fixed_split.yaml --panel data/monthly_panel.csv --output outputs/fixed
.venv/bin/python scripts/run_walk_forward.py --config configs/walk_forward.yaml --panel data/monthly_panel.csv --output outputs/walk
```

An optional AKShare path is also available:

```bash
.venv/bin/python scripts/fetch_sw_quotes.py --database data/sw_quotes.sqlite --start-date 20150101 --end-date 20260630
.venv/bin/python scripts/build_monthly_panel.py --database data/sw_quotes.sqlite --config configs/fixed_split.yaml --output data/monthly_panel.csv
```

AKShare and upstream supplier interfaces may change. Inspect the retrieved schema, date coverage, duplicates, and the 27-industry month grid before using the results.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_repository_hygiene.py
.venv/bin/python -m compileall -q src scripts examples tests
.venv/bin/python -m pip check
```

The temporal audit must contain zero `leakage_flag` rows. Run manifests and report manifests contain SHA-256 hashes so result files can be tied to an exact execution.
