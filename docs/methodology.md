# Methodology

## Prediction Task

For industry `i` at decision month `t`, the model uses information available by that month-end to estimate

```text
next_month_excess_return = next_month_return - next_month_hs300_return
```

The output is a continuous score. The portfolio layer ranks all 27 scores and holds the highest three industries at equal weights during the following month. Models do not choose TopK, weights, the holding period, or the transaction-cost assumption.

## Registered Predictors

The public registry contains 18 columns.

| Family | Predictors | Daily source |
|---|---|---|
| Industry momentum | `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m` | Industry close |
| Industry risk | `vol_1m`, `vol_3m`, `drawdown_1m` | Industry close |
| Trading activity | `avg_turnover_1m` | Industry amount |
| Compatibility fields | `avg_turnover_rate_1m`, `positive_stock_ratio_1m`, `coverage_ratio`, `coverage_ratio_1m` | Unavailable at official-index level; fixed at zero |
| Market state | `hs300_ret_1m`, `hs300_ret_3m`, `hs300_vol_1m`, `hs300_vol_3m` | HS300 close |
| Relative momentum | `industry_minus_hs300_1m`, `industry_minus_hs300_3m` | Industry and HS300 close |

The four compatibility fields preserve the audited 18-column schema but have zero variance and cannot create a tree split. Economic interpretation is limited to the 14 varying predictors. Trading-day windows are fixed at 21, 63, 126, and 252 observations. Missing early lookbacks are not forward-filled.

## Models and Selection

The supervised comparison contains XGBoost, LightGBM, Random Forest, and Linear Lasso. Model grids are versioned in `configs/`. Every estimator uses median imputation fitted only on its training or refit sample. Lasso additionally uses training-fitted standardisation. Random seeds are fixed at 42 and model jobs at one.

Candidate selection uses validation Top3 mean target return first, mean monthly Rank IC second, and lower mean squared error third. In the fixed protocol, the selected model is fitted on the 2015-2022 training split rather than refitted on 2023. In the walk-forward protocol, selection is repeated using a trailing 12-month validation block and the selected estimator is refitted on all labelled prior observations.

## Portfolio and Benchmarks

- **Model Top3:** equal weights, one-month holding, monthly target-weight rebalancing.
- **1M and 3M momentum:** rank the same industries by trailing one- or three-month return.
- **Industry Equal Weight:** passive exposure to the same 27-index universe.
- **HS300:** broad-market comparison.
- **Random Top3:** random monthly sets of three industries with the same concentration, timing, and Top3 cost rule.
- **Linear Lasso:** formal supervised benchmark using the same target and predictors.

Top3 transaction cost is `0.001 x one-way turnover`. Initial Top3 turnover is one. Later turnover is half the absolute change in target weights across all industries. This is a consistent target-weight approximation and does not reconstruct within-month pre-trade weight drift. Passive benchmarks are reported without strategy turnover costs.

## Evaluation

- **Rank IC:** monthly Spearman correlation between scores and realised next-month excess returns.
- **Gross selection excess:** Top3 gross return minus Industry Equal Weight return.
- **Gross active return:** Top3 gross return minus HS300 return.
- **Portfolio metrics:** geometric annualised return, annualised volatility, zero-risk-free-rate Sharpe ratio, and maximum drawdown.
- **Random Top3 comparison:** exceedance count and finite-simulation one-sided probability based on annualised net return.
- **Uncertainty:** circular moving-block bootstrap with six-month blocks, 2,000 simulations, and seed 42.

No one metric establishes persistent forecasting skill. Ranking, selection, passive exposure, random concentration, costs, and risk must be read together.
