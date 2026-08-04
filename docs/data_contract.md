# Data Contract

## Provenance

SWS Research publishes the Shenwan industry indices. AKShare exposes an interface through which users may retrieve those published series. These roles are distinct: AKShare is an access interface and is not identified here as the original publisher. No market data are committed to this repository.

Users are responsible for provider terms, interface changes, and any right required to download or store data. The optional fetch script is deliberately separate from the research pipeline.

## Daily Quote Schema

One row represents one index on one trading date.

| Column | Type | Rule |
|---|---|---|
| `index_code` | string | Six-digit Shenwan code |
| `index_name` | string | Display name; filled from the registry when absent |
| `date` | date | Unique with `index_code` |
| `open` | numeric | Required raw field |
| `high` | numeric | Required raw field |
| `low` | numeric | Required raw field |
| `close` | numeric | Strictly positive |
| `volume` | numeric | Required raw field |
| `amount` | numeric | Trading amount |

The supplier adapter accepts the corresponding Chinese AKShare headings and converts them immediately to this internal schema. Downstream modules do not depend on supplier-language column names.

The baseline predictors use closing-price paths, trading amount, and HS300 closing-price paths. Open, high, low, and volume remain part of the validated raw contract but are not direct varying predictors in the baseline specification.

## Continuity Sample

Every retained decision month must contain exactly the 27 codes in `industry_rotation.samples.CONTINUITY_SAMPLE`. Coal (`801950`), Petroleum and Petrochemicals (`801960`), Environmental Protection (`801970`), and Beauty Care (`801980`) are excluded because their histories are incomplete for the common 2015 start. Missing industries and duplicate `index_code + date` keys fail validation; genuine coverage gaps are not forward-filled.

## Monthly Panel Schema

Required identifiers and labels are:

- `decision_date`, `decision_month`, `next_month_start`, `next_month_end`;
- `index_code`, `index_name`, and `split`;
- all 18 registered feature columns documented in `methodology.md`;
- `next_month_return`, `next_month_hs300_return`, and `next_month_excess_return`.

The target is attached only after sorting each industry by decision date. Early long-lookback features remain missing and are imputed inside the relevant training or refit sample. The public synthetic fixture adds `source=synthetic_test_fixture` and contains no empirical observations.
