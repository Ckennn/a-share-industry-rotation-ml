"""Quote schema adapters and fail-closed data checks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .samples import continuity_codes, continuity_names


class DataContractError(ValueError):
    """Raised when input quotes or panel coverage violate the contract."""


INTERNAL_QUOTE_COLUMNS = (
    "index_code",
    "index_name",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

SUPPLIER_COLUMN_MAP = {
    "代码": "index_code",
    "名称": "index_name",
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}


def _normalise_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(columns)].isna().any().any():
        raise DataContractError("quote numeric columns contain invalid values")
    return result


def normalise_industry_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """Map supplier columns to the stable internal industry quote schema."""

    result = frame.rename(columns=SUPPLIER_COLUMN_MAP).copy()
    required = {"index_code", "date", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(result.columns)
    if missing:
        raise DataContractError(f"required quote columns are missing: {sorted(missing)}")
    result["index_code"] = result["index_code"].astype(str).str.strip()
    if "index_name" not in result:
        result["index_name"] = result["index_code"].map(continuity_names()).fillna(result["index_code"])
    else:
        result["index_name"] = result["index_name"].astype(str).str.strip()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise DataContractError("quote dates contain invalid values")
    result = _normalise_numeric(result, ("open", "high", "low", "close", "volume", "amount"))
    if (result["close"] <= 0).any():
        raise DataContractError("close values must be positive")
    if result.duplicated(["index_code", "date"]).any():
        raise DataContractError("duplicate index_code + date keys")
    return result.loc[:, INTERNAL_QUOTE_COLUMNS].sort_values(["date", "index_code"]).reset_index(drop=True)


def normalise_market_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise an HS300 quote frame without adding an industry identifier."""

    result = frame.rename(columns=SUPPLIER_COLUMN_MAP).copy()
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(result.columns)
    if missing:
        raise DataContractError(f"required market quote columns are missing: {sorted(missing)}")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise DataContractError("market quote dates contain invalid values")
    result = _normalise_numeric(result, ("open", "high", "low", "close", "volume", "amount"))
    if (result["close"] <= 0).any() or result.duplicated(["date"]).any():
        raise DataContractError("market quotes require positive closes and unique dates")
    return result.loc[:, ["date", "open", "high", "low", "close", "volume", "amount"]].sort_values("date").reset_index(drop=True)


def validate_monthly_universe(frame: pd.DataFrame) -> None:
    """Require every month to contain exactly the registered 27 industries."""

    required = {"decision_date", "index_code"}
    if not required.issubset(frame.columns):
        raise DataContractError("monthly universe check requires decision_date and index_code")
    expected = set(continuity_codes())
    for decision_date, group in frame.groupby("decision_date", sort=True):
        observed = set(group["index_code"].astype(str))
        if observed != expected or len(group) != len(expected):
            raise DataContractError(
                f"decision month {decision_date} must contain exactly 27 continuity industries"
            )


def load_sqlite_table(path: Path, table: str) -> pd.DataFrame:
    """Load an explicitly named table from a user-supplied SQLite database."""

    if not path.exists():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as connection:
        available = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if table not in available:
            raise DataContractError(f"SQLite table not found: {table}")
        return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
