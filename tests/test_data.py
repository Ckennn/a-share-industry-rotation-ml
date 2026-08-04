import pandas as pd
import pytest

from industry_rotation.data import (
    DataContractError,
    normalise_industry_quotes,
    validate_monthly_universe,
)
from industry_rotation.samples import continuity_codes


def test_continuity_sample_has_27_unique_codes() -> None:
    codes = continuity_codes()
    assert len(codes) == 27
    assert len(set(codes)) == 27
    assert {"801950", "801960", "801970", "801980"}.isdisjoint(codes)


def test_duplicate_index_dates_are_rejected(daily_bundle: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    industry_daily, _ = daily_bundle
    duplicated = pd.concat([industry_daily, industry_daily.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate"):
        normalise_industry_quotes(duplicated)


def test_supplier_columns_are_normalised_and_numeric() -> None:
    frame = pd.DataFrame(
        {
            "代码": ["801010"],
            "名称": ["Agriculture"],
            "日期": ["2024-01-31"],
            "开盘": ["100"],
            "最高": ["101"],
            "最低": ["99"],
            "收盘": ["100.5"],
            "成交量": ["1000"],
            "成交额": ["2000"],
        }
    )
    result = normalise_industry_quotes(frame)
    assert result.loc[0, "index_code"] == "801010"
    assert result.loc[0, "date"] == pd.Timestamp("2024-01-31")
    assert result.loc[0, "close"] == pytest.approx(100.5)


def test_required_quote_columns_are_enforced() -> None:
    with pytest.raises(DataContractError, match="required"):
        normalise_industry_quotes(pd.DataFrame({"index_code": ["801010"]}))


def test_missing_industry_in_month_is_rejected() -> None:
    frame = pd.DataFrame(
        {"decision_date": ["20240131"] * 26, "index_code": list(continuity_codes())[:-1]}
    )
    with pytest.raises(DataContractError, match="27"):
        validate_monthly_universe(frame)
