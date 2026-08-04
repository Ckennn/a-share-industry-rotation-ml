"""Fail-closed temporal-boundary audit helpers."""

from __future__ import annotations

import pandas as pd


class LeakageError(RuntimeError):
    """Raised when fitting or selection reaches the decision date."""


STRICT_PRIOR_COLUMNS = (
    "training_end",
    "validation_end",
    "config_validation_end",
    "refit_end",
)


def leakage_flags(audit: pd.DataFrame) -> pd.Series:
    if "decision_date" not in audit:
        raise LeakageError("audit requires decision_date")
    decision = audit["decision_date"].astype(str)
    flags = pd.Series(False, index=audit.index)
    found = False
    for column in STRICT_PRIOR_COLUMNS:
        if column in audit:
            found = True
            raw = audit[column]
            flags |= raw.notna() & (raw.astype(str) >= decision)
    if "target_realisation_end" in audit:
        found = True
        raw = audit["target_realisation_end"]
        flags |= raw.notna() & (raw.astype(str) > decision)
    if not found:
        raise LeakageError("audit contains no temporal boundary columns")
    return flags


def assert_prior_only(audit: pd.DataFrame) -> None:
    flags = leakage_flags(audit)
    if flags.any():
        columns = ["decision_date", *[name for name in STRICT_PRIOR_COLUMNS if name in audit]]
        if "target_realisation_end" in audit:
            columns.append("target_realisation_end")
        raise LeakageError(f"non-prior temporal boundary: {audit.loc[flags, columns].to_dict('records')}")
