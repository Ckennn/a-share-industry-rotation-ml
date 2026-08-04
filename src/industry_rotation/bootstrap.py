"""Deterministic circular moving-block bootstrap inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntervalResult:
    mean: float
    lower: float
    upper: float
    observations: int
    block_months: int
    simulations: int
    seed: int


def circular_block_indices(
    observations: int,
    block_months: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if observations < 1 or block_months < 1:
        raise ValueError("observations and block_months must be positive")
    blocks = int(np.ceil(observations / block_months))
    starts = rng.integers(0, observations, size=blocks)
    indices = np.concatenate(
        [(start + np.arange(block_months, dtype=int)) % observations for start in starts]
    )
    return indices[:observations]


def paired_block_interval(
    values: pd.Series,
    block_months: int,
    simulations: int,
    seed: int,
) -> IntervalResult:
    clean = pd.Series(values, dtype=float).dropna().to_numpy(dtype=float)
    if clean.size == 0:
        raise ValueError("bootstrap requires at least one finite observation")
    rng = np.random.default_rng(seed)
    means = np.empty(simulations, dtype=float)
    for simulation in range(simulations):
        indices = circular_block_indices(len(clean), block_months, rng)
        means[simulation] = float(clean[indices].mean())
    lower, upper = np.quantile(means, [0.025, 0.975])
    return IntervalResult(
        mean=float(clean.mean()),
        lower=float(lower),
        upper=float(upper),
        observations=int(clean.size),
        block_months=int(block_months),
        simulations=int(simulations),
        seed=int(seed),
    )
