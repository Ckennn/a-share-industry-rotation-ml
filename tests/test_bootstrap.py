import numpy as np
import pandas as pd

from industry_rotation.bootstrap import circular_block_indices, paired_block_interval


def test_block_bootstrap_is_reproducible() -> None:
    values = pd.Series(np.arange(24, dtype=float))
    first = paired_block_interval(values, 6, 2000, 42)
    second = paired_block_interval(values, 6, 2000, 42)
    assert first == second


def test_circular_blocks_preserve_requested_length() -> None:
    first = circular_block_indices(10, 3, np.random.default_rng(42))
    second = circular_block_indices(10, 3, np.random.default_rng(42))
    assert len(first) == 10
    np.testing.assert_array_equal(first, second)


def test_paired_interval_drops_missing_values() -> None:
    result = paired_block_interval(pd.Series([1.0, np.nan, 3.0]), 2, 100, 42)
    assert result.observations == 2
    assert result.mean == 2.0
