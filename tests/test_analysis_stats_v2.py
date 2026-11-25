from __future__ import annotations

import math

import numpy as np
import pytest

from flight_delay.analysis.stats import (
    jensen_shannon_divergence,
    population_stability_index,
    total_variation_distance,
    wilson_interval,
)


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(20, 100)
    assert low < 0.2 < high


def test_wilson_interval_rejects_no_support_with_nan() -> None:
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_categorical_drift_is_zero_for_identical_distributions() -> None:
    distribution = {"AA": 0.6, "UA": 0.4}
    assert total_variation_distance(distribution, distribution) == 0
    assert jensen_shannon_divergence(distribution, distribution) == pytest.approx(0)


def test_categorical_drift_detects_shift() -> None:
    reference = {"AA": 0.9, "UA": 0.1}
    current = {"AA": 0.2, "UA": 0.8}
    assert total_variation_distance(reference, current) == pytest.approx(0.7)
    assert jensen_shannon_divergence(reference, current) > 0.3


def test_population_stability_index_detects_numeric_shift() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, size=2_000)
    same = rng.normal(0.0, 1.0, size=2_000)
    shifted = rng.normal(2.5, 1.0, size=2_000)
    assert population_stability_index(ref, same) < 0.05
    assert population_stability_index(ref, shifted) > 0.5
