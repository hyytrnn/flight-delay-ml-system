from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_delay.models.calibrate_v2 import (
    calendar_block_bootstrap_ci,
    evaluate_binary_metrics,
    expected_calibration_error,
    fit_calibrator,
    threshold_sweep,
)


def test_calibrators_and_ece_preserve_probability_bounds() -> None:
    rng = np.random.default_rng(42)
    raw = np.clip(rng.uniform(0.05, 0.95, size=300), 1e-4, 1 - 1e-4)
    y = (rng.uniform(0, 1, size=300) < raw).astype(int)
    for method in ["none", "platt", "isotonic"]:
        cal = fit_calibrator(method, raw, y, seed=42)  # type: ignore[arg-type]
        preds = cal.predict(raw)
        assert np.all((preds > 0.0) & (preds < 1.0))
        metrics = evaluate_binary_metrics(y, preds, threshold=0.5)
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= expected_calibration_error(y, preds) <= 1.0


def test_threshold_sweep_and_block_bootstrap_ci() -> None:
    rng = np.random.default_rng(7)
    dates = pd.Series(pd.date_range("2025-09-01", periods=10, freq="D").repeat(20))
    scores = rng.uniform(0.1, 0.9, size=len(dates))
    y = (scores > 0.55).astype(int)
    sweep = threshold_sweep(y, scores)
    assert not sweep.empty
    assert {"threshold", "f1", "expected_cost_per_flight"}.issubset(sweep.columns)

    ci = calendar_block_bootstrap_ci(y, scores, dates, threshold=0.5, n_bootstrap=40, seed=7)
    assert "pr_auc" in ci
    assert ci["pr_auc"]["ci95_low"] <= ci["pr_auc"]["mean"] <= ci["pr_auc"]["ci95_high"]
    assert ci["pr_auc"]["n_boot"] == pytest.approx(40, abs=5)
