from __future__ import annotations

import polars as pl

from flight_delay.monitoring.report import build_report, population_stability_index


def test_psi_is_zero_for_identical_values() -> None:
    values = pl.Series([1.0, 2.0, 3.0, 4.0]).to_numpy()
    assert population_stability_index(values, values) == 0.0


def test_monitoring_report_separates_drift_from_performance() -> None:
    reference = pl.DataFrame(
        {"distance": [100.0, 200.0], "crs_elapsed_time": [60.0, 90.0], "dep_del15": [0, 1]}
    )
    current = pl.DataFrame(
        {"distance": [800.0, 900.0], "crs_elapsed_time": [180.0, 200.0], "dep_del15": [1, 1]}
    )
    report = build_report(reference, current)
    assert "not proof of performance drift" in str(report["interpretation"])

