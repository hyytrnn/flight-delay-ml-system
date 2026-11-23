"""Create a transparent data and prediction drift report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    reference_hist, _ = np.histogram(reference, bins=edges)
    current_hist, _ = np.histogram(current, bins=edges)
    reference_rate = np.clip(reference_hist / max(reference_hist.sum(), 1), 1e-6, 1)
    current_rate = np.clip(current_hist / max(current_hist.sum(), 1), 1e-6, 1)
    return float(np.sum((current_rate - reference_rate) * np.log(current_rate / reference_rate)))


def build_report(reference: pl.DataFrame, current: pl.DataFrame) -> dict[str, object]:
    numeric = ["distance", "crs_elapsed_time"]
    psi = {
        column: population_stability_index(
            reference.get_column(column).drop_nulls().to_numpy(),
            current.get_column(column).drop_nulls().to_numpy(),
        )
        for column in numeric
    }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reference_rows": reference.height,
        "current_rows": current.height,
        "feature_psi": psi,
        "reference_positive_rate": float(reference.get_column("dep_del15").mean()),
        "current_positive_rate": float(current.get_column("dep_del15").mean()),
        "interpretation": "Data drift is not proof of performance drift; labels are required.",
    }


def write_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

