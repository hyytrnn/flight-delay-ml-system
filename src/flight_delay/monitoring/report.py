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
    numeric = [col for col in ["distance", "crs_elapsed_time", "origin_elevation"] if col in reference.columns and col in current.columns]
    psi = {
        column: population_stability_index(
            reference.get_column(column).drop_nulls().to_numpy(),
            current.get_column(column).drop_nulls().to_numpy(),
        )
        for column in numeric
    }
    if {"origin", "dest"}.issubset(reference.columns) and {"origin", "dest"}.issubset(
        current.columns
    ):
        ref_routes = set(
            (reference.get_column("origin") + "|" + reference.get_column("dest"))
            .unique()
            .to_list()
        )
        cur_routes = (current.get_column("origin") + "|" + current.get_column("dest")).to_list()
        unseen_route_rate: float | None = (
            float(np.mean([route not in ref_routes for route in cur_routes])) if cur_routes else 0.0
        )
    else:
        unseen_route_rate = None
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reference_rows": reference.height,
        "current_rows": current.height,
        "feature_psi": psi,
        "unseen_route_rate": unseen_route_rate,
        "reference_positive_rate": float(reference.get_column("dep_del15").mean()),
        "current_positive_rate": float(current.get_column("dep_del15").mean()),
        "interpretation": (
            "Data drift is not proof of performance drift; delayed DepDel15 labels are "
            "required for outcome monitoring."
        ),
    }


def write_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

