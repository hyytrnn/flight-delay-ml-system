"""Create a transparent data, prediction score, and delayed-outcome drift report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import brier_score_loss

from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame
from flight_delay.models.calibrate_v2 import evaluate_binary_metrics
from flight_delay.models.predictor import ModelBundle, load_bundle

DEFAULT_MONITORING_CONFIG: dict[str, Any] = {
    "thresholds": {
        "psi_warning": 0.15,
        "psi_critical": 0.25,
        "unseen_route_rate_warning": 0.03,
        "unseen_route_rate_critical": 0.06,
        "ece_warning": 0.04,
        "brier_increase_warning": 0.025,
    },
    "monitored_numeric_columns": ["distance", "crs_elapsed_time", "origin_elevation"],
    "histogram_bins": 10,
}


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    reference_hist, _ = np.histogram(reference, bins=edges)
    current_hist, _ = np.histogram(current, bins=edges)
    reference_rate = np.clip(reference_hist / max(reference_hist.sum(), 1), 1e-6, 1)
    current_rate = np.clip(current_hist / max(current_hist.sum(), 1), 1e-6, 1)
    return float(np.sum((current_rate - reference_rate) * np.log(current_rate / reference_rate)))


def _load_monitoring_config(config_path: Path | None) -> dict[str, Any]:
    cfg = {
        "thresholds": dict(DEFAULT_MONITORING_CONFIG["thresholds"]),
        "monitored_numeric_columns": list(DEFAULT_MONITORING_CONFIG["monitored_numeric_columns"]),
        "histogram_bins": int(DEFAULT_MONITORING_CONFIG["histogram_bins"]),
    }
    if config_path is not None and config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if "thresholds" in loaded and isinstance(loaded["thresholds"], dict):
            cfg["thresholds"].update(loaded["thresholds"])
        if "monitored_numeric_columns" in loaded:
            cfg["monitored_numeric_columns"] = list(loaded["monitored_numeric_columns"])
        if "histogram_bins" in loaded:
            cfg["histogram_bins"] = int(loaded["histogram_bins"])
    return cfg


def _score_bundle(frame: pl.DataFrame, bundle: ModelBundle) -> np.ndarray:
    group: FeatureGroup = getattr(bundle, "feature_group", "F0")  # type: ignore[assignment]
    feats = build_v2_feature_frame(frame, group)
    return bundle.predict_proba(feats)


def build_report(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    bundle_path: Path | None = None,
    config_path: Path | None = None,
    window_label: str | None = None,
) -> dict[str, object]:
    cfg = _load_monitoring_config(config_path)
    bins = int(cfg["histogram_bins"])
    thresholds = cfg["thresholds"]

    numeric = [
        col
        for col in cfg["monitored_numeric_columns"]
        if col in reference.columns and col in current.columns
    ]
    psi = {
        column: population_stability_index(
            reference.get_column(column).drop_nulls().to_numpy(),
            current.get_column(column).drop_nulls().to_numpy(),
            bins=bins,
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

    score_psi: float | None = None
    score_histogram: dict[str, Any] | None = None
    predicted_score_summary: dict[str, Any] | None = None
    outcome_metrics: dict[str, float] | None = None
    reference_brier: float | None = None
    brier_delta: float | None = None
    model_version: str | None = None

    if bundle_path is not None:
        bundle = load_bundle(bundle_path)
        model_version = str(bundle.metadata.get("model_version", "unknown"))
        cur_probs = _score_bundle(current, bundle)
        counts, edges = np.histogram(cur_probs, bins=np.linspace(0.0, 1.0, bins + 1))
        score_histogram = {
            "bin_edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts],
            "proportions": [float(c / max(len(cur_probs), 1)) for c in counts],
        }
        predicted_score_summary = {
            "mean": float(np.mean(cur_probs)),
            "std": float(np.std(cur_probs)),
            "p50": float(np.quantile(cur_probs, 0.50)),
            "p90": float(np.quantile(cur_probs, 0.90)),
            "p95": float(np.quantile(cur_probs, 0.95)),
            "threshold_default": float(bundle.threshold),
            "alert_rate_default": float(np.mean(cur_probs >= bundle.threshold)),
        }
        p35_thr = bundle.metadata.get("threshold_precision_target_35")
        if p35_thr is not None:
            predicted_score_summary["threshold_precision_target_35"] = float(p35_thr)
            predicted_score_summary["alert_rate_precision_target_35"] = float(
                np.mean(cur_probs >= float(p35_thr))
            )

        ref_probs = _score_bundle(reference, bundle)
        score_psi = population_stability_index(ref_probs, cur_probs, bins=bins)

        if "dep_del15" in current.columns:
            y_cur = current.get_column("dep_del15").to_numpy().astype(int)
            if len(np.unique(y_cur)) >= 2:
                outcome_metrics = evaluate_binary_metrics(
                    y_cur, cur_probs, threshold=float(bundle.threshold)
                )
        if "dep_del15" in reference.columns:
            y_ref = reference.get_column("dep_del15").to_numpy().astype(int)
            if len(np.unique(y_ref)) >= 2:
                reference_brier = float(brier_score_loss(y_ref, ref_probs))
                if outcome_metrics is not None:
                    brier_delta = float(outcome_metrics["brier"] - reference_brier)

    psi_warn_cols = [c for c, val in psi.items() if val >= float(thresholds["psi_warning"])]
    psi_crit_cols = [c for c, val in psi.items() if val >= float(thresholds["psi_critical"])]
    unseen_status = "unknown"
    if unseen_route_rate is not None:
        if unseen_route_rate >= float(thresholds["unseen_route_rate_critical"]):
            unseen_status = "critical"
        elif unseen_route_rate >= float(thresholds["unseen_route_rate_warning"]):
            unseen_status = "warning"
        else:
            unseen_status = "ok"

    ece_warn = bool(
        outcome_metrics is not None and outcome_metrics["ece"] >= float(thresholds["ece_warning"])
    )
    brier_warn = bool(
        brier_delta is not None and brier_delta >= float(thresholds["brier_increase_warning"])
    )
    reasons: list[str] = []
    if psi_crit_cols:
        reasons.append(f"critical_feature_psi:{','.join(psi_crit_cols)}")
    if unseen_status == "critical":
        reasons.append("critical_unseen_route_rate")
    if ece_warn:
        reasons.append(f"ece_warning:{outcome_metrics['ece']:.4f}>={thresholds['ece_warning']}")
    if brier_warn:
        reasons.append(
            f"brier_increase_warning:{brier_delta:.4f}>={thresholds['brier_increase_warning']}"
        )

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "window_label": window_label,
        "model_version": model_version,
        "reference_rows": reference.height,
        "current_rows": current.height,
        "feature_psi": psi,
        "score_psi": score_psi,
        "unseen_route_rate": unseen_route_rate,
        "reference_positive_rate": float(reference.get_column("dep_del15").mean()),
        "current_positive_rate": float(current.get_column("dep_del15").mean()),
        "score_histogram": score_histogram,
        "predicted_score_summary": predicted_score_summary,
        "outcome_metrics": outcome_metrics,
        "reference_brier": reference_brier,
        "brier_delta": brier_delta,
        "alerts": {
            "psi_warning_features": psi_warn_cols,
            "psi_critical_features": psi_crit_cols,
            "unseen_route_status": unseen_status,
            "ece_warning": ece_warn,
            "brier_warning": brier_warn,
            "retraining_recommended": bool(reasons),
            "alert_reasons": reasons,
        },
        "interpretation": (
            "Data drift is not proof of performance drift; delayed DepDel15 labels are "
            "required for outcome monitoring."
        ),
    }


def write_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
