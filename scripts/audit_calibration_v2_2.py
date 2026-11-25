"""V2.2 calibration drift audit: Brier decomposition, multi-bin ECE, subgroup ECE & temp scaling."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame
from flight_delay.models.calibrate_v2 import (
    evaluate_binary_metrics,
    fit_calibrator,
    paired_calendar_block_bootstrap,
)
from flight_delay.models.predictor import load_bundle


def _ece(y: np.ndarray, p: np.ndarray, *, n_bins: int = 10, strategy: str = "uniform") -> float:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p_clip, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 3:
            return 0.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_clip, edges[1:-1], right=False), 0, len(edges) - 2)
    n = max(len(y), 1)
    ece_val = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        if np.any(mask):
            ece_val += (np.sum(mask) / n) * abs(float(y[mask].mean()) - float(p_clip[mask].mean()))
    return float(ece_val)


def _brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Murphy (1973) Brier decomposition: Brier = Reliability - Resolution + Uncertainty."""
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_clip, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(y)
    y_bar = float(y.mean())
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        nk = int(np.sum(mask))
        if nk == 0:
            continue
        fk = float(p_clip[mask].mean())
        ok = float(y[mask].mean())
        reliability += (nk / n) * ((fk - ok) ** 2)
        resolution += (nk / n) * ((ok - y_bar) ** 2)
    uncertainty = y_bar * (1.0 - y_bar)
    return {
        "brier": float(np.mean((y - p_clip) ** 2)),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def _fit_temperature_scaler(raw_probs: np.ndarray, y: np.ndarray) -> float:
    p_clip = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(p_clip / (1.0 - p_clip))

    def _nll(temp: float) -> float:
        scaled = 1.0 / (1.0 + np.exp(-logits / temp))
        return float(log_loss(y, np.clip(scaled, 1e-6, 1.0 - 1e-6)))

    res = minimize_scalar(_nll, bounds=(0.25, 4.0), method="bounded")
    return float(res.x)


def _apply_temperature(raw_probs: np.ndarray, temp: float) -> np.ndarray:
    p_clip = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(p_clip / (1.0 - p_clip))
    return np.clip(1.0 / (1.0 + np.exp(-logits / temp)), 1e-6, 1.0 - 1e-6)


def main() -> int:
    bundle = load_bundle(Path("models/flight_delay_v2_1_bundle.joblib"))
    fg: FeatureGroup = bundle.feature_group  # type: ignore[assignment]
    dev = (
        pl.scan_parquet("data/processed/v2/year=*/month=*/flights.parquet")
        .filter((pl.col("flight_date") >= date(2025, 2, 1)) & (pl.col("flight_date") < date(2025, 9, 1)))
        .collect(engine="streaming")
        .sort("flight_date")
    )
    feats = build_v2_feature_frame(dev, fg)
    raw_probs = bundle.predict_proba(feats)
    meta = dev.select("flight_date", "reporting_airline", "origin", "dest", "dep_del15").to_pandas()
    meta["year_month"] = pd.to_datetime(meta["flight_date"]).dt.strftime("%Y-%m")
    meta["raw_prob"] = raw_probs
    meta["route_known"] = [
        f"{o}|{d}" in bundle.route_lookup for o, d in zip(meta["origin"], meta["dest"], strict=True)
    ]

    fit_mask = meta["year_month"].isin(["2025-02", "2025-03"]).to_numpy()
    eval_mask = meta["year_month"].isin(["2025-04", "2025-05", "2025-06", "2025-07", "2025-08"]).to_numpy()
    y_fit = meta.loc[fit_mask, "dep_del15"].to_numpy().astype(int)
    p_fit = meta.loc[fit_mask, "raw_prob"].to_numpy()
    y_eval = meta.loc[eval_mask, "dep_del15"].to_numpy().astype(int)
    p_eval_raw = meta.loc[eval_mask, "raw_prob"].to_numpy()
    dates_eval = pd.to_datetime(meta.loc[eval_mask, "flight_date"])

    platt = fit_calibrator("platt", p_fit, y_fit, seed=42)
    iso = fit_calibrator("isotonic", p_fit, y_fit, seed=42)
    temp = _fit_temperature_scaler(p_fit, y_fit)

    candidates = {
        "none_frozen": p_eval_raw,
        "platt_feb_mar": platt.predict(p_eval_raw),
        "isotonic_feb_mar": iso.predict(p_eval_raw),
        f"temperature_T{temp:.3f}": _apply_temperature(p_eval_raw, temp),
    }

    method_rows: list[dict[str, Any]] = []
    for name, probs in candidates.items():
        m = evaluate_binary_metrics(y_eval, probs, threshold=bundle.threshold)
        decomp = _brier_decomposition(y_eval, probs, n_bins=10)
        ci = (
            paired_calendar_block_bootstrap(y_eval, p_eval_raw, probs, dates_eval, n_bootstrap=200, seed=42)
            if name != "none_frozen"
            else None
        )
        method_rows.append(
            {
                "method": name,
                **m,
                "ece_15_uniform": _ece(y_eval, probs, n_bins=15, strategy="uniform"),
                "ece_20_uniform": _ece(y_eval, probs, n_bins=20, strategy="uniform"),
                "ece_10_quantile": _ece(y_eval, probs, n_bins=10, strategy="quantile"),
                "brier_reliability": decomp["reliability"],
                "brier_resolution": decomp["resolution"],
                "brier_uncertainty": decomp["uncertainty"],
                "delta_brier_vs_none": ci["delta_brier"]["mean"] if ci else 0.0,
                "delta_brier_ci95_low": ci["delta_brier"]["ci95_low"] if ci else 0.0,
                "delta_brier_ci95_high": ci["delta_brier"]["ci95_high"] if ci else 0.0,
            }
        )

    monthly_rows: list[dict[str, Any]] = []
    for ym, grp in meta.groupby("year_month", sort=True):
        yg = grp["dep_del15"].to_numpy().astype(int)
        pg = grp["raw_prob"].to_numpy()
        mg = evaluate_binary_metrics(yg, pg, threshold=bundle.threshold)
        dg = _brier_decomposition(yg, pg, n_bins=10)
        monthly_rows.append(
            {
                "year_month": str(ym),
                "rows": len(grp),
                "positive_rate": mg["positive_rate"],
                "ece_10_uniform": mg["ece"],
                "ece_15_uniform": _ece(yg, pg, n_bins=15, strategy="uniform"),
                "ece_10_quantile": _ece(yg, pg, n_bins=10, strategy="quantile"),
                "brier": dg["brier"],
                "brier_reliability": dg["reliability"],
                "brier_resolution": dg["resolution"],
            }
        )

    subgroup_rows: list[dict[str, Any]] = []
    eval_df = meta.loc[eval_mask]
    for col, vals in [
        ("reporting_airline", eval_df["reporting_airline"].value_counts().head(10).index.tolist()),
        ("route_known", [True, False]),
    ]:
        for val in vals:
            sub = eval_df.loc[eval_df[col] == val]
            if len(sub) < 500:
                continue
            ys = sub["dep_del15"].to_numpy().astype(int)
            ps = sub["raw_prob"].to_numpy()
            ms = evaluate_binary_metrics(ys, ps, threshold=bundle.threshold)
            subgroup_rows.append(
                {
                    "slice_col": col,
                    "slice_val": str(val),
                    "rows": len(sub),
                    "positive_rate": ms["positive_rate"],
                    "mean_predicted_prob": float(ps.mean()),
                    "ece_10_uniform": ms["ece"],
                    "brier": ms["brier"],
                    "pr_auc": ms["pr_auc"],
                }
            )

    # Reliability diagram + score histogram figure across 2025-04..08
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    edges = np.linspace(0.0, 1.0, 11)
    for ym in ["2025-04", "2025-06", "2025-08"]:
        sub = meta.loc[meta["year_month"] == ym]
        ys = sub["dep_del15"].to_numpy()
        ps = sub["raw_prob"].to_numpy()
        idx = np.clip(np.digitize(ps, edges[1:-1], right=False), 0, 9)
        xs, ys_emp = [], []
        for b in range(10):
            m = idx == b
            if np.sum(m) >= 50:
                xs.append(float(ps[m].mean()))
                ys_emp.append(float(ys[m].mean()))
        axes[0].plot(xs, ys_emp, marker="o", label=ym)
    axes[0].plot([0, 0.7], [0, 0.7], "--", color="gray", label="Perfect")
    axes[0].set_title("Monthly Reliability (2025-04, 06, 08)")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Empirical DepDel15 Rate")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].hist(p_eval_raw, bins=25, color="#0f5b78", alpha=0.85, edgecolor="white")
    axes[1].axvline(bundle.threshold, color="crimson", linestyle="--", label=f"thr={bundle.threshold}")
    axes[1].set_title("Predicted Score Histogram (2025-04..08)")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Flight Count")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig_path = Path("reports/figures/calibration_v2_2_reliability.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_version": bundle.metadata["model_version"],
        "calibrator_fit_window": "2025-02-01..2025-03-31",
        "forward_eval_window": "2025-04-01..2025-08-31",
        "note": "Prospective 2025-10..11 holdout is excluded from calibrator selection/tuning.",
        "temperature_fitted": temp,
        "calibrator_comparison": method_rows,
        "monthly_brier_decomposition": monthly_rows,
        "subgroup_calibration": subgroup_rows,
        "figure_path": str(fig_path.as_posix()),
    }
    out_json = Path("reports/monitoring/calibration_audit_v2_2.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(pd.DataFrame(method_rows)[["method", "brier", "log_loss", "ece", "ece_15_uniform", "pr_auc", "delta_brier_vs_none"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
