"""Separated V2 probability calibration, threshold selection, and block-bootstrap metrics."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

CalibrationMethod = Literal["none", "platt", "isotonic"]


@dataclass(frozen=True)
class FittedCalibrator:
    method: CalibrationMethod
    estimator: Any

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(raw_scores, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.method == "none" or self.estimator is None:
            return clipped
        if self.method == "isotonic":
            return np.clip(np.asarray(self.estimator.predict(clipped), dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return np.clip(
            np.asarray(self.estimator.predict_proba(logits)[:, 1], dtype=float),
            1e-6,
            1.0 - 1e-6,
        )


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, *, n_bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y))
    for low, high in itertools.pairwise(edges):
        if high == 1.0:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def precision_at_top_k(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    n = len(y_true)
    k = max(1, round(n * fraction))
    order = np.argsort(scores)[::-1][:k]
    return float(np.asarray(y_true)[order].mean())


def fit_calibrator(
    method: CalibrationMethod, raw_scores: np.ndarray, y_true: np.ndarray, seed: int = 42
) -> FittedCalibrator:
    clipped = np.clip(np.asarray(raw_scores, dtype=float), 1e-6, 1.0 - 1e-6)
    if method == "none":
        return FittedCalibrator(method="none", estimator=None)
    if method == "isotonic":
        iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        iso.fit(clipped, y_true)
        return FittedCalibrator(method="isotonic", estimator=iso)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    platt = LogisticRegression(C=1_000.0, solver="lbfgs", random_state=seed)
    platt.fit(logits, y_true)
    return FittedCalibrator(method="platt", estimator=platt)


def evaluate_binary_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    preds = (p >= threshold).astype(int)
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, p))
    return {
        "rows": float(len(y)),
        "positive_rate": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_uplift_abs": pr_auc - prevalence,
        "pr_auc_uplift_rel": (pr_auc / prevalence - 1.0) if prevalence > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - p, p]))),
        "ece": expected_calibration_error(y, p),
        "precision_at_top_5pct": precision_at_top_k(y, p, 0.05),
        "precision_at_top_10pct": precision_at_top_k(y, p, 0.10),
        "threshold": float(threshold),
    }


def threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    fn_cost: float = 1.0,
    fp_cost: float = 0.25,
) -> pd.DataFrame:
    candidates = np.round(np.linspace(0.05, 0.80, 151), 4)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    rows: list[dict[str, float]] = []
    n = max(1, len(y))
    for thr in candidates:
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "alert_rate": float(pred.mean()),
                "expected_cost_per_flight": float((fn_cost * fn + fp_cost * fp) / n),
            }
        )
    return pd.DataFrame(rows)


def calendar_block_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    dates: pd.Series,
    threshold: float,
    *,
    n_bootstrap: int = 300,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Resample whole flight_date blocks to preserve within-day correlation."""

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    date_values = pd.to_datetime(dates).dt.date.to_numpy()
    unique_days = np.unique(date_values)
    day_indices = {day: np.where(date_values == day)[0] for day in unique_days}

    rng = np.random.default_rng(seed)
    metrics_samples: dict[str, list[float]] = {
        "pr_auc": [],
        "roc_auc": [],
        "brier": [],
        "f1": [],
        "positive_rate": [],
    }
    for _ in range(n_bootstrap):
        sampled_days = rng.choice(unique_days, size=len(unique_days), replace=True)
        idx = np.concatenate([day_indices[d] for d in sampled_days])
        yb = y[idx]
        pb = p[idx]
        if yb.min() == yb.max():
            continue
        metrics_samples["pr_auc"].append(float(average_precision_score(yb, pb)))
        metrics_samples["roc_auc"].append(float(roc_auc_score(yb, pb)))
        metrics_samples["brier"].append(float(brier_score_loss(yb, pb)))
        metrics_samples["f1"].append(float(f1_score(yb, (pb >= threshold).astype(int), zero_division=0)))
        metrics_samples["positive_rate"].append(float(yb.mean()))

    summary: dict[str, dict[str, float]] = {}
    for key, values in metrics_samples.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "n_boot": len(values),
        }
    return summary
