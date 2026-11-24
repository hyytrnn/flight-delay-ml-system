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


def _prepare_presorted_metric_index(
    scores: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Pre-sort scores descending once so repeated weighted bootstraps need O(N) passes."""

    order = np.argsort(-scores, kind="mergesort")
    s_ord = scores[order]
    y_ord = y_true[order].astype(float)
    is_new = np.empty(len(s_ord), dtype=bool)
    is_new[0] = True
    if len(s_ord) > 1:
        np.not_equal(s_ord[1:], s_ord[:-1], out=is_new[1:])
    gid = np.cumsum(is_new) - 1
    num_groups = int(gid[-1]) + 1
    return order, s_ord, y_ord, gid, num_groups


def _weighted_ranking_metrics(
    weights_sorted: np.ndarray, y_ord: np.ndarray, gid: np.ndarray, num_groups: int
) -> tuple[float, float] | None:
    pos_g = np.bincount(gid, weights=y_ord * weights_sorted, minlength=num_groups)
    tot_g = np.bincount(gid, weights=weights_sorted, minlength=num_groups)
    mask = tot_g > 0
    pg = pos_g[mask]
    fg = tot_g[mask] - pg
    tp = np.cumsum(pg)
    fp = np.cumsum(fg)
    total_pos = float(tp[-1]) if len(tp) else 0.0
    total_neg = float(fp[-1]) if len(fp) else 0.0
    if total_pos <= 0.0 or total_neg <= 0.0:
        return None
    prec = tp / (tp + fp)
    ap = float(np.sum((pg / total_pos) * prec))
    tp_prev = np.concatenate(([0.0], tp[:-1]))
    roc = float(np.sum(fg * (tp_prev + 0.5 * pg)) / (total_pos * total_neg))
    return ap, roc


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
    _, day_id = np.unique(date_values, return_inverse=True)
    n_days = int(day_id.max()) + 1 if len(day_id) else 0

    order, s_ord, y_ord, gid, num_groups = _prepare_presorted_metric_index(p, y)
    day_id_ord = day_id[order]
    pred_ord = (s_ord >= threshold).astype(float)
    sq_err_ord = (y_ord - s_ord) ** 2

    rng = np.random.default_rng(seed)
    metrics_samples: dict[str, list[float]] = {
        "pr_auc": [],
        "roc_auc": [],
        "brier": [],
        "f1": [],
        "positive_rate": [],
    }
    for _ in range(n_bootstrap):
        sampled_days = rng.integers(0, n_days, size=n_days)
        day_w = np.bincount(sampled_days, minlength=n_days).astype(float)
        w_ord = day_w[day_id_ord]
        total_w = float(w_ord.sum())
        if total_w <= 0.0:
            continue
        ranked = _weighted_ranking_metrics(w_ord, y_ord, gid, num_groups)
        if ranked is None:
            continue
        ap, roc = ranked
        brier = float(np.dot(w_ord, sq_err_ord) / total_w)
        tp_thr = float(np.sum(w_ord * y_ord * pred_ord))
        pos_w = float(np.dot(w_ord, y_ord))
        pred_w = float(np.dot(w_ord, pred_ord))
        denom = pred_w + pos_w
        f1 = (2.0 * tp_thr / denom) if denom > 0.0 else 0.0

        metrics_samples["pr_auc"].append(ap)
        metrics_samples["roc_auc"].append(roc)
        metrics_samples["brier"].append(brier)
        metrics_samples["f1"].append(f1)
        metrics_samples["positive_rate"].append(pos_w / total_w)

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


def paired_calendar_block_bootstrap(
    y_true: np.ndarray,
    scores_base: np.ndarray,
    scores_cand: np.ndarray,
    dates: pd.Series,
    *,
    n_bootstrap: int = 300,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute paired block-bootstrap CIs for candidate - baseline over identical rows."""

    y = np.asarray(y_true, dtype=int)
    p_base = np.clip(np.asarray(scores_base, dtype=float), 1e-6, 1.0 - 1e-6)
    p_cand = np.clip(np.asarray(scores_cand, dtype=float), 1e-6, 1.0 - 1e-6)
    if not (len(y) == len(p_base) == len(p_cand) == len(dates)):
        raise ValueError("y_true, scores_base, scores_cand, and dates must have equal length")

    date_values = pd.to_datetime(dates).dt.date.to_numpy()
    _, day_id = np.unique(date_values, return_inverse=True)
    n_days = int(day_id.max()) + 1 if len(day_id) else 0

    ord_b, s_b_ord, y_b_ord, gid_b, n_g_b = _prepare_presorted_metric_index(p_base, y)
    ord_c, s_c_ord, y_c_ord, gid_c, n_g_c = _prepare_presorted_metric_index(p_cand, y)
    day_b_ord = day_id[ord_b]
    day_c_ord = day_id[ord_c]
    sq_b_ord = (y_b_ord - s_b_ord) ** 2
    sq_c_ord = (y_c_ord - s_c_ord) ** 2

    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {
        "delta_pr_auc": [],
        "delta_roc_auc": [],
        "delta_brier": [],
    }
    for _ in range(n_bootstrap):
        sampled_days = rng.integers(0, n_days, size=n_days)
        day_w = np.bincount(sampled_days, minlength=n_days).astype(float)
        wb = day_w[day_b_ord]
        wc = day_w[day_c_ord]
        tot_w = float(wb.sum())
        if tot_w <= 0.0:
            continue
        rb = _weighted_ranking_metrics(wb, y_b_ord, gid_b, n_g_b)
        rc = _weighted_ranking_metrics(wc, y_c_ord, gid_c, n_g_c)
        if rb is None or rc is None:
            continue
        brier_b = float(np.dot(wb, sq_b_ord) / tot_w)
        brier_c = float(np.dot(wc, sq_c_ord) / tot_w)
        deltas["delta_pr_auc"].append(rc[0] - rb[0])
        deltas["delta_roc_auc"].append(rc[1] - rb[1])
        deltas["delta_brier"].append(brier_c - brier_b)

    summary: dict[str, dict[str, float]] = {}
    for key, values in deltas.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "prob_positive": float(np.mean(arr > 0.0)),
            "n_boot": len(values),
        }
    return summary
