"""Shared frozen-model fitting, airport lookup extraction, and eval plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from flight_delay.features.v2 import FeatureGroup, v2_feature_columns
from flight_delay.models.predictor import CatBoostBundleEstimator, CategoricalFrameTransformer
from flight_delay.models.tuning_v2 import (
    ModelName,
    _bounded_best_iterations,
    _inner_chronological_split,
    _onehot_preprocessor,
)


def airport_lookup_from_frame(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    cols = [
        c
        for c in [
            "origin",
            "origin_timezone",
            "origin_latitude",
            "origin_longitude",
            "origin_elevation",
            "origin_subdivision",
        ]
        if c in frame.columns
    ]
    if "origin" in cols:
        for row in frame.select(cols).unique(subset=["origin"]).to_dicts():
            code = str(row["origin"]).upper()
            lookup[code] = {
                "timezone": row.get("origin_timezone"),
                "latitude": row.get("origin_latitude"),
                "longitude": row.get("origin_longitude"),
                "elevation": row.get("origin_elevation"),
                "subdivision": row.get("origin_subdivision") or "UNKNOWN",
            }
    dest_cols = [
        c
        for c in [
            "dest",
            "dest_timezone",
            "dest_latitude",
            "dest_longitude",
            "dest_elevation",
            "dest_subdivision",
        ]
        if c in frame.columns
    ]
    if "dest" in dest_cols:
        for row in frame.select(dest_cols).unique(subset=["dest"]).to_dicts():
            code = str(row["dest"]).upper()
            if code not in lookup:
                lookup[code] = {
                    "timezone": row.get("dest_timezone"),
                    "latitude": row.get("dest_latitude"),
                    "longitude": row.get("dest_longitude"),
                    "elevation": row.get("dest_elevation"),
                    "subdivision": row.get("dest_subdivision") or "UNKNOWN",
                }
    return lookup


def fit_frozen_model(
    *,
    model_name: ModelName,
    params: dict[str, Any],
    feature_group: FeatureGroup,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    dates_train: pd.Series,
    seed: int,
) -> Any:
    """Fit inner early-stopped probe, then refit on 100% of training data cleanly."""

    numeric, categorical = v2_feature_columns(feature_group)
    run_params = dict(params)
    inner_split = _inner_chronological_split(dates_train, fraction=0.15)

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        train = x_train.copy()
        for col in categorical:
            train[col] = train[col].astype("string").fillna("UNKNOWN")
        if inner_split is not None and int(run_params.get("iterations", 0)) > 50:
            in_tr, in_va = inner_split
            probe = CatBoostClassifier(
                **run_params,
                cat_features=categorical,
                random_seed=seed,
                loss_function="Logloss",
                eval_metric="PRAUC",
                early_stopping_rounds=50,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
            probe.fit(
                train.loc[in_tr],
                y_train[in_tr],
                eval_set=(train.loc[in_va], y_train[in_va]),
            )
            best_iter = probe.get_best_iteration()
            if best_iter is not None and best_iter >= 0:
                run_params["iterations"] = _bounded_best_iterations(
                    int(run_params["iterations"]), int(best_iter) + 1
                )
        est = CatBoostClassifier(
            **run_params,
            cat_features=categorical,
            random_seed=seed,
            loss_function="Logloss",
            eval_metric="PRAUC",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        est.fit(train, y_train)
        return CatBoostBundleEstimator(est, categorical)

    if inner_split is not None and int(run_params.get("n_estimators", 0)) > 50:
        in_tr, in_va = inner_split
        pre: Any = (
            CategoricalFrameTransformer(numeric, categorical, min_frequency=100, max_categories=300)
            if model_name == "xgboost"
            else _onehot_preprocessor(numeric, categorical)
        )
        x_in_tr = pre.fit_transform(x_train.loc[in_tr])
        x_in_va = pre.transform(x_train.loc[in_va])
        if model_name == "xgboost":
            probe = XGBClassifier(
                **run_params,
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
                enable_categorical=True,
                max_cat_to_onehot=16,
                eval_metric="logloss",
                early_stopping_rounds=50,
            )
            probe.fit(x_in_tr, y_train[in_tr], eval_set=[(x_in_va, y_train[in_va])], verbose=False)
            best_iter = getattr(probe, "best_iteration", None)
            if best_iter is not None and best_iter >= 0:
                run_params["n_estimators"] = _bounded_best_iterations(
                    int(run_params["n_estimators"]), int(best_iter) + 1
                )
        else:
            import lightgbm as lgb

            probe = LGBMClassifier(**run_params, random_state=seed, n_jobs=-1, verbosity=-1)
            probe.fit(
                x_in_tr,
                y_train[in_tr],
                eval_X=x_in_va,
                eval_y=y_train[in_va],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            best_iter = getattr(probe, "best_iteration_", None)
            if best_iter is not None and int(best_iter) > 0:
                run_params["n_estimators"] = _bounded_best_iterations(
                    int(run_params["n_estimators"]), int(best_iter)
                )

    if model_name == "xgboost":
        estimator: Any = XGBClassifier(
            **run_params,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,
            max_cat_to_onehot=16,
            eval_metric="logloss",
        )
        final_pre: Any = CategoricalFrameTransformer(
            numeric, categorical, min_frequency=100, max_categories=300
        )
    else:
        estimator = LGBMClassifier(**run_params, random_state=seed, n_jobs=-1, verbosity=-1)
        final_pre = _onehot_preprocessor(numeric, categorical)
    return Pipeline([("preprocessor", final_pre), ("classifier", estimator)]).fit(x_train, y_train)


def save_eval_plots(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    title_prefix: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    prec, rec, _ = precision_recall_curve(y_true, scores)
    axes[0].plot(rec, prec, color="#0f5b78")
    axes[0].axhline(float(np.mean(y_true)), linestyle="--", color="gray", label="Prevalence")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"{title_prefix} Precision-Recall")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    fpr, tpr, _ = roc_curve(y_true, scores)
    axes[1].plot(fpr, tpr, color="#2e8b57")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title(f"{title_prefix} ROC")
    axes[1].grid(alpha=0.2)

    bins = pd.cut(scores, bins=np.linspace(0, 1, 11), include_lowest=True)
    rel = (
        pd.DataFrame({"score": scores, "target": y_true, "bin": bins})
        .groupby("bin", observed=True)
        .agg(mean_score=("score", "mean"), observed_rate=("target", "mean"))
    )
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[2].plot(rel["mean_score"], rel["observed_rate"], marker="o", color="#994433")
    axes[2].set_xlabel("Mean predicted prob")
    axes[2].set_ylabel("Observed delay rate")
    axes[2].set_title(f"{title_prefix} Reliability")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
