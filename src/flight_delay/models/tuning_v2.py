"""Optuna model tuning and baseline comparison on expanding temporal folds without test access."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import polars as pl
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from flight_delay.data.temporal import TemporalFold, fold_from_mapping
from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame, v2_feature_columns
from flight_delay.models.predictor import CategoricalFrameTransformer

ModelName = Literal["xgboost", "lightgbm", "catboost"]
AllModelName = Literal[
    "dummy_prior",
    "logistic_regression",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
]
ClassWeightMode = Literal["none", "balanced"]
EvalStrategy = Literal["probe_only", "refit_full_train"]


@dataclass(frozen=True)
class FoldResult:
    fold: str
    rows_train: int
    rows_validation: int
    positive_rate_validation: float
    pr_auc: float
    roc_auc: float
    brier: float
    log_loss: float
    fit_seconds: float
    rows_fit_actual: int = 0
    rows_inner_train: int = 0
    rows_inner_val: int = 0
    best_iteration: int | None = None
    eval_strategy: str = "refit_full_train"


@dataclass
class _EncodedFold:
    name: str
    rows_train: int
    rows_validation: int
    x_in_tr: Any
    y_in_tr: np.ndarray
    x_in_va: Any
    y_in_va: np.ndarray
    x_val_inner: Any
    x_full_tr: Any
    y_full_tr: np.ndarray
    x_val_full: Any
    y_val: np.ndarray
    dates_val: np.ndarray
    pos_weight_in: float
    pos_weight_full: float


def load_splits(path: Path) -> list[TemporalFold]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [fold_from_mapping(mapping) for mapping in config["rolling_folds"]]


def load_development_frame(data_root: Path, end_exclusive: date) -> pl.DataFrame:
    pattern = str(data_root / "year=*" / "month=*" / "flights.parquet")
    return (
        pl.scan_parquet(pattern)
        .filter(pl.col("flight_date") < end_exclusive)
        .collect(engine="streaming")
        .sort("flight_date")
    )


def _xgb_preprocessor(numeric: list[str], categorical: list[str]) -> CategoricalFrameTransformer:
    return CategoricalFrameTransformer(numeric, categorical, min_frequency=100, max_categories=300)


def _onehot_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    high_card = {"carrier_origin", "carrier_route", "origin_hour", "route_season"}
    base_cat = [col for col in categorical if col not in high_card]
    hc_cat = [col for col in categorical if col in high_card]

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    base_cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True),
            ),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", numeric_pipe, numeric),
        ("cat_base", base_cat_pipe, base_cat),
    ]
    if hc_cat:
        hc_cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=500,
                        max_categories=300,
                        sparse_output=True,
                    ),
                ),
            ]
        )
        transformers.append(("cat_hc", hc_cat_pipe, hc_cat))
    return ColumnTransformer(transformers)


def _dense_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = SimpleImputer(strategy="median", add_indicator=True)
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)]
    )


def _inner_chronological_split(
    dates_train: pd.Series, fraction: float = 0.15
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split train rows chronologically into inner-train and inner-val for early stopping."""

    if len(dates_train) < 500:
        return None
    cutoff = dates_train.quantile(1.0 - fraction)
    inner_train = (dates_train < cutoff).to_numpy()
    inner_val = (dates_train >= cutoff).to_numpy()
    if inner_train.sum() < 200 or inner_val.sum() < 100:
        return None
    return inner_train, inner_val


def _pos_weight(y_train: np.ndarray, mode: ClassWeightMode) -> float:
    if mode != "balanced":
        return 1.0
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    return neg / pos if pos > 0 else 1.0


def suggest_params(trial: optuna.Trial, model_name: ModelName) -> dict[str, Any]:
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 8.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1_400, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 100.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 100.0, log=True),
        }
    return {
        "iterations": trial.suggest_int("iterations", 300, 700, step=100),
        "depth": trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
    }


def _fit_predict_baseline(
    model_name: Literal["dummy_prior", "logistic_regression", "hist_gradient_boosting"],
    params: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    seed: int,
    class_weight_mode: ClassWeightMode = "none",
) -> tuple[np.ndarray, float]:
    class_weight = "balanced" if class_weight_mode == "balanced" else None
    if model_name == "dummy_prior":
        pipeline = Pipeline(
            [
                ("preprocessor", _onehot_preprocessor(numeric, categorical)),
                ("classifier", DummyClassifier(strategy="prior")),
            ]
        )
    elif model_name == "logistic_regression":
        pipeline = Pipeline(
            [
                ("preprocessor", _onehot_preprocessor(numeric, categorical)),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(params.get("C", 1.0)),
                        max_iter=int(params.get("max_iter", 200)),
                        tol=1e-3,
                        class_weight=class_weight,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("preprocessor", _dense_preprocessor(numeric, categorical)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=float(params.get("learning_rate", 0.08)),
                        max_iter=int(params.get("max_iter", 220)),
                        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
                        min_samples_leaf=int(params.get("min_samples_leaf", 50)),
                        class_weight=class_weight,
                        random_state=seed,
                    ),
                ),
            ]
        )
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    return pipeline.predict_proba(x_validation)[:, 1], elapsed


def _bounded_best_iterations(orig_iters: int, detected_iters: int) -> int:
    floor = min(50, max(1, orig_iters))
    return max(floor, min(orig_iters, detected_iters))


def _fit_predict_onehot(
    model_name: Literal["xgboost", "lightgbm"],
    params: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    seed: int,
    *,
    dates_train: pd.Series | None = None,
    class_weight_mode: ClassWeightMode = "none",
    early_stopping_rounds: int = 50,
    eval_strategy: EvalStrategy = "refit_full_train",
    return_meta: bool = False,
) -> Any:
    pos_weight = _pos_weight(y_train, class_weight_mode)
    run_params = dict(params)
    inner_split = (
        _inner_chronological_split(dates_train) if dates_train is not None else None
    )
    meta: dict[str, Any] = {
        "rows_fit_actual": len(x_train),
        "rows_inner_train": len(x_train),
        "rows_inner_val": 0,
        "best_iteration": None,
        "eval_strategy": eval_strategy if inner_split is not None else "direct_full_fit",
    }

    started = time.perf_counter()
    if inner_split is not None and int(run_params.get("n_estimators", 0)) > early_stopping_rounds:
        in_tr_mask, in_va_mask = inner_split
        meta["rows_inner_train"] = int(in_tr_mask.sum())
        meta["rows_inner_val"] = int(in_va_mask.sum())
        inner_pre: Any = (
            _xgb_preprocessor(numeric, categorical)
            if model_name == "xgboost"
            else _onehot_preprocessor(numeric, categorical)
        )
        x_in_tr = inner_pre.fit_transform(x_train.loc[in_tr_mask])
        x_in_va = inner_pre.transform(x_train.loc[in_va_mask])
        y_in_tr = y_train[in_tr_mask]
        y_in_va = y_train[in_va_mask]
        in_pos_weight = _pos_weight(y_in_tr, class_weight_mode)
        if model_name == "xgboost":
            probe = XGBClassifier(
                **run_params,
                scale_pos_weight=in_pos_weight,
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
                enable_categorical=True,
                max_cat_to_onehot=16,
                eval_metric="logloss",
                early_stopping_rounds=early_stopping_rounds,
            )
            probe.fit(x_in_tr, y_in_tr, eval_set=[(x_in_va, y_in_va)], verbose=False)
            raw_best = getattr(probe, "best_iteration", None)
            detected = int(raw_best) + 1 if raw_best is not None else int(run_params["n_estimators"])
        else:
            probe = LGBMClassifier(
                **run_params,
                scale_pos_weight=in_pos_weight,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            )
            probe.fit(
                x_in_tr,
                y_in_tr,
                eval_X=x_in_va,
                eval_y=y_in_va,
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            raw_best = getattr(probe, "best_iteration_", None)
            detected = int(raw_best) if raw_best is not None and int(raw_best) > 0 else int(run_params["n_estimators"])

        best_iters = _bounded_best_iterations(int(run_params["n_estimators"]), detected)
        meta["best_iteration"] = best_iters
        if eval_strategy == "probe_only":
            elapsed = time.perf_counter() - started
            meta["rows_fit_actual"] = int(in_tr_mask.sum())
            x_val_tx = inner_pre.transform(x_validation)
            preds = probe.predict_proba(x_val_tx)[:, 1]
            return (preds, elapsed, meta) if return_meta else (preds, elapsed)
        run_params["n_estimators"] = best_iters

    if model_name == "xgboost":
        estimator = XGBClassifier(
            **run_params,
            scale_pos_weight=pos_weight,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,
            max_cat_to_onehot=16,
            eval_metric="logloss",
        )
        pre: Any = _xgb_preprocessor(numeric, categorical)
    else:
        estimator = LGBMClassifier(
            **run_params,
            scale_pos_weight=pos_weight,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        pre = _onehot_preprocessor(numeric, categorical)
    pipeline = Pipeline([("preprocessor", pre), ("classifier", estimator)])
    pipeline.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    preds = pipeline.predict_proba(x_validation)[:, 1]
    return (preds, elapsed, meta) if return_meta else (preds, elapsed)


def _fit_predict_catboost(
    params: dict[str, Any],
    categorical: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    seed: int,
    *,
    dates_train: pd.Series | None = None,
    class_weight_mode: ClassWeightMode = "none",
    early_stopping_rounds: int = 50,
    eval_strategy: EvalStrategy = "refit_full_train",
    return_meta: bool = False,
) -> Any:
    train = x_train.copy()
    validation = x_validation.copy()
    for column in categorical:
        train[column] = train[column].astype("string").fillna("UNKNOWN")
        validation[column] = validation[column].astype("string").fillna("UNKNOWN")

    run_params = dict(params)
    auto_weights = "Balanced" if class_weight_mode == "balanced" else None
    inner_split = (
        _inner_chronological_split(dates_train) if dates_train is not None else None
    )
    meta: dict[str, Any] = {
        "rows_fit_actual": len(train),
        "rows_inner_train": len(train),
        "rows_inner_val": 0,
        "best_iteration": None,
        "eval_strategy": eval_strategy if inner_split is not None else "direct_full_fit",
    }

    started = time.perf_counter()
    if inner_split is not None and int(run_params.get("iterations", 0)) > early_stopping_rounds:
        in_tr_mask, in_va_mask = inner_split
        meta["rows_inner_train"] = int(in_tr_mask.sum())
        meta["rows_inner_val"] = int(in_va_mask.sum())
        probe = CatBoostClassifier(
            **run_params,
            cat_features=categorical,
            auto_class_weights=auto_weights,
            random_seed=seed,
            loss_function="Logloss",
            eval_metric="PRAUC",
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        probe.fit(
            train.loc[in_tr_mask],
            y_train[in_tr_mask],
            eval_set=(train.loc[in_va_mask], y_train[in_va_mask]),
        )
        raw_best = probe.get_best_iteration()
        detected = (
            int(raw_best) + 1
            if raw_best is not None and int(raw_best) >= 0
            else int(run_params["iterations"])
        )
        best_iters = _bounded_best_iterations(int(run_params["iterations"]), detected)
        meta["best_iteration"] = best_iters
        if eval_strategy == "probe_only":
            elapsed = time.perf_counter() - started
            meta["rows_fit_actual"] = int(in_tr_mask.sum())
            preds = probe.predict_proba(validation)[:, 1]
            return (preds, elapsed, meta) if return_meta else (preds, elapsed)
        run_params["iterations"] = best_iters

    estimator = CatBoostClassifier(
        **run_params,
        cat_features=categorical,
        auto_class_weights=auto_weights,
        random_seed=seed,
        loss_function="Logloss",
        eval_metric="PRAUC",
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    estimator.fit(train, y_train)
    elapsed = time.perf_counter() - started
    preds = estimator.predict_proba(validation)[:, 1]
    return (preds, elapsed, meta) if return_meta else (preds, elapsed)


def cleanup_stale_running_trials(storage: str, study_name: str | None = None) -> int:
    """Mark any orphaned Optuna RUNNING trials in SQLite as FAIL before resuming."""

    if not storage.startswith("sqlite:///"):
        return 0
    db_path = Path(storage.removeprefix("sqlite:///"))
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        if study_name is None:
            cur = conn.execute("UPDATE trials SET state = 'FAIL' WHERE state = 'RUNNING'")
        else:
            cur = conn.execute(
                """
                UPDATE trials
                SET state = 'FAIL'
                WHERE state = 'RUNNING'
                  AND study_id IN (SELECT study_id FROM studies WHERE study_name = ?)
                """,
                (study_name,),
            )
        conn.commit()
        return int(cur.rowcount)


def evaluate_params_with_oof(
    *,
    model_name: AllModelName,
    params: dict[str, Any],
    features: pd.DataFrame,
    labels: np.ndarray,
    dates: pd.Series,
    folds: list[TemporalFold],
    feature_group: FeatureGroup,
    seed: int,
    class_weight_mode: ClassWeightMode = "none",
    use_inner_early_stopping: bool = True,
    eval_strategy: EvalStrategy = "refit_full_train",
    extra_numeric: list[str] | None = None,
    extra_categorical: list[str] | None = None,
    trial: optuna.Trial | None = None,
) -> tuple[list[FoldResult], pd.DataFrame]:
    base_num, base_cat = v2_feature_columns(feature_group)
    numeric = list(base_num) + list(extra_numeric or [])
    categorical = list(base_cat) + list(extra_categorical or [])
    results: list[FoldResult] = []
    oof_frames: list[pd.DataFrame] = []
    for fold_idx, fold in enumerate(folds):
        fold.validate()
        train_mask = (dates >= pd.Timestamp(fold.train_start)) & (
            dates < pd.Timestamp(fold.train_end_exclusive)
        )
        validation_mask = (dates >= pd.Timestamp(fold.validation_start)) & (
            dates < pd.Timestamp(fold.validation_end_exclusive)
        )
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(f"empty data for {fold.name}")
        x_train, x_validation = features.loc[train_mask], features.loc[validation_mask]
        y_train, y_validation = labels[train_mask], labels[validation_mask]
        dates_train = dates.loc[train_mask] if use_inner_early_stopping else None
        meta: dict[str, Any] = {
            "rows_fit_actual": int(train_mask.sum()),
            "rows_inner_train": int(train_mask.sum()),
            "rows_inner_val": 0,
            "best_iteration": None,
            "eval_strategy": "baseline_direct",
        }

        if model_name in {"dummy_prior", "logistic_regression", "hist_gradient_boosting"}:
            scores, fit_seconds = _fit_predict_baseline(
                model_name,
                params,
                numeric,
                categorical,
                x_train,
                y_train,
                x_validation,
                seed,
                class_weight_mode=class_weight_mode,
            )
        elif model_name == "catboost":
            scores, fit_seconds, meta = _fit_predict_catboost(
                params,
                categorical,
                x_train,
                y_train,
                x_validation,
                seed,
                dates_train=dates_train,
                class_weight_mode=class_weight_mode,
                eval_strategy=eval_strategy,
                return_meta=True,
            )
        else:
            scores, fit_seconds, meta = _fit_predict_onehot(
                model_name,
                params,
                numeric,
                categorical,
                x_train,
                y_train,
                x_validation,
                seed,
                dates_train=dates_train,
                class_weight_mode=class_weight_mode,
                eval_strategy=eval_strategy,
                return_meta=True,
            )
        clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
        pr_auc = float(average_precision_score(y_validation, scores))
        results.append(
            FoldResult(
                fold=fold.name,
                rows_train=int(train_mask.sum()),
                rows_validation=int(validation_mask.sum()),
                positive_rate_validation=float(y_validation.mean()),
                pr_auc=pr_auc,
                roc_auc=float(roc_auc_score(y_validation, scores)),
                brier=float(brier_score_loss(y_validation, scores)),
                log_loss=float(log_loss(y_validation, np.column_stack([1.0 - clipped, clipped]))),
                fit_seconds=fit_seconds,
                rows_fit_actual=int(meta["rows_fit_actual"]),
                rows_inner_train=int(meta["rows_inner_train"]),
                rows_inner_val=int(meta["rows_inner_val"]),
                best_iteration=meta["best_iteration"],
                eval_strategy=str(meta["eval_strategy"]),
            )
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "fold": fold.name,
                    "flight_date": pd.to_datetime(dates.loc[validation_mask]).to_numpy(),
                    "y_true": y_validation.astype(int),
                    "y_prob": scores.astype(float),
                }
            )
        )
        if trial is not None:
            trial.report(float(np.mean([r.pr_auc for r in results])), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    return results, oof_df


def evaluate_params(
    *,
    model_name: AllModelName,
    params: dict[str, Any],
    features: pd.DataFrame,
    labels: np.ndarray,
    dates: pd.Series,
    folds: list[TemporalFold],
    feature_group: FeatureGroup,
    seed: int,
    class_weight_mode: ClassWeightMode = "none",
    use_inner_early_stopping: bool = True,
    eval_strategy: EvalStrategy = "refit_full_train",
    extra_numeric: list[str] | None = None,
    extra_categorical: list[str] | None = None,
) -> list[FoldResult]:
    results, _ = evaluate_params_with_oof(
        model_name=model_name,
        params=params,
        features=features,
        labels=labels,
        dates=dates,
        folds=folds,
        feature_group=feature_group,
        seed=seed,
        class_weight_mode=class_weight_mode,
        use_inner_early_stopping=use_inner_early_stopping,
        eval_strategy=eval_strategy,
        extra_numeric=extra_numeric,
        extra_categorical=extra_categorical,
    )
    return results


def summarize_fold_results(
    *,
    model_name: str,
    feature_group: str,
    class_weight_mode: str,
    results: list[FoldResult],
    source: str,
) -> dict[str, Any]:
    pr_scores = [r.pr_auc for r in results]
    return {
        "model": model_name,
        "feature_group": feature_group,
        "class_weight": class_weight_mode,
        "source": source,
        "mean_pr_auc": float(np.mean(pr_scores)),
        "std_pr_auc": float(np.std(pr_scores)),
        "worst_fold_pr_auc": float(np.min(pr_scores)),
        "mean_roc_auc": float(np.mean([r.roc_auc for r in results])),
        "mean_brier": float(np.mean([r.brier for r in results])),
        "mean_log_loss": float(np.mean([r.log_loss for r in results])),
        "mean_prevalence_baseline": float(
            np.mean([r.positive_rate_validation for r in results])
        ),
        "mean_fit_seconds": float(np.mean([r.fit_seconds for r in results])),
    }


def _prepare_encoded_folds(
    *,
    model_name: Literal["xgboost", "lightgbm"],
    features: pd.DataFrame,
    labels: np.ndarray,
    dates: pd.Series,
    folds: list[TemporalFold],
    feature_group: FeatureGroup,
    class_weight_mode: ClassWeightMode,
    include_full_refit: bool = True,
) -> list[_EncodedFold]:
    numeric, categorical = v2_feature_columns(feature_group)
    encoded: list[_EncodedFold] = []
    for fold in folds:
        fold.validate()
        train_mask = (dates >= pd.Timestamp(fold.train_start)) & (
            dates < pd.Timestamp(fold.train_end_exclusive)
        )
        validation_mask = (dates >= pd.Timestamp(fold.validation_start)) & (
            dates < pd.Timestamp(fold.validation_end_exclusive)
        )
        x_train, x_val = features.loc[train_mask], features.loc[validation_mask]
        y_train, y_val = labels[train_mask], labels[validation_mask]
        split = _inner_chronological_split(dates.loc[train_mask])
        if split is None:
            in_tr_mask = np.ones(len(x_train), dtype=bool)
            in_va_mask = np.ones(len(x_train), dtype=bool)
        else:
            in_tr_mask, in_va_mask = split
        pre_inner: Any = (
            _xgb_preprocessor(numeric, categorical)
            if model_name == "xgboost"
            else _onehot_preprocessor(numeric, categorical)
        )
        x_in_tr = pre_inner.fit_transform(x_train.loc[in_tr_mask])
        x_in_va = pre_inner.transform(x_train.loc[in_va_mask])
        x_val_inner = pre_inner.transform(x_val)
        y_in_tr = y_train[in_tr_mask]
        y_in_va = y_train[in_va_mask]

        if include_full_refit:
            pre_full: Any = (
                _xgb_preprocessor(numeric, categorical)
                if model_name == "xgboost"
                else _onehot_preprocessor(numeric, categorical)
            )
            x_full_tr = pre_full.fit_transform(x_train)
            x_val_full = pre_full.transform(x_val)
        else:
            x_full_tr = x_in_tr
            x_val_full = x_val_inner

        encoded.append(
            _EncodedFold(
                name=fold.name,
                rows_train=int(train_mask.sum()),
                rows_validation=int(validation_mask.sum()),
                x_in_tr=x_in_tr,
                y_in_tr=y_in_tr,
                x_in_va=x_in_va,
                y_in_va=y_in_va,
                x_val_inner=x_val_inner,
                x_full_tr=x_full_tr,
                y_full_tr=y_train,
                x_val_full=x_val_full,
                y_val=y_val,
                dates_val=pd.to_datetime(dates.loc[validation_mask]).to_numpy(),
                pos_weight_in=_pos_weight(y_in_tr, class_weight_mode),
                pos_weight_full=_pos_weight(y_train, class_weight_mode),
            )
        )
    return encoded


def _evaluate_encoded_folds(
    *,
    model_name: Literal["xgboost", "lightgbm"],
    params: dict[str, Any],
    encoded_folds: list[_EncodedFold],
    seed: int,
    early_stopping_rounds: int = 40,
    eval_strategy: EvalStrategy = "refit_full_train",
    trial: optuna.Trial | None = None,
) -> tuple[list[FoldResult], pd.DataFrame]:
    results: list[FoldResult] = []
    oof_frames: list[pd.DataFrame] = []
    for fold_idx, ef in enumerate(encoded_folds):
        started = time.perf_counter()
        orig_n = int(params.get("n_estimators", 300))
        if model_name == "xgboost":
            probe = XGBClassifier(
                **params,
                scale_pos_weight=ef.pos_weight_in,
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
                enable_categorical=True,
                max_cat_to_onehot=16,
                eval_metric="logloss",
                early_stopping_rounds=early_stopping_rounds,
            )
            probe.fit(ef.x_in_tr, ef.y_in_tr, eval_set=[(ef.x_in_va, ef.y_in_va)], verbose=False)
            raw_best = getattr(probe, "best_iteration", None)
            detected = int(raw_best) + 1 if raw_best is not None else orig_n
        else:
            probe = LGBMClassifier(
                **params,
                scale_pos_weight=ef.pos_weight_in,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            )
            probe.fit(
                ef.x_in_tr,
                ef.y_in_tr,
                eval_X=ef.x_in_va,
                eval_y=ef.y_in_va,
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            raw_best = getattr(probe, "best_iteration_", None)
            detected = int(raw_best) if raw_best is not None and int(raw_best) > 0 else orig_n

        best_iters = _bounded_best_iterations(orig_n, detected)
        if eval_strategy == "probe_only":
            fit_seconds = time.perf_counter() - started
            scores = probe.predict_proba(ef.x_val_inner)[:, 1]
            rows_fit_actual = len(ef.y_in_tr)
        else:
            refit_params = {**params, "n_estimators": best_iters}
            if model_name == "xgboost":
                est = XGBClassifier(
                    **refit_params,
                    scale_pos_weight=ef.pos_weight_full,
                    random_state=seed,
                    n_jobs=-1,
                    tree_method="hist",
                    enable_categorical=True,
                    max_cat_to_onehot=16,
                    eval_metric="logloss",
                )
            else:
                est = LGBMClassifier(
                    **refit_params,
                    scale_pos_weight=ef.pos_weight_full,
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=-1,
                )
            est.fit(ef.x_full_tr, ef.y_full_tr)
            fit_seconds = time.perf_counter() - started
            scores = est.predict_proba(ef.x_val_full)[:, 1]
            rows_fit_actual = int(ef.rows_train)

        clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
        pr_auc = float(average_precision_score(ef.y_val, scores))
        results.append(
            FoldResult(
                fold=ef.name,
                rows_train=ef.rows_train,
                rows_validation=ef.rows_validation,
                positive_rate_validation=float(ef.y_val.mean()),
                pr_auc=pr_auc,
                roc_auc=float(roc_auc_score(ef.y_val, scores)),
                brier=float(brier_score_loss(ef.y_val, scores)),
                log_loss=float(log_loss(ef.y_val, np.column_stack([1.0 - clipped, clipped]))),
                fit_seconds=fit_seconds,
                rows_fit_actual=rows_fit_actual,
                rows_inner_train=len(ef.y_in_tr),
                rows_inner_val=len(ef.y_in_va),
                best_iteration=best_iters,
                eval_strategy=eval_strategy,
            )
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "fold": ef.name,
                    "flight_date": ef.dates_val,
                    "y_true": ef.y_val.astype(int),
                    "y_prob": scores.astype(float),
                }
            )
        )
        if trial is not None:
            trial.report(float(np.mean([r.pr_auc for r in results])), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    return results, oof_df


def make_objective(
    *,
    model_name: ModelName,
    features: pd.DataFrame,
    labels: np.ndarray,
    dates: pd.Series,
    folds: list[TemporalFold],
    feature_group: FeatureGroup,
    seed: int,
    class_weight_mode: ClassWeightMode = "none",
    encoded_folds: list[_EncodedFold] | None = None,
    eval_strategy: EvalStrategy = "refit_full_train",
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        if encoded_folds is not None and model_name in {"xgboost", "lightgbm"}:
            results, _ = _evaluate_encoded_folds(
                model_name=model_name,
                params=params,
                encoded_folds=encoded_folds,
                seed=seed,
                eval_strategy=eval_strategy,
                trial=trial,
            )
        else:
            results, _ = evaluate_params_with_oof(
                model_name=model_name,
                params=params,
                features=features,
                labels=labels,
                dates=dates,
                folds=folds,
                feature_group=feature_group,
                seed=seed,
                class_weight_mode=class_weight_mode,
                use_inner_early_stopping=True,
                eval_strategy=eval_strategy,
                trial=trial,
            )
        scores = [result.pr_auc for result in results]
        trial.set_user_attr("pr_auc_by_fold", scores)
        trial.set_user_attr("mean_pr_auc", float(np.mean(scores)))
        trial.set_user_attr("std_pr_auc", float(np.std(scores)))
        trial.set_user_attr("worst_fold_pr_auc", float(np.min(scores)))
        trial.set_user_attr("mean_roc_auc", float(np.mean([r.roc_auc for r in results])))
        trial.set_user_attr("mean_brier", float(np.mean([r.brier for r in results])))
        trial.set_user_attr("mean_log_loss", float(np.mean([r.log_loss for r in results])))
        trial.set_user_attr("mean_fit_seconds", float(np.mean([r.fit_seconds for r in results])))
        trial.set_user_attr("eval_strategy", eval_strategy)
        trial.set_user_attr("fold_results", [result.__dict__ for result in results])
        return float(np.mean(scores))

    return objective


def run_study(
    *,
    model_name: ModelName,
    frame: pl.DataFrame,
    feature_group: FeatureGroup,
    folds: list[TemporalFold],
    storage: str,
    trials: int,
    timeout: int,
    seed: int,
    class_weight_mode: ClassWeightMode = "none",
    study_name_prefix: str = "v2_1",
    eval_strategy: EvalStrategy = "refit_full_train",
    use_pruner: bool = True,
) -> optuna.Study:
    features = build_v2_feature_frame(frame, feature_group)
    labels = frame.get_column("dep_del15").to_numpy().astype(int)
    dates = pd.to_datetime(frame.get_column("flight_date").to_pandas())
    study_name = f"{study_name_prefix}_{model_name}_{feature_group}_{class_weight_mode}_{eval_strategy}"
    cleanup_stale_running_trials(storage, study_name)
    encoded_folds = (
        _prepare_encoded_folds(
            model_name=model_name,
            features=features,
            labels=labels,
            dates=dates,
            folds=folds,
            feature_group=feature_group,
            class_weight_mode=class_weight_mode,
            include_full_refit=(eval_strategy == "refit_full_train"),
        )
        if model_name in {"xgboost", "lightgbm"}
        else None
    )
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        if use_pruner
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(
        make_objective(
            model_name=model_name,
            features=features,
            labels=labels,
            dates=dates,
            folds=folds,
            feature_group=feature_group,
            seed=seed,
            class_weight_mode=class_weight_mode,
            encoded_folds=encoded_folds,
            eval_strategy=eval_strategy,
        ),
        n_trials=trials,
        timeout=timeout,
    )
    return study


def export_study(study: optuna.Study, output_root: Path, model_name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_root / f"{model_name}_trials.csv", index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        (output_root / f"{model_name}_best_params.yaml").write_text(
            yaml.safe_dump(study.best_params, sort_keys=True), encoding="utf-8"
        )
        (output_root / f"{model_name}_best_trial.json").write_text(
            json.dumps(
                {
                    "number": study.best_trial.number,
                    "value": study.best_value,
                    "params": study.best_params,
                    "user_attrs": study.best_trial.user_attrs,
                    "complete_trials": len(completed),
                    "pruned_trials": sum(
                        t.state == optuna.trial.TrialState.PRUNED for t in study.trials
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if len(completed) >= 2:
        try:
            importances = optuna.importance.get_param_importances(study)
            if importances:
                names = list(importances.keys())[::-1]
                vals = [importances[k] for k in names]
                plt.figure(figsize=(7.5, max(3.5, 0.4 * len(names))))
                plt.barh(names, vals, color="#2e6f95")
                plt.xlabel("Hyperparameter importance")
                plt.title(f"Optuna hyperparameter importance ({model_name})")
                plt.tight_layout()
                plt.savefig(output_root / f"{model_name}_optuna_importance.png", dpi=150)
                plt.close()
        except (ValueError, RuntimeError):
            plt.close()
