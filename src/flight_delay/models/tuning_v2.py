"""Optuna model tuning on expanding temporal folds without test access."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
import polars as pl
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from flight_delay.data.temporal import TemporalFold, fold_from_mapping
from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame, v2_feature_columns

ModelName = Literal["xgboost", "lightgbm", "catboost"]


@dataclass(frozen=True)
class FoldResult:
    fold: str
    rows_train: int
    rows_validation: int
    positive_rate_validation: float
    pr_auc: float
    roc_auc: float
    brier: float
    fit_seconds: float


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


def _onehot_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)]
    )


def suggest_params(trial: optuna.Trial, model_name: ModelName) -> dict[str, Any]:
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1_200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
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
        "iterations": trial.suggest_int("iterations", 300, 1_200, step=100),
        "depth": trial.suggest_int("depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
    }


def _fit_predict_onehot(
    model_name: Literal["xgboost", "lightgbm"],
    params: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, float]:
    if model_name == "xgboost":
        estimator = XGBClassifier(
            **params,
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
            eval_metric="logloss",
        )
    else:
        estimator = LGBMClassifier(
            **params,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
        )
    pipeline = Pipeline(
        [("preprocessor", _onehot_preprocessor(numeric, categorical)), ("classifier", estimator)]
    )
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    return pipeline.predict_proba(x_validation)[:, 1], elapsed


def _fit_predict_catboost(
    params: dict[str, Any],
    categorical: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, float]:
    train = x_train.copy()
    validation = x_validation.copy()
    for column in categorical:
        train[column] = train[column].astype("string").fillna("UNKNOWN")
        validation[column] = validation[column].astype("string").fillna("UNKNOWN")
    estimator = CatBoostClassifier(
        **params,
        cat_features=categorical,
        random_seed=seed,
        loss_function="Logloss",
        eval_metric="PRAUC",
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )
    started = time.perf_counter()
    estimator.fit(train, y_train)
    elapsed = time.perf_counter() - started
    return estimator.predict_proba(validation)[:, 1], elapsed


def evaluate_params(
    *,
    model_name: ModelName,
    params: dict[str, Any],
    features: pd.DataFrame,
    labels: np.ndarray,
    dates: pd.Series,
    folds: list[TemporalFold],
    feature_group: FeatureGroup,
    seed: int,
) -> list[FoldResult]:
    numeric, categorical = v2_feature_columns(feature_group)
    results: list[FoldResult] = []
    for fold in folds:
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
        if model_name == "catboost":
            scores, fit_seconds = _fit_predict_catboost(
                params,
                categorical,
                x_train,
                y_train,
                x_validation,
                seed,
            )
        else:
            scores, fit_seconds = _fit_predict_onehot(
                model_name,
                params,
                numeric,
                categorical,
                x_train,
                y_train,
                x_validation,
                seed,
            )
        results.append(
            FoldResult(
                fold=fold.name,
                rows_train=int(train_mask.sum()),
                rows_validation=int(validation_mask.sum()),
                positive_rate_validation=float(y_validation.mean()),
                pr_auc=float(average_precision_score(y_validation, scores)),
                roc_auc=float(roc_auc_score(y_validation, scores)),
                brier=float(brier_score_loss(y_validation, scores)),
                fit_seconds=fit_seconds,
            )
        )
    return results


def make_objective(
    *,
    model_name: ModelName,
    features: pd.DataFrame,
    labels: np.ndarray,
    dates: pd.Series,
    folds: list[TemporalFold],
    feature_group: FeatureGroup,
    seed: int,
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        results = evaluate_params(
            model_name=model_name,
            params=params,
            features=features,
            labels=labels,
            dates=dates,
            folds=folds,
            feature_group=feature_group,
            seed=seed,
        )
        scores = [result.pr_auc for result in results]
        trial.set_user_attr("pr_auc_by_fold", scores)
        trial.set_user_attr("mean_pr_auc", float(np.mean(scores)))
        trial.set_user_attr("std_pr_auc", float(np.std(scores)))
        trial.set_user_attr("worst_fold_pr_auc", float(np.min(scores)))
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
) -> optuna.Study:
    features = build_v2_feature_frame(frame, feature_group)
    labels = frame.get_column("dep_del15").to_numpy().astype(int)
    dates = pd.to_datetime(frame.get_column("flight_date").to_pandas())
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=f"v2_{model_name}_{feature_group}",
        storage=storage,
        direction="maximize",
        sampler=sampler,
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
        ),
        n_trials=trials,
        timeout=timeout,
    )
    return study


def export_study(study: optuna.Study, output_root: Path, model_name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_root / f"{model_name}_trials.csv", index=False)
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
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

