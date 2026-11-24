"""Chronological model selection, holdout calibration and locked-test evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from flight_delay.features.build import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_frame,
)
from flight_delay.models.predictor import ModelBundle, save_bundle


@dataclass(frozen=True)
class TrainingOutputs:
    comparison: pd.DataFrame
    test_metrics: dict[str, Any]
    selected_model: str
    artifact_path: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def chronological_split(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    date = pl.col("flight_date")
    partitions = {
        "train": frame.filter(date < pl.date(2024, 1, 1)),
        "validation": frame.filter((date >= pl.date(2024, 1, 1)) & (date < pl.date(2024, 10, 1))),
        "calibration": frame.filter((date >= pl.date(2024, 10, 1)) & (date < pl.date(2025, 1, 1))),
        "test": frame.filter(date >= pl.date(2025, 1, 1)),
    }
    for name, part in partitions.items():
        if part.height == 0:
            raise ValueError(f"Chronological partition is empty: {name}")
    max_train = partitions["train"].get_column("flight_date").max()
    min_validation = partitions["validation"].get_column("flight_date").min()
    min_calibration = partitions["calibration"].get_column("flight_date").min()
    min_test = partitions["test"].get_column("flight_date").min()
    if not (max_train < min_validation < min_calibration < min_test):
        raise ValueError("Temporal partitions overlap or are out of order")
    return partitions


def sparse_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )


def dense_preprocessor() -> ColumnTransformer:
    numeric = SimpleImputer(strategy="median", add_indicator=True)
    categorical = Pipeline(
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
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )


def candidate_models(seed: int) -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline(
            [("preprocessor", sparse_preprocessor()), ("classifier", DummyClassifier(strategy="prior"))]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocessor", sparse_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1_500, random_state=seed, solver="liblinear"),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", dense_preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=220,
                        max_leaf_nodes=31,
                        min_samples_leaf=50,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocessor", sparse_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=280,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=5,
                        reg_alpha=0.05,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=4,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = (scores >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - scores, scores]))),
        "threshold": float(threshold),
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.80, 151)
    f1_values = [f1_score(y_true, scores >= value, zero_division=0) for value in candidates]
    return float(candidates[int(np.argmax(f1_values))])


def route_reference(
    frame: pl.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, str]]:
    grouped = (
        frame.group_by("origin", "dest")
        .agg(
            pl.col("distance").median().alias("distance"),
            pl.col("crs_elapsed_time").median().alias("crs_elapsed_time"),
        )
        .to_dicts()
    )
    lookup = {
        f"{row['origin']}|{row['dest']}": {
            "distance": float(row["distance"]),
            "crs_elapsed_time": float(row["crs_elapsed_time"]),
        }
        for row in grouped
    }
    defaults = {
        "distance": float(frame.get_column("distance").median()),
        "crs_elapsed_time": float(frame.get_column("crs_elapsed_time").median()),
    }
    airport_timezones = (
        {
            str(row["origin"]): str(row["origin_timezone"])
            for row in frame.select("origin", "origin_timezone").unique().to_dicts()
            if row["origin_timezone"] is not None
        }
        if "origin_timezone" in frame.columns
        else {}
    )
    return lookup, defaults, airport_timezones


def save_plots(y_true: np.ndarray, scores: np.ndarray, threshold: float, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = (scores >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, labels, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(recall, precision, color="#0f5b78")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Locked Test Precision Recall Curve")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output / "pr_curve.png", dpi=160)
    plt.close()

    bins = pd.cut(scores, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = pd.DataFrame({"score": scores, "target": y_true, "bin": bins}).groupby(
        "bin", observed=True
    ).agg(mean_score=("score", "mean"), observed_rate=("target", "mean"))
    plt.figure(figsize=(6.5, 4.5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(calibration["mean_score"], calibration["observed_rate"], marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed delay rate")
    plt.title("Locked Test Reliability Diagram")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output / "reliability_diagram.png", dpi=160)
    plt.close()


def train_evaluate(
    *,
    data_path: Path,
    artifact_path: Path,
    metrics_dir: Path,
    figures_dir: Path,
    project_root: Path,
    seed: int = 42,
) -> TrainingOutputs:
    frame = pl.read_parquet(data_path).sort("flight_date")
    partitions = chronological_split(frame)
    feature_sets = {name: build_feature_frame(part) for name, part in partitions.items()}
    labels = {
        name: part.get_column("dep_del15").to_numpy().astype(int)
        for name, part in partitions.items()
    }

    rows: list[dict[str, Any]] = []
    trained: dict[str, Pipeline] = {}
    for name, model in candidate_models(seed).items():
        started = time.perf_counter()
        model.fit(feature_sets["train"], labels["train"])
        validation_scores = model.predict_proba(feature_sets["validation"])[:, 1]
        row = {"model": name, **metrics(labels["validation"], validation_scores, 0.5)}
        row["fit_seconds"] = time.perf_counter() - started
        rows.append(row)
        trained[name] = model
        print(json.dumps(row, sort_keys=True))

    comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    selected_name = str(comparison.iloc[0]["model"])
    selected = trained[selected_name]

    calibration_raw = np.asarray(
        selected.predict_proba(feature_sets["calibration"])[:, 1], dtype=float
    )
    clipped = np.clip(calibration_raw, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000.0, solver="lbfgs", random_state=seed)
    calibrator.fit(logits, labels["calibration"])
    calibration_scores = calibrator.predict_proba(logits)[:, 1]
    threshold = choose_threshold(labels["calibration"], calibration_scores)

    test_raw = np.asarray(selected.predict_proba(feature_sets["test"])[:, 1], dtype=float)
    test_logits = np.log(np.clip(test_raw, 1e-6, 1 - 1e-6) / np.clip(1 - test_raw, 1e-6, 1)).reshape(-1, 1)
    test_scores = calibrator.predict_proba(test_logits)[:, 1]
    test_metrics = metrics(labels["test"], test_scores, threshold)
    test_metrics.update(
        {
            "rows": len(labels["test"]),
            "positive_rate": float(labels["test"].mean()),
            "period_start": str(partitions["test"].get_column("flight_date").min()),
            "period_end": str(partitions["test"].get_column("flight_date").max()),
            "selected_model": selected_name,
        }
    )

    lookup, defaults, airport_timezones = route_reference(partitions["train"])
    metadata = {
        "model_version": "flight-only-v1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": selected_name,
        "feature_columns": MODEL_FEATURES,
        "feature_contract_version": "t24-v1",
        "target": "dep_del15",
        "prediction_horizon_hours": 24,
        "data_sha256": file_sha256(data_path),
        "code_commit": git_sha(project_root),
        "weather_policy": "not_used_offline_no_asof_coverage_for_full_2023_2025_window",
        "partition_rows": {name: part.height for name, part in partitions.items()},
        "test_metrics": test_metrics,
    }
    bundle = ModelBundle(
        model=selected,
        calibrator=calibrator,
        threshold=threshold,
        metadata=metadata,
        route_lookup=lookup,
        defaults=defaults,
        airport_timezones=airport_timezones,
    )
    save_bundle(bundle, artifact_path)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    (metrics_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (metrics_dir / "artifact_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_plots(labels["test"], test_scores, threshold, figures_dir)
    return TrainingOutputs(comparison, test_metrics, selected_name, artifact_path)

