"""Separated V2 calibration, threshold freeze, and one-time untouched holdout evaluation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame, v2_feature_columns
from flight_delay.models.calibrate_v2 import (
    CalibrationMethod,
    calendar_block_bootstrap_ci,
    evaluate_binary_metrics,
    fit_calibrator,
    threshold_sweep,
)
from flight_delay.models.predictor import (
    CatBoostBundleEstimator,
    CategoricalFrameTransformer,
    ModelBundle,
    save_bundle,
)
from flight_delay.models.train import file_sha256, git_sha, route_reference
from flight_delay.models.tuning_v2 import (
    ModelName,
    _inner_chronological_split,
    _onehot_preprocessor,
)


def _airport_lookup_from_frame(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
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


def _fit_frozen_model(
    *,
    model_name: ModelName,
    params: dict[str, Any],
    feature_group: FeatureGroup,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    dates_train: pd.Series,
    seed: int,
) -> Any:
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
                thread_count=4,
            )
            probe.fit(
                train.loc[in_tr],
                y_train[in_tr],
                eval_set=(train.loc[in_va], y_train[in_va]),
            )
            best_iter = probe.get_best_iteration()
            if best_iter is not None and best_iter >= 0:
                run_params["iterations"] = max(50, int(best_iter) + 1)
        est = CatBoostClassifier(
            **run_params,
            cat_features=categorical,
            random_seed=seed,
            loss_function="Logloss",
            eval_metric="PRAUC",
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
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
                run_params["n_estimators"] = max(50, int(best_iter) + 1)
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
                run_params["n_estimators"] = max(50, int(best_iter))

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


def _save_eval_plots(
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--holdout-data-root", type=Path, default=Path("data/processed/v2_holdout"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_v2.yaml"))
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--params-yaml", type=Path, required=True)
    parser.add_argument("--feature-group", type=str, default="F3")
    parser.add_argument("--bundle-output", type=Path, default=Path("models/flight_delay_v2_bundle.joblib"))
    parser.add_argument("--run-final-holdout", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits_cfg = yaml.safe_load(args.splits.read_text(encoding="utf-8"))
    params = yaml.safe_load(args.params_yaml.read_text(encoding="utf-8"))
    model_name: ModelName = args.model_name  # type: ignore[assignment]
    feature_group: FeatureGroup = args.feature_group  # type: ignore[assignment]

    pattern = str(args.data_root / "year=*" / "month=*" / "flights.parquet")
    dev_all = (
        pl.scan_parquet(pattern)
        .filter(pl.col("flight_date") < date(2025, 9, 1))
        .collect(engine="streaming")
        .sort("flight_date")
    )

    def _window(cfg_key: str) -> pl.DataFrame:
        win = splits_cfg[cfg_key]
        return dev_all.filter(
            (pl.col("flight_date") >= date.fromisoformat(str(win["start"])))
            & (pl.col("flight_date") < date.fromisoformat(str(win["end_exclusive"])))
        )

    # 1. Historical reference check (Jan 2025): fit strictly < 2025-01-01
    hist_train = dev_all.filter(pl.col("flight_date") < date(2025, 1, 1))
    hist_eval = _window("historical_reference")
    hist_model = _fit_frozen_model(
        model_name=model_name,
        params=params,
        feature_group=feature_group,
        x_train=build_v2_feature_frame(hist_train, feature_group),
        y_train=hist_train.get_column("dep_del15").to_numpy().astype(int),
        dates_train=pd.to_datetime(hist_train.get_column("flight_date").to_pandas()),
        seed=args.seed,
    )
    hist_raw = hist_model.predict_proba(build_v2_feature_frame(hist_eval, feature_group))[:, 1]
    hist_metrics = evaluate_binary_metrics(
        hist_eval.get_column("dep_del15").to_numpy().astype(int), hist_raw, threshold=0.5
    )

    # 2. Frozen V2 base model trained up to calibration start (< 2025-02-01)
    cal_start = date.fromisoformat(str(splits_cfg["calibration"]["start"]))
    base_train = dev_all.filter(pl.col("flight_date") < cal_start)
    started = time.perf_counter()
    frozen_model = _fit_frozen_model(
        model_name=model_name,
        params=params,
        feature_group=feature_group,
        x_train=build_v2_feature_frame(base_train, feature_group),
        y_train=base_train.get_column("dep_del15").to_numpy().astype(int),
        dates_train=pd.to_datetime(base_train.get_column("flight_date").to_pandas()),
        seed=args.seed,
    )
    fit_seconds = time.perf_counter() - started

    # 3. Calibration selection on dedicated calibration window (split Feb fit -> Mar eval)
    cal_df = _window("calibration")
    cal_raw = frozen_model.predict_proba(build_v2_feature_frame(cal_df, feature_group))[:, 1]
    cal_y = cal_df.get_column("dep_del15").to_numpy().astype(int)
    cal_dates = pd.to_datetime(cal_df.get_column("flight_date").to_pandas())
    cal_mid = cal_dates.quantile(0.5)
    cal_fit_mask = (cal_dates < cal_mid).to_numpy()
    cal_eval_mask = (cal_dates >= cal_mid).to_numpy()

    cal_rows: list[dict[str, Any]] = []
    for method in ["none", "platt", "isotonic"]:
        m_name: CalibrationMethod = method  # type: ignore[assignment]
        fitted = fit_calibrator(m_name, cal_raw[cal_fit_mask], cal_y[cal_fit_mask], seed=args.seed)
        cal_probs = fitted.predict(cal_raw[cal_eval_mask])
        m_dict = evaluate_binary_metrics(cal_y[cal_eval_mask], cal_probs, threshold=0.5)
        cal_rows.append({"calibration_method": m_name, **m_dict})

    cal_table = pd.DataFrame(cal_rows).sort_values(["brier", "log_loss"])
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    cal_table.to_csv("reports/metrics/v2_calibration.csv", index=False)
    best_cal_method: CalibrationMethod = str(cal_table.iloc[0]["calibration_method"])  # type: ignore[assignment]
    final_calibrator = fit_calibrator(best_cal_method, cal_raw, cal_y, seed=args.seed)

    # 4. Threshold selection ONLY on threshold_selection window
    thr_df = _window("threshold_selection")
    thr_raw = frozen_model.predict_proba(build_v2_feature_frame(thr_df, feature_group))[:, 1]
    thr_probs = final_calibrator.predict(thr_raw)
    thr_y = thr_df.get_column("dep_del15").to_numpy().astype(int)
    sweep = threshold_sweep(thr_y, thr_probs)
    sweep.to_csv("reports/metrics/v2_threshold_grid.csv", index=False)
    best_thr_row = sweep.sort_values("f1", ascending=False).iloc[0]
    selected_threshold = float(best_thr_row["threshold"])

    # 5. Pre-holdout validation check
    pre_df = _window("pre_holdout_validation")
    pre_raw = frozen_model.predict_proba(build_v2_feature_frame(pre_df, feature_group))[:, 1]
    pre_probs = final_calibrator.predict(pre_raw)
    pre_y = pre_df.get_column("dep_del15").to_numpy().astype(int)
    pre_metrics = evaluate_binary_metrics(pre_y, pre_probs, selected_threshold)

    summary_rows = [
        {"stage": "historical_reference_2025_01", **hist_metrics},
        {"stage": "calibration_eval_slice", **cal_table.iloc[0].drop("calibration_method").to_dict()},
        {
            "stage": "threshold_selection_2025_04_05",
            **evaluate_binary_metrics(thr_y, thr_probs, selected_threshold),
        },
        {"stage": "pre_holdout_validation_2025_06_08", **pre_metrics},
    ]

    lookup, defaults, airport_timezones = route_reference(base_train)
    airport_lookup = _airport_lookup_from_frame(base_train)
    for code, tz in airport_timezones.items():
        airport_lookup.setdefault(code, {})["timezone"] = tz

    bundle_meta: dict[str, Any] = {
        "model_version": f"flight-only-v2.0.0-{model_name}",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": model_name,
        "feature_group": feature_group,
        "feature_columns": build_v2_feature_frame(base_train.head(1), feature_group).columns.tolist(),
        "feature_contract_version": f"t24-v2-{feature_group}",
        "require_origin_timezone": True,
        "target": "dep_del15",
        "prediction_horizon_hours": 24,
        "calibration_method": best_cal_method,
        "threshold": selected_threshold,
        "threshold_objective": "max_f1_on_threshold_selection_window",
        "training_range": f"{base_train.get_column('flight_date').min()}..{base_train.get_column('flight_date').max()}",
        "calibration_range": f"{cal_df.get_column('flight_date').min()}..{cal_df.get_column('flight_date').max()}",
        "threshold_range": f"{thr_df.get_column('flight_date').min()}..{thr_df.get_column('flight_date').max()}",
        "pre_holdout_range": f"{pre_df.get_column('flight_date').min()}..{pre_df.get_column('flight_date').max()}",
        "code_commit": git_sha(Path(".")),
        "data_manifest_sha256": (
            file_sha256(Path("data/manifests/bts_v2.json"))
            if Path("data/manifests/bts_v2.json").exists()
            else "unavailable"
        ),
        "fit_seconds": fit_seconds,
        "weather_status": "not_used_flight_only_v2",
        "data_freshness": "bts_v2_2023_01_to_2025_08",
    }

    holdout_metrics: dict[str, Any] | None = None
    if args.run_final_holdout:
        holdout_pattern = str(args.holdout_data_root / "year=*" / "month=*" / "flights.parquet")
        holdout_df = pl.scan_parquet(holdout_pattern).collect(engine="streaming").sort("flight_date")
        h_feats = build_v2_feature_frame(holdout_df, feature_group)
        h_raw = frozen_model.predict_proba(h_feats)[:, 1]
        h_probs = final_calibrator.predict(h_raw)
        h_y = holdout_df.get_column("dep_del15").to_numpy().astype(int)
        h_dates = pd.to_datetime(holdout_df.get_column("flight_date").to_pandas())

        holdout_metrics = evaluate_binary_metrics(h_y, h_probs, selected_threshold)
        holdout_metrics["period_start"] = str(holdout_df.get_column("flight_date").min())
        holdout_metrics["period_end"] = str(holdout_df.get_column("flight_date").max())
        holdout_metrics["calibration_method"] = best_cal_method
        holdout_metrics["bootstrap_ci_95"] = calendar_block_bootstrap_ci(
            h_y, h_probs, h_dates, selected_threshold, n_bootstrap=300, seed=args.seed
        )
        summary_rows.append({"stage": "final_holdout_2025_09_once", **evaluate_binary_metrics(h_y, h_probs, selected_threshold)})

        # Subgroup slices
        h_pd = holdout_df.select("reporting_airline", "origin", "dest", "scheduled_departure_local").to_pandas()
        h_pd["y"] = h_y
        h_pd["p"] = h_probs
        h_pd["dep_hour"] = pd.to_datetime(h_pd["scheduled_departure_local"]).dt.hour
        h_pd["route_known"] = [
            f"{o}|{d}" in lookup for o, d in zip(h_pd["origin"], h_pd["dest"], strict=True)
        ]
        slice_rows: list[dict[str, Any]] = []
        for col, values in [
            ("reporting_airline", h_pd["reporting_airline"].value_counts().head(10).index.tolist()),
            ("route_known", [True, False]),
        ]:
            for val in values:
                sub = h_pd.loc[h_pd[col] == val]
                if len(sub) < 50 or sub["y"].nunique() < 2:
                    continue
                slice_rows.append(
                    {
                        "slice_col": col,
                        "slice_val": str(val),
                        **evaluate_binary_metrics(
                            sub["y"].to_numpy(), sub["p"].to_numpy(), selected_threshold
                        ),
                    }
                )
        Path("reports/holdout").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(slice_rows).to_csv("reports/holdout/v2_holdout_slices.csv", index=False)
        Path("reports/holdout/v2_holdout_metrics.json").write_text(
            json.dumps(holdout_metrics, indent=2), encoding="utf-8"
        )
        bundle_meta["holdout_range"] = f"{holdout_metrics['period_start']}..{holdout_metrics['period_end']}"
        bundle_meta["holdout_metrics"] = holdout_metrics
        _save_eval_plots(
            y_true=h_y,
            scores=h_probs,
            title_prefix="Holdout 2025-09",
            output_path=Path("reports/figures/pr_roc_reliability_v2.png"),
        )
        final_md = f"""# V2 Final Untouched Holdout Report (Evaluated Once)

- Model: `{model_name}` (`{feature_group}`)
- Training window: `{bundle_meta['training_range']}` ({base_train.height:,} rows)
- Calibration: `{best_cal_method}` on `{bundle_meta['calibration_range']}` ({cal_df.height:,} rows)
- Threshold: `{selected_threshold:.4f}` (`max_f1` on `{bundle_meta['threshold_range']}`, {thr_df.height:,} rows)
- Final Holdout: `{bundle_meta['holdout_range']}` ({holdout_df.height:,} rows, prevalence={holdout_metrics['positive_rate']:.4f})

## Holdout Metrics
- PR-AUC: `{holdout_metrics['pr_auc']:.4f}` (uplift vs prevalence: `+{holdout_metrics['pr_auc_uplift_abs']:.4f}`, 95% block-bootstrap CI `[{holdout_metrics['bootstrap_ci_95']['pr_auc']['ci95_low']:.4f}, {holdout_metrics['bootstrap_ci_95']['pr_auc']['ci95_high']:.4f}]`)
- ROC-AUC: `{holdout_metrics['roc_auc']:.4f}` (95% CI `[{holdout_metrics['bootstrap_ci_95']['roc_auc']['ci95_low']:.4f}, {holdout_metrics['bootstrap_ci_95']['roc_auc']['ci95_high']:.4f}]`)
- Precision: `{holdout_metrics['precision']:.4f}` | Recall: `{holdout_metrics['recall']:.4f}` | F1: `{holdout_metrics['f1']:.4f}`
- Brier: `{holdout_metrics['brier']:.4f}` | LogLoss: `{holdout_metrics['log_loss']:.4f}` | ECE: `{holdout_metrics['ece']:.4f}`
- Precision@Top5%: `{holdout_metrics['precision_at_top_5pct']:.4f}` | Precision@Top10%: `{holdout_metrics['precision_at_top_10pct']:.4f}`
"""
        Path("reports/holdout/FINAL_ONCE.md").write_text(final_md, encoding="utf-8")
    else:
        _save_eval_plots(
            y_true=pre_y,
            scores=pre_probs,
            title_prefix="Pre-Holdout 2025-06..08",
            output_path=Path("reports/figures/pr_roc_reliability_v2.png"),
        )

    pd.DataFrame(summary_rows).to_csv("reports/metrics/v2_fold_summary.csv", index=False)
    bundle = ModelBundle(
        model=frozen_model,
        calibrator=final_calibrator.estimator,
        threshold=selected_threshold,
        metadata=bundle_meta,
        route_lookup=lookup,
        defaults=defaults,
        airport_timezones=airport_timezones,
        airport_lookup=airport_lookup,
        feature_group=feature_group,
        calibration_method=best_cal_method,
    )
    save_bundle(bundle, args.bundle_output)
    print(json.dumps(summary_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
