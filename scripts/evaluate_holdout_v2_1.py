"""Freeze V2.1 flight-only champion, audit reproducibility on 2025-09, and evaluate 2025-10..11 once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import catboost
import lightgbm
import pandas as pd
import polars as pl
import sklearn
import xgboost
import yaml

from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame
from flight_delay.models.calibrate_v2 import (
    CalibrationMethod,
    calendar_block_bootstrap_ci,
    evaluate_binary_metrics,
    fit_calibrator,
    paired_calendar_block_bootstrap,
    threshold_sweep,
)
from flight_delay.models.holdout_helpers import (
    airport_lookup_from_frame as _airport_lookup_from_frame,
)
from flight_delay.models.holdout_helpers import (
    fit_frozen_model as _fit_frozen_model,
)
from flight_delay.models.holdout_helpers import (
    save_eval_plots as _save_eval_plots,
)
from flight_delay.models.predictor import ModelBundle, load_bundle, save_bundle
from flight_delay.models.train import file_sha256, git_sha, route_reference
from flight_delay.models.tuning_v2 import ModelName


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument(
        "--v2-holdout-root", type=Path, default=Path("data/processed/v2_holdout")
    )
    parser.add_argument(
        "--prospective-root", type=Path, default=Path("data/processed/v2_1_future_holdout")
    )
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_v2_1.yaml"))
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--params-yaml", type=Path, required=True)
    parser.add_argument("--feature-group", type=str, required=True)
    parser.add_argument(
        "--bundle-output", type=Path, default=Path("models/flight_delay_v2_1_bundle.joblib")
    )
    parser.add_argument(
        "--manifest-output", type=Path, default=Path("models/manifest_v2_1.json")
    )
    parser.add_argument(
        "--v2-bundle", type=Path, default=Path("models/flight_delay_v2_bundle.joblib")
    )
    parser.add_argument("--run-prospective-holdout", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits_cfg = yaml.safe_load(args.splits.read_text(encoding="utf-8"))
    model_name: ModelName = args.model_name  # type: ignore[assignment]
    feature_group: FeatureGroup = args.feature_group  # type: ignore[assignment]

    if args.run_prospective_holdout:
        if not args.bundle_output.exists():
            raise FileNotFoundError(
                f"Frozen V2.1 bundle must already exist before --run-prospective-holdout: {args.bundle_output}"
            )
        bundle = load_bundle(args.bundle_output)
        manifest = json.loads(args.manifest_output.read_text(encoding="utf-8"))
        actual_sha = _sha256(args.bundle_output)
        if actual_sha != manifest["bundle_sha256"]:
            raise ValueError("Frozen V2.1 bundle SHA256 mismatch before prospective holdout evaluation")

        pattern = str(args.prospective_root / "year=*" / "month=*" / "flights.parquet")
        prospective_df = pl.scan_parquet(pattern).collect(engine="streaming").sort("flight_date")
        p_feats = build_v2_feature_frame(prospective_df, bundle.feature_group)  # type: ignore[arg-type]
        p_probs = bundle.predict_proba(p_feats)
        p_y = prospective_df.get_column("dep_del15").to_numpy().astype(int)
        p_dates = pd.to_datetime(prospective_df.get_column("flight_date").to_pandas())

        metrics = evaluate_binary_metrics(p_y, p_probs, bundle.threshold)
        p35_thr = float(bundle.metadata.get("threshold_precision_target_35", bundle.threshold))
        metrics["precision_target_35_threshold"] = p35_thr
        metrics["precision_target_35_eval"] = evaluate_binary_metrics(p_y, p_probs, p35_thr)
        metrics["period_start"] = str(prospective_df.get_column("flight_date").min())
        metrics["period_end"] = str(prospective_df.get_column("flight_date").max())
        metrics["calibration_method"] = bundle.calibration_method
        metrics["bundle_sha256"] = actual_sha
        metrics["bootstrap_ci_95"] = calendar_block_bootstrap_ci(
            p_y, p_probs, p_dates, bundle.threshold, n_bootstrap=300, seed=args.seed
        )

        # Post-freeze paired benchmark vs frozen V2 bundle on 2025-10..2025-11
        paired_vs_v2 = None
        if args.v2_bundle.exists():
            v2_bundle = load_bundle(args.v2_bundle)
            v2_feats = build_v2_feature_frame(prospective_df, v2_bundle.feature_group)  # type: ignore[arg-type]
            v2_probs = v2_bundle.predict_proba(v2_feats)
            v2_metrics = evaluate_binary_metrics(p_y, v2_probs, v2_bundle.threshold)
            paired_ci = paired_calendar_block_bootstrap(
                p_y, v2_probs, p_probs, p_dates, n_bootstrap=300, seed=args.seed
            )
            paired_vs_v2 = {
                "v2_model_version": v2_bundle.metadata.get("model_version"),
                "v2_feature_group": v2_bundle.feature_group,
                "v2_threshold": v2_bundle.threshold,
                "v2_metrics": v2_metrics,
                "paired_delta_v2_1_minus_v2": paired_ci,
            }
            metrics["paired_vs_frozen_v2"] = paired_vs_v2

        # Monthly breakdown (2025-10, 2025-11)
        p_pd = prospective_df.select(
            "flight_date", "reporting_airline", "origin", "dest", "scheduled_departure_local"
        ).to_pandas()
        p_pd["year_month"] = pd.to_datetime(p_pd["flight_date"]).dt.strftime("%Y-%m")
        p_pd["y"] = p_y
        p_pd["p"] = p_probs
        p_pd["route_known"] = [
            f"{o}|{d}" in bundle.route_lookup
            for o, d in zip(p_pd["origin"], p_pd["dest"], strict=True)
        ]

        monthly_rows = []
        for ym, grp in p_pd.groupby("year_month", sort=True):
            m_dict = evaluate_binary_metrics(
                grp["y"].to_numpy(), grp["p"].to_numpy(), bundle.threshold
            )
            m_ci = calendar_block_bootstrap_ci(
                grp["y"].to_numpy(),
                grp["p"].to_numpy(),
                pd.to_datetime(grp["flight_date"]),
                bundle.threshold,
                n_bootstrap=300,
                seed=args.seed,
            )
            monthly_rows.append(
                {
                    "year_month": str(ym),
                    "pr_auc_ci95_low": m_ci["pr_auc"]["ci95_low"],
                    "pr_auc_ci95_high": m_ci["pr_auc"]["ci95_high"],
                    **m_dict,
                }
            )

        slice_rows = []
        for col, vals in [
            ("reporting_airline", p_pd["reporting_airline"].value_counts().head(10).index.tolist()),
            ("route_known", [True, False]),
        ]:
            for val in vals:
                sub = p_pd.loc[p_pd[col] == val]
                if len(sub) < 50 or sub["y"].nunique() < 2:
                    continue
                slice_rows.append(
                    {
                        "slice_col": col,
                        "slice_val": str(val),
                        **evaluate_binary_metrics(
                            sub["y"].to_numpy(), sub["p"].to_numpy(), bundle.threshold
                        ),
                    }
                )

        out_dir = Path("reports/holdout_v2_1")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "v2_1_holdout_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        pd.DataFrame(monthly_rows).to_csv(out_dir / "v2_1_monthly_holdout.csv", index=False)
        pd.DataFrame(slice_rows).to_csv(out_dir / "v2_1_holdout_slices.csv", index=False)
        _save_eval_plots(
            y_true=p_y,
            scores=p_probs,
            title_prefix="V2.1 Prospective Holdout 2025-10..11",
            output_path=Path("reports/figures/pr_roc_reliability_v2_1.png"),
        )
        once_md = f"""# V2.1 Prospective Untouched Holdout Report (Evaluated Once)

- Model: `{bundle.metadata['model_version']}` (`{bundle.feature_group}`)
- Frozen Bundle SHA256: `{actual_sha}`
- Calibration: `{bundle.calibration_method}` (`2025-02-01..2025-03-31`)
- Threshold: `{bundle.threshold:.4f}` (`max_f1` on `2025-04-01..2025-05-31`)
- Prospective Holdout Window: `{metrics['period_start']}..{metrics['period_end']}` ({prospective_df.height:,} rows, prevalence `{metrics['positive_rate']:.4f}`)

## Prospective Holdout Metrics (`2025-10..2025-11`)
- PR-AUC: `{metrics['pr_auc']:.4f}` (uplift vs prevalence: `+{metrics['pr_auc_uplift_abs']:.4f}` / `+{metrics['pr_auc_uplift_rel']*100:.2f}%`, 95% block-bootstrap CI `[{metrics['bootstrap_ci_95']['pr_auc']['ci95_low']:.4f}, {metrics['bootstrap_ci_95']['pr_auc']['ci95_high']:.4f}]`)
- ROC-AUC: `{metrics['roc_auc']:.4f}` (95% CI `[{metrics['bootstrap_ci_95']['roc_auc']['ci95_low']:.4f}, {metrics['bootstrap_ci_95']['roc_auc']['ci95_high']:.4f}]`)
- Precision (at max-F1 `{bundle.threshold:.4f}`): `{metrics['precision']:.4f}` | Recall: `{metrics['recall']:.4f}` | F1: `{metrics['f1']:.4f}`
- Precision-targeted operating point (`{p35_thr:.4f}`): Precision `{metrics['precision_target_35_eval']['precision']:.4f}` | Recall `{metrics['precision_target_35_eval']['recall']:.4f}` | F1 `{metrics['precision_target_35_eval']['f1']:.4f}`
- Brier: `{metrics['brier']:.4f}` | LogLoss: `{metrics['log_loss']:.4f}` | ECE: `{metrics['ece']:.4f}`
- Precision@Top5%: `{metrics['precision_at_top_5pct']:.4f}` | Precision@Top10%: `{metrics['precision_at_top_10pct']:.4f}`
"""
        if paired_vs_v2 is not None:
            p_ci = paired_vs_v2["paired_delta_v2_1_minus_v2"]["delta_pr_auc"]
            once_md += (
                f"\n## Post-Freeze Paired Comparison vs Locked V2 (`{paired_vs_v2['v2_model_version']}`, `{paired_vs_v2['v2_feature_group']}`)\n"
                f"- V2 PR-AUC on `2025-10..11`: `{paired_vs_v2['v2_metrics']['pr_auc']:.4f}` | ROC-AUC: `{paired_vs_v2['v2_metrics']['roc_auc']:.4f}` | Brier: `{paired_vs_v2['v2_metrics']['brier']:.4f}`\n"
                f"- Paired delta PR-AUC (`V2.1 - V2`): `{p_ci['mean']:+.5f}` (95% block CI `[{p_ci['ci95_low']:+.5f}, {p_ci['ci95_high']:+.5f}]`, `P(>0)={p_ci['prob_positive']:.3f}`)\n"
            )
        (out_dir / "PROSPECTIVE_EVAL_ONCE.md").write_text(once_md, encoding="utf-8")
        manifest["prospective_holdout_evaluated_once"] = True
        manifest["prospective_holdout_window"] = f"{metrics['period_start']}..{metrics['period_end']}"
        manifest["prospective_holdout_manifest_sha256"] = (
            file_sha256(Path("data/manifests/bts_v2_1_future_holdout.json"))
            if Path("data/manifests/bts_v2_1_future_holdout.json").exists()
            else "unavailable"
        )
        args.manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return 0

    # Pre-holdout freeze run: fit on <2025-02-01, cal 2025-02..03, thr 2025-04..05, check 2025-06..08 & 2025-09 reproducibility
    params = yaml.safe_load(args.params_yaml.read_text(encoding="utf-8"))
    pattern = str(args.data_root / "year=*" / "month=*" / "flights.parquet")
    dev_all = (
        pl.scan_parquet(pattern)
        .filter(pl.col("flight_date") < date(2025, 9, 1))
        .collect(engine="streaming")
        .sort("flight_date")
    )

    def _window(cfg_key: str) -> pl.DataFrame:
        win = splits_cfg[cfg_key]
        s_d = date.fromisoformat(str(win["start"]))
        e_d = date.fromisoformat(str(win["end_exclusive"]))
        return dev_all.filter((pl.col("flight_date") >= s_d) & (pl.col("flight_date") < e_d))

    base_train = dev_all.filter(pl.col("flight_date") < date(2025, 2, 1))
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

    hist_v1_df = _window("historical_v1_reference")
    hist_v1_raw = frozen_model.predict_proba(build_v2_feature_frame(hist_v1_df, feature_group))[:, 1]
    hist_v1_y = hist_v1_df.get_column("dep_del15").to_numpy().astype(int)

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
    cal_table.to_csv("reports/metrics/v2_1_calibration.csv", index=False)
    best_cal_method: CalibrationMethod = str(cal_table.iloc[0]["calibration_method"])  # type: ignore[assignment]
    final_calibrator = fit_calibrator(best_cal_method, cal_raw, cal_y, seed=args.seed)

    thr_df = _window("threshold_selection")
    thr_raw = frozen_model.predict_proba(build_v2_feature_frame(thr_df, feature_group))[:, 1]
    thr_probs = final_calibrator.predict(thr_raw)
    thr_y = thr_df.get_column("dep_del15").to_numpy().astype(int)
    sweep = threshold_sweep(thr_y, thr_probs)
    sweep.to_csv("reports/metrics/v2_1_threshold_grid.csv", index=False)
    selected_threshold = float(sweep.sort_values("f1", ascending=False).iloc[0]["threshold"])
    p35_rows = sweep.loc[sweep["precision"] >= 0.35]
    precision35_thr = float(p35_rows.sort_values("threshold").iloc[0]["threshold"]) if not p35_rows.empty else selected_threshold

    hist_v1_probs = final_calibrator.predict(hist_v1_raw)
    hist_v1_metrics = evaluate_binary_metrics(hist_v1_y, hist_v1_probs, selected_threshold)

    pre_df = _window("pre_holdout_validation")
    pre_probs = final_calibrator.predict(
        frozen_model.predict_proba(build_v2_feature_frame(pre_df, feature_group))[:, 1]
    )
    pre_y = pre_df.get_column("dep_del15").to_numpy().astype(int)
    pre_metrics = evaluate_binary_metrics(pre_y, pre_probs, selected_threshold)

    summary_rows = [
        {"stage": "historical_v1_overlap_2025_01", **hist_v1_metrics},
        {"stage": "calibration_eval_slice", **cal_table.iloc[0].drop("calibration_method").to_dict()},
        {
            "stage": "threshold_selection_2025_04_05",
            **evaluate_binary_metrics(thr_y, thr_probs, selected_threshold),
        },
        {"stage": "pre_holdout_validation_2025_06_08", **pre_metrics},
    ]

    if args.v2_holdout_root.exists():
        h2_pattern = str(args.v2_holdout_root / "year=*" / "month=*" / "flights.parquet")
        h2_df = pl.scan_parquet(h2_pattern).collect(engine="streaming").sort("flight_date")
        h2_probs = final_calibrator.predict(
            frozen_model.predict_proba(build_v2_feature_frame(h2_df, feature_group))[:, 1]
        )
        h2_y = h2_df.get_column("dep_del15").to_numpy().astype(int)
        h2_metrics = evaluate_binary_metrics(h2_y, h2_probs, selected_threshold)
        summary_rows.append(
            {
                "stage": "REPRODUCIBILITY_CHECK_VIEWED_IN_V2_2025_09",
                **h2_metrics,
            }
        )

    lookup, defaults, airport_timezones = route_reference(base_train)
    airport_lookup = _airport_lookup_from_frame(base_train)
    for code, tz in airport_timezones.items():
        airport_lookup.setdefault(code, {})["timezone"] = tz

    bundle_meta: dict[str, Any] = {
        "model_version": f"flight-only-v2.1.0-{model_name}",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": model_name,
        "feature_group": feature_group,
        "feature_columns": build_v2_feature_frame(base_train.head(1), feature_group).columns.tolist(),
        "feature_contract_version": f"t24-v2.1-{feature_group}",
        "eval_strategy": "refit_full_train",
        "require_origin_timezone": True,
        "target": "dep_del15",
        "prediction_horizon_hours": 24,
        "calibration_method": best_cal_method,
        "threshold": selected_threshold,
        "threshold_precision_target_35": precision35_thr,
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
        "weather_status": "not_used_flight_only_v2_1",
        "data_freshness": "bts_v2_2023_01_to_2025_08",
    }

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
    bundle_sha = _sha256(args.bundle_output)

    manifest = {
        "model_version": bundle_meta["model_version"],
        "bundle_path": str(args.bundle_output.as_posix()),
        "bundle_sha256": bundle_sha,
        "bundle_bytes": args.bundle_output.stat().st_size,
        "feature_group": feature_group,
        "feature_contract_version": bundle_meta["feature_contract_version"],
        "selected_model": model_name,
        "eval_strategy": "refit_full_train",
        "calibration_method": best_cal_method,
        "threshold": selected_threshold,
        "threshold_precision_target_35": precision35_thr,
        "weather_status": bundle_meta["weather_status"],
        "data_manifest_sha256": bundle_meta["data_manifest_sha256"],
        "code_commit": bundle_meta["code_commit"],
        "package_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        },
        "prospective_holdout_evaluated_once": False,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv("reports/metrics/v2_1_fold_summary.csv", index=False)
    print(json.dumps(summary_rows, indent=2))
    print("FROZEN_V2_1_BUNDLE_SHA256:", bundle_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
