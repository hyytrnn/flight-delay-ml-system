"""Gate G4: Error-driven EDA, 2023-2025 data quality audit, and monthly/threshold robustness."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml

from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame
from flight_delay.models.calibrate_v2 import (
    calendar_block_bootstrap_ci,
    evaluate_binary_metrics,
    fit_calibrator,
    threshold_sweep,
)
from flight_delay.models.holdout_helpers import fit_frozen_model
from flight_delay.models.tuning_v2 import (
    ModelName,
    evaluate_params_with_oof,
    load_splits,
)


def _build_data_quality(dev: pl.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    counts = pd.read_csv("reports/data/v2/monthly_counts.csv")
    rates = pd.read_csv("reports/data/v2/monthly_label_rates.csv")
    merged = counts.merge(rates, on=["year", "month"])
    merged["year_month"] = merged["year"].astype(str) + "-" + merged["month"].astype(int).map(lambda m: f"{m:02d}")
    mean_rate = float(merged["sample_positive_rate"].mean())
    std_rate = float(merged["sample_positive_rate"].std())
    merged["rate_zscore"] = (merged["sample_positive_rate"] - mean_rate) / max(std_rate, 1e-6)
    merged["seasonal_anomaly_flag"] = merged["rate_zscore"].abs() >= 1.5

    null_rates = {
        col: float(dev.get_column(col).null_count() / max(1, dev.height))
        for col in dev.columns
    }
    dup_keys = int(dev.height - dev.select("flight_key").n_unique()) if "flight_key" in dev.columns else -1
    summary = {
        "total_dev_rows": dev.height,
        "months_covered": len(merged),
        "duplicate_flight_keys": dup_keys,
        "max_abs_sampling_positive_rate_delta": float(merged["positive_rate_delta"].abs().max()),
        "mean_sample_positive_rate": mean_rate,
        "min_month_positive_rate": float(merged["sample_positive_rate"].min()),
        "max_month_positive_rate": float(merged["sample_positive_rate"].max()),
        "non_zero_null_columns": {k: v for k, v in null_rates.items() if v > 0.0},
    }
    return merged, summary


def _assign_hour_block(series: pd.Series) -> pd.Series:
    h = pd.to_datetime(series).dt.hour
    bins = [-1, 5, 9, 13, 17, 21, 24]
    labels = ["00-05", "06-09", "10-13", "14-17", "18-21", "22-23"]
    return pd.cut(h, bins=bins, labels=labels).astype(str)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_v2_1.yaml"))
    parser.add_argument("--model-name", type=str, default="xgboost")
    parser.add_argument("--selected-group", type=str, default=None)
    parser.add_argument("--params-yaml", type=Path, default=None)
    parser.add_argument("--oof-cache", type=Path, default=Path("data/interim/v2_1_champion_oof_2024.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/eda_v2_1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.selected_group is None:
        sel_path = Path("configs/best_v2_1/selection_summary.json")
        args.selected_group = (
            json.loads(sel_path.read_text(encoding="utf-8"))["selected_feature_group"]
            if sel_path.exists()
            else "F0"
        )
    group: FeatureGroup = args.selected_group  # type: ignore[assignment]
    model_name: ModelName = args.model_name  # type: ignore[assignment]
    if args.params_yaml is None:
        args.params_yaml = Path(f"configs/best_v2_1/{model_name}_{group}_final.yaml")
    params = yaml.safe_load(args.params_yaml.read_text(encoding="utf-8"))

    out_dir = args.output_root
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(args.data_root / "year=*" / "month=*" / "flights.parquet")
    dev = (
        pl.scan_parquet(pattern)
        .filter(pl.col("flight_date") < date(2025, 9, 1))
        .collect(engine="streaming")
        .sort("flight_date")
    )

    dq_df, dq_summary = _build_data_quality(dev)
    dq_df.to_csv(out_dir / "data_quality_2023_2025.csv", index=False)
    (out_dir / "data_quality_summary.json").write_text(
        json.dumps(dq_summary, indent=2), encoding="utf-8"
    )

    # 1. Obtain 2024 OOF predictions (reuse cache if matches)
    if args.oof_cache.exists():
        oof_2024 = pd.read_parquet(args.oof_cache)
    else:
        folds = load_splits(args.splits)
        dev_pre25 = dev.filter(pl.col("flight_date") < date(2025, 1, 1))
        _, oof_2024 = evaluate_params_with_oof(
            model_name=model_name,
            params=params,
            features=build_v2_feature_frame(dev_pre25, group),
            labels=dev_pre25.get_column("dep_del15").to_numpy().astype(int),
            dates=pd.to_datetime(dev_pre25.get_column("flight_date").to_pandas()),
            folds=folds,
            feature_group=group,
            seed=args.seed,
            eval_strategy="refit_full_train",
        )

    # 2. Fit base model on 2023-01..2025-01 and evaluate 2025-02..2025-08
    base_train = dev.filter(pl.col("flight_date") < date(2025, 2, 1))
    frozen_model = fit_frozen_model(
        model_name=model_name,
        params=params,
        feature_group=group,
        x_train=build_v2_feature_frame(base_train, group),
        y_train=base_train.get_column("dep_del15").to_numpy().astype(int),
        dates_train=pd.to_datetime(base_train.get_column("flight_date").to_pandas()),
        seed=args.seed,
    )
    fwd_df = dev.filter(
        (pl.col("flight_date") >= date(2025, 2, 1)) & (pl.col("flight_date") < date(2025, 9, 1))
    )
    fwd_feats = build_v2_feature_frame(fwd_df, group)
    fwd_probs = np.clip(frozen_model.predict_proba(fwd_feats)[:, 1], 1e-6, 1.0 - 1e-6)
    fwd_pd = fwd_df.select(
        ["flight_date", "reporting_airline", "origin", "dest", "scheduled_departure_local", "dep_del15"]
    ).to_pandas()
    fwd_pd["y_true"] = fwd_pd["dep_del15"].astype(int)
    fwd_pd["y_prob"] = fwd_probs

    # Choose thresholds ONLY on 2025-04..2025-05
    thr_mask = (pd.to_datetime(fwd_pd["flight_date"]) >= pd.Timestamp("2025-04-01")) & (
        pd.to_datetime(fwd_pd["flight_date"]) < pd.Timestamp("2025-06-01")
    )
    sweep_thr = threshold_sweep(
        fwd_pd.loc[thr_mask, "y_true"].to_numpy(), fwd_pd.loc[thr_mask, "y_prob"].to_numpy()
    )
    thr_f1 = float(sweep_thr.sort_values("f1", ascending=False).iloc[0]["threshold"])
    prec35 = sweep_thr.loc[sweep_thr["precision"] >= 0.35]
    thr_p35 = float(prec35.sort_values("threshold").iloc[0]["threshold"]) if not prec35.empty else thr_f1

    # Combine 2024 OOF and 2025-02..08 forward predictions for monthly trajectory
    oof_2024_slim = pd.DataFrame(
        {
            "flight_date": pd.to_datetime(oof_2024["flight_date"]),
            "y_true": oof_2024["y_true"].astype(int),
            "y_prob": oof_2024["y_prob"].astype(float),
            "eval_source": "2024_oof_cv",
        }
    )
    fwd_slim = pd.DataFrame(
        {
            "flight_date": pd.to_datetime(fwd_pd["flight_date"]),
            "y_true": fwd_pd["y_true"].astype(int),
            "y_prob": fwd_pd["y_prob"].astype(float),
            "eval_source": "2025_frozen_forward",
        }
    )
    monthly_all = pd.concat([oof_2024_slim, fwd_slim], ignore_index=True)
    monthly_all["year_month"] = monthly_all["flight_date"].dt.strftime("%Y-%m")

    m_rows = []
    for ym, grp in monthly_all.groupby("year_month", sort=True):
        y_m = grp["y_true"].to_numpy()
        p_m = grp["y_prob"].to_numpy()
        m_dict = evaluate_binary_metrics(y_m, p_m, threshold=thr_f1)
        ci = calendar_block_bootstrap_ci(y_m, p_m, grp["flight_date"], threshold=thr_f1, n_bootstrap=200, seed=args.seed)
        m_rows.append(
            {
                "year_month": str(ym),
                "eval_source": str(grp["eval_source"].iloc[0]),
                "alert_rate_fixed_thr": float((p_m >= thr_f1).mean()),
                "pr_auc_ci95_low": ci["pr_auc"]["ci95_low"],
                "pr_auc_ci95_high": ci["pr_auc"]["ci95_high"],
                **m_dict,
            }
        )
    monthly_metrics_df = pd.DataFrame(m_rows)
    monthly_metrics_df.to_csv(out_dir / "monthly_metrics.csv", index=False)

    # 3. Threshold policies & cost curve on 2025-04..05 & 2025-06..08
    policy_rows = []
    fwd_pd["year_month"] = pd.to_datetime(fwd_pd["flight_date"]).dt.strftime("%Y-%m")
    for w_name, w_mask in [
        ("threshold_sel_2025_04_05", fwd_pd["year_month"].isin(["2025-04", "2025-05"])),
        ("pre_holdout_2025_06", fwd_pd["year_month"] == "2025-06"),
        ("pre_holdout_2025_07", fwd_pd["year_month"] == "2025-07"),
        ("pre_holdout_2025_08", fwd_pd["year_month"] == "2025-08"),
        ("pre_holdout_all_2025_06_08", fwd_pd["year_month"].isin(["2025-06", "2025-07", "2025-08"])),
    ]:
        yw = fwd_pd.loc[w_mask, "y_true"].to_numpy()
        pw = fwd_pd.loc[w_mask, "y_prob"].to_numpy()
        for pol_name, thr_val in [
            ("fixed_max_f1", thr_f1),
            ("precision_target_35", thr_p35),
            ("top_10pct_budget", float(np.quantile(pw, 0.90))),
            ("top_5pct_budget", float(np.quantile(pw, 0.95))),
        ]:
            pred = (pw >= thr_val).astype(int)
            tp = int(((pred == 1) & (yw == 1)).sum())
            fp = int(((pred == 1) & (yw == 0)).sum())
            fn = int(((pred == 0) & (yw == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            policy_rows.append(
                {
                    "window": w_name,
                    "policy": pol_name,
                    "threshold": thr_val,
                    "rows": len(yw),
                    "prevalence": float(yw.mean()),
                    "alert_rate": float(pred.mean()),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "false_alerts": fp,
                    "missed_delays": fn,
                }
            )
    pd.DataFrame(policy_rows).to_csv(out_dir / "threshold_policy_comparison.csv", index=False)

    pre_mask = fwd_pd["year_month"].isin(["2025-06", "2025-07", "2025-08"])
    y_pre = fwd_pd.loc[pre_mask, "y_true"].to_numpy()
    p_pre = fwd_pd.loc[pre_mask, "y_prob"].to_numpy()
    cost_frames = []
    for ratio_name, fn_c, fp_c in [("cost_2_to_1", 2.0, 1.0), ("cost_4_to_1", 4.0, 1.0), ("cost_8_to_1", 8.0, 1.0)]:
        c_df = threshold_sweep(y_pre, p_pre, fn_cost=fn_c, fp_cost=fp_c)
        c_df["cost_regime"] = ratio_name
        cost_frames.append(c_df)
    pd.concat(cost_frames, ignore_index=True).to_csv(out_dir / "cost_curve_v2_1.csv", index=False)

    # 4. Rolling calibration check (month m-1 Platt vs static none on 2025-03..08)
    months_fwd = sorted(fwd_pd["year_month"].unique())
    roll_rows = []
    for idx in range(1, len(months_fwd)):
        prev_m, curr_m = months_fwd[idx - 1], months_fwd[idx]
        p_tr = fwd_pd.loc[fwd_pd["year_month"] == prev_m, "y_prob"].to_numpy()
        y_tr = fwd_pd.loc[fwd_pd["year_month"] == prev_m, "y_true"].to_numpy()
        p_va = fwd_pd.loc[fwd_pd["year_month"] == curr_m, "y_prob"].to_numpy()
        y_va = fwd_pd.loc[fwd_pd["year_month"] == curr_m, "y_true"].to_numpy()
        cal_platt = fit_calibrator("platt", p_tr, y_tr, seed=args.seed)
        raw_m = evaluate_binary_metrics(y_va, p_va, threshold=thr_f1)
        platt_m = evaluate_binary_metrics(y_va, cal_platt.predict(p_va), threshold=thr_f1)
        roll_rows.append({"month": curr_m, "calibration": "static_none", **raw_m})
        roll_rows.append({"month": curr_m, "calibration": "rolling_platt_prev_month", **platt_m})
    pd.DataFrame(roll_rows).to_csv(out_dir / "rolling_recalibration_check.csv", index=False)

    # 5. Detailed error slices on 2025-06..08 pre-holdout
    pre_df = fwd_pd.loc[pre_mask].copy().reset_index(drop=True)
    train_pd = base_train.select(["origin", "dest", "reporting_airline"]).to_pandas()
    train_routes = train_pd["origin"] + "|" + train_pd["dest"]
    route_counts = train_routes.value_counts().to_dict()
    pre_df["route"] = pre_df["origin"] + "|" + pre_df["dest"]
    pre_df["route_support"] = pre_df["route"].map(route_counts).fillna(0).astype(int)
    pre_df["route_support_bucket"] = np.select(
        [
            pre_df["route_support"] == 0,
            pre_df["route_support"] < 100,
            pre_df["route_support"] < 1000,
        ],
        ["unseen_0", "low_1_to_99", "medium_100_to_999"],
        default="high_1000_plus",
    )
    pre_df["dep_hour_block"] = _assign_hour_block(pre_df["scheduled_departure_local"])

    f1_feats = build_v2_feature_frame(dev.filter((pl.col("flight_date") >= date(2025, 6, 1)) & (pl.col("flight_date") < date(2025, 9, 1))), "F1")
    d_hol = f1_feats["days_to_nearest_us_federal_holiday"].to_numpy()
    pre_df["holiday_window"] = np.select([d_hol == 0, d_hol <= 3], ["0d_holiday", "1_to_3d"], default="4plus_d")

    top25 = set(train_pd["origin"].value_counts().head(25).index.tolist())
    pre_df["weather_top25_flag"] = np.select(
        [pre_df["origin"].isin(top25) & pre_df["dest"].isin(top25), pre_df["origin"].isin(top25) | pre_df["dest"].isin(top25)],
        ["both_top25", "partial_top25"],
        default="neither_top25",
    )

    slice_rows = []
    for col, max_vals in [
        ("reporting_airline", 15),
        ("origin", 15),
        ("dest", 15),
        ("route_support_bucket", 10),
        ("dep_hour_block", 10),
        ("holiday_window", 10),
        ("weather_top25_flag", 10),
    ]:
        vals = pre_df[col].value_counts().head(max_vals).index.tolist()
        for val in vals:
            sub = pre_df.loc[pre_df[col] == val]
            if len(sub) < 50 or sub["y_true"].nunique() < 2:
                continue
            m = evaluate_binary_metrics(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy(), threshold=thr_f1)
            slice_rows.append({"slice_col": col, "slice_val": str(val), **m})
    pd.DataFrame(slice_rows).to_csv(out_dir / "per_slice_metrics.csv", index=False)

    # Anonymized high-confidence errors
    hc_fp = (
        pre_df.loc[(pre_df["y_true"] == 0) & (pre_df["y_prob"] >= 0.50)]
        .sort_values("y_prob", ascending=False)
        .head(40)
        .assign(error_type="high_conf_FP")
    )
    hc_fn = (
        pre_df.loc[(pre_df["y_true"] == 1) & (pre_df["y_prob"] <= 0.10)]
        .sort_values("y_prob", ascending=True)
        .head(40)
        .assign(error_type="high_conf_FN")
    )
    err_cols = ["error_type", "reporting_airline", "origin", "dest", "dep_hour_block", "route_support_bucket", "holiday_window", "y_prob", "y_true"]
    pd.concat([hc_fp[err_cols], hc_fn[err_cols]], ignore_index=True).to_csv(
        out_dir / "high_conf_errors_anonymized.csv", index=False
    )

    # 6. Save 4 labeled diagnostic figures
    plt.figure(figsize=(10.5, 4.5))
    x_idx = np.arange(len(monthly_metrics_df))
    plt.errorbar(
        x_idx,
        monthly_metrics_df["pr_auc"],
        yerr=[
            monthly_metrics_df["pr_auc"] - monthly_metrics_df["pr_auc_ci95_low"],
            monthly_metrics_df["pr_auc_ci95_high"] - monthly_metrics_df["pr_auc"],
        ],
        fmt="-o",
        color="#0f5b78",
        label="Monthly PR-AUC (95% block CI)",
    )
    plt.plot(x_idx, monthly_metrics_df["positive_rate"], "--s", color="#b03a2e", label="Delay prevalence")
    plt.xticks(x_idx, monthly_metrics_df["year_month"], rotation=45, ha="right")
    plt.ylabel("Score / Rate")
    plt.title("Monthly PR-AUC vs Base Prevalence (2024 OOF + 2025 Frozen Validation)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / "01_monthly_pr_auc_and_prevalence.png", dpi=160)
    plt.close()

    pol_df = pd.DataFrame(policy_rows)
    sub_pol = pol_df[pol_df["window"].isin(["pre_holdout_2025_06", "pre_holdout_2025_07", "pre_holdout_2025_08"])]
    plt.figure(figsize=(8.8, 4.4))
    for pol_name, grp in sub_pol.groupby("policy"):
        plt.plot(grp["window"], grp["alert_rate"], marker="o", label=f"{pol_name} (alert_rate)")
    plt.xticks(rotation=15)
    plt.ylabel("Alert Rate")
    plt.title("Alert Rate Stability by Decision Policy (Summer Pre-Holdout 2025-06..08)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / "02_decision_policy_stability_by_month.png", dpi=160)
    plt.close()

    hr_df = pd.DataFrame([r for r in slice_rows if r["slice_col"] == "dep_hour_block"])
    if not hr_df.empty:
        plt.figure(figsize=(8.0, 4.4))
        plt.bar(hr_df["slice_val"], hr_df["pr_auc_uplift_abs"], color="#2e8b57")
        plt.xlabel("Local Departure Hour Block")
        plt.ylabel("PR-AUC Uplift Over Slice Prevalence")
        plt.title("PR-AUC Uplift by Departure Hour Block (2025-06..08)")
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        plt.savefig(fig_dir / "03_slice_pr_auc_uplift_and_support.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8.0, 4.4))
    for ratio_name, grp in pd.concat(cost_frames).groupby("cost_regime"):
        plt.plot(grp["threshold"], grp["expected_cost_per_flight"], label=ratio_name)
    plt.xlabel("Decision Threshold")
    plt.ylabel("Expected Cost per Flight")
    plt.title("Operational Expected Cost Curves (2025-06..08)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / "04_calibration_and_cost_curves.png", dpi=160)
    plt.close()

    print("Saved Gate G4 error analysis artifacts to", out_dir, "thr_f1=", thr_f1, "thr_p35=", thr_p35)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
