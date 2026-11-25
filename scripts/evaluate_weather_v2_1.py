"""Gate G6: Weather coverage, selection-bias audit, and 3-fold matched + fallback ablation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from flight_delay.data.temporal import TemporalFold
from flight_delay.data.weather_asof import WEATHER_BASE_VARIABLES, join_weather_asof
from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame
from flight_delay.models.calibrate_v2 import (
    evaluate_binary_metrics,
    paired_calendar_block_bootstrap,
)
from flight_delay.models.tuning_v2 import XGBOOST_FIXED_PARAMS, evaluate_params_with_oof

WEATHER_FEATURES = [
    *[f"origin_{v}" for v in WEATHER_BASE_VARIABLES],
    *[f"dest_{v}" for v in WEATHER_BASE_VARIABLES],
    "origin_weather_missing",
    "dest_weather_missing",
]

WEATHER_FOLDS = [
    TemporalFold(
        name="weather_fold_1_q4_2024",
        train_start=date(2024, 4, 1),
        train_end_exclusive=date(2024, 10, 1),
        validation_start=date(2024, 10, 1),
        validation_end_exclusive=date(2025, 1, 1),
    ),
    TemporalFold(
        name="weather_fold_2_early_2025",
        train_start=date(2024, 4, 1),
        train_end_exclusive=date(2025, 1, 1),
        validation_start=date(2025, 2, 1),
        validation_end_exclusive=date(2025, 5, 1),
    ),
    TemporalFold(
        name="weather_fold_3_summer_2025",
        train_start=date(2024, 4, 1),
        train_end_exclusive=date(2025, 5, 1),
        validation_start=date(2025, 5, 1),
        validation_end_exclusive=date(2025, 9, 1),
    ),
]


def _coverage_tables(with_both: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = with_both.copy()
    df["year_month"] = pd.to_datetime(df["flight_date"]).dt.strftime("%Y-%m")
    df["origin_cov"] = (df["origin_weather_missing"] == 0).astype(float)
    df["dest_cov"] = (df["dest_weather_missing"] == 0).astype(float)
    df["both_cov"] = ((df["origin_weather_missing"] == 0) & (df["dest_weather_missing"] == 0)).astype(bool)

    month_rows = []
    for ym, grp in df.groupby("year_month", sort=True):
        m_grp = grp.loc[grp["both_cov"]]
        u_grp = grp.loc[~grp["both_cov"]]
        month_rows.append(
            {
                "year_month": str(ym),
                "rows_total": len(grp),
                "rows_matched_both": len(m_grp),
                "origin_covered_rate": float(grp["origin_cov"].mean()),
                "dest_covered_rate": float(grp["dest_cov"].mean()),
                "both_matched_rate": float(grp["both_cov"].mean()),
                "overall_positive_rate": float(grp["dep_del15"].mean()),
                "matched_positive_rate": float(m_grp["dep_del15"].mean()) if len(m_grp) else np.nan,
                "unmatched_positive_rate": float(u_grp["dep_del15"].mean()) if len(u_grp) else np.nan,
            }
        )

    slice_rows = []
    for col in ["reporting_airline", "origin"]:
        top_vals = df[col].value_counts().head(25).index.tolist()
        for val in top_vals:
            grp = df.loc[df[col] == val]
            m_grp = grp.loc[grp["both_cov"]]
            u_grp = grp.loc[~grp["both_cov"]]
            slice_rows.append(
                {
                    "slice_col": col,
                    "slice_val": str(val),
                    "rows_total": len(grp),
                    "origin_covered_rate": float(grp["origin_cov"].mean()),
                    "both_matched_rate": float(grp["both_cov"].mean()),
                    "matched_positive_rate": float(m_grp["dep_del15"].mean()) if len(m_grp) else np.nan,
                    "unmatched_positive_rate": float(u_grp["dep_del15"].mean()) if len(u_grp) else np.nan,
                }
            )
    return pd.DataFrame(month_rows), pd.DataFrame(slice_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/weather_v2.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--selected-group", type=str, default=None)
    parser.add_argument("--params-yaml", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.selected_group is None:
        sel_path = Path("configs/best_v2_1/selection_summary.json")
        args.selected_group = (
            json.loads(sel_path.read_text(encoding="utf-8"))["selected_feature_group"]
            if sel_path.exists()
            else "F0"
        )
    selected_group: FeatureGroup = args.selected_group  # type: ignore[assignment]

    if args.params_yaml and args.params_yaml.exists():
        params = yaml.safe_load(args.params_yaml.read_text(encoding="utf-8"))
    else:
        params = XGBOOST_FIXED_PARAMS

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    start_d = date.fromisoformat(str(cfg["start_date"]))
    end_d = date.fromisoformat(str(cfg["end_date"]))
    weather_pl = pl.read_parquet(Path(cfg["cache_path"]))
    weather_pd = weather_pl.to_pandas()

    pattern = str(args.data_root / "year=*" / "month=*" / "flights.parquet")
    dev = (
        pl.scan_parquet(pattern)
        .filter((pl.col("flight_date") >= start_d) & (pl.col("flight_date") <= end_d))
        .collect(engine="streaming")
        .sort("flight_date")
    )
    dev_pd = dev.to_pandas()
    with_origin = join_weather_asof(
        dev_pd,
        weather_pd,
        role="origin",
        tolerance_minutes=int(cfg["tolerance_minutes"]),
        default_availability_lag_hours=int(cfg["assumed_availability_lag_hours"]),
    )
    with_both = join_weather_asof(
        with_origin,
        weather_pd,
        role="dest",
        tolerance_minutes=int(cfg["tolerance_minutes"]),
        default_availability_lag_hours=int(cfg["assumed_availability_lag_hours"]),
    )

    out_dir = Path("reports/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    m_cov, s_cov = _coverage_tables(with_both)
    m_cov.to_csv(out_dir / "weather_coverage_by_month.csv", index=False)
    s_cov.to_csv(out_dir / "weather_coverage_by_carrier_airport.csv", index=False)

    comparison_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    paired_cis: dict[str, object] = {}
    bias_rows: list[dict[str, object]] = []

    matched_mask = (with_both["origin_weather_missing"] == 0) & (
        with_both["dest_weather_missing"] == 0
    )

    for track_name, subset_df in [
        ("matched_both_top25", with_both.loc[matched_mask].copy().reset_index(drop=True)),
        ("all_flights_with_fallback", with_both.copy().reset_index(drop=True)),
    ]:
        base_feats = build_v2_feature_frame(subset_df, selected_group)
        f5_feats = pd.concat([base_feats, subset_df[WEATHER_FEATURES]], axis=1)
        labels = subset_df["dep_del15"].to_numpy().astype(int)
        dates = pd.to_datetime(subset_df["flight_date"])

        base_res, base_oof = evaluate_params_with_oof(
            model_name="xgboost",
            params=params,
            features=base_feats,
            labels=labels,
            dates=dates,
            folds=WEATHER_FOLDS,
            feature_group=selected_group,
            seed=args.seed,
            eval_strategy="refit_full_train",
        )
        f5_res, f5_oof = evaluate_params_with_oof(
            model_name="xgboost",
            params=params,
            features=f5_feats,
            labels=labels,
            dates=dates,
            folds=WEATHER_FOLDS,
            feature_group=selected_group,
            seed=args.seed,
            eval_strategy="refit_full_train",
            extra_numeric=WEATHER_FEATURES,
        )
        ci = paired_calendar_block_bootstrap(
            base_oof["y_true"].to_numpy(),
            base_oof["y_prob"].to_numpy(),
            f5_oof["y_prob"].to_numpy(),
            f5_oof["flight_date"],
            n_bootstrap=300,
            seed=args.seed,
        )
        paired_cis[track_name] = ci

        for v_label, res_list in [
            (f"{selected_group}_flight_only", base_res),
            (f"{selected_group}_plus_F5_weather", f5_res),
        ]:
            pr_scores = [r.pr_auc for r in res_list]
            is_f5 = "F5" in v_label
            comparison_rows.append(
                {
                    "track": track_name,
                    "variant": v_label,
                    "feature_group": selected_group,
                    "mean_pr_auc": float(np.mean(pr_scores)),
                    "std_pr_auc": float(np.std(pr_scores)),
                    "worst_fold_pr_auc": float(np.min(pr_scores)),
                    "mean_roc_auc": float(np.mean([r.roc_auc for r in res_list])),
                    "mean_brier": float(np.mean([r.brier for r in res_list])),
                    "mean_prevalence": float(np.mean([r.positive_rate_validation for r in res_list])),
                    "paired_delta_pr_auc": ci["delta_pr_auc"]["mean"] if is_f5 else 0.0,
                    "paired_ci95_low": ci["delta_pr_auc"]["ci95_low"] if is_f5 else 0.0,
                    "paired_ci95_high": ci["delta_pr_auc"]["ci95_high"] if is_f5 else 0.0,
                    "paired_delta_brier": ci["delta_brier"]["mean"] if is_f5 else 0.0,
                }
            )
            fold_rows.extend({"track": track_name, "variant": v_label, **asdict(r)} for r in res_list)

        if track_name == "all_flights_with_fallback":
            val_mask_all = np.zeros(len(subset_df), dtype=bool)
            for fold in WEATHER_FOLDS:
                val_mask_all |= (dates >= pd.Timestamp(fold.validation_start)).to_numpy() & (
                    dates < pd.Timestamp(fold.validation_end_exclusive)
                ).to_numpy()
            val_rows = subset_df.loc[val_mask_all].copy().reset_index(drop=True)
            is_matched_val = (
                (val_rows["origin_weather_missing"] == 0)
                & (val_rows["dest_weather_missing"] == 0)
            ).to_numpy()
            for model_label, oof_df in [
                (f"{selected_group}_flight_only", base_oof),
                (f"{selected_group}_plus_F5_weather", f5_oof),
            ]:
                for seg_name, seg_mask in [
                    ("matched_both_top25", is_matched_val),
                    ("unmatched_other_routes", ~is_matched_val),
                ]:
                    m_dict = evaluate_binary_metrics(
                        oof_df["y_true"].to_numpy()[seg_mask],
                        oof_df["y_prob"].to_numpy()[seg_mask],
                        threshold=0.185,
                    )
                    bias_rows.append({"model": model_label, "segment": seg_name, **m_dict})

    pd.DataFrame(comparison_rows).to_csv(out_dir / "weather_v2_1.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "weather_v2_1_folds.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(out_dir / "weather_selection_bias.csv", index=False)
    (out_dir / "weather_v2_1_paired_ci.json").write_text(
        json.dumps(paired_cis, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(comparison_rows)[["track", "variant", "mean_pr_auc", "paired_delta_pr_auc", "paired_ci95_low", "paired_ci95_high"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
