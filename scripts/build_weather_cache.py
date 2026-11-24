"""Build bounded Open-Meteo issued-run weather cache and matched-sample ablation."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from flight_delay.data.temporal import TemporalFold
from flight_delay.data.weather_asof import (
    WEATHER_BASE_VARIABLES,
    fetch_weather_cache_for_airports,
    join_weather_asof,
)
from flight_delay.features.v2 import build_v2_feature_frame, v2_feature_columns
from flight_delay.models.tuning_v2 import _fit_predict_onehot, evaluate_params

WEATHER_FEATURES = [
    *[f"origin_{v}" for v in WEATHER_BASE_VARIABLES],
    *[f"dest_{v}" for v in WEATHER_BASE_VARIABLES],
    "origin_weather_missing",
    "dest_weather_missing",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/weather_v2.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--airports", type=Path, default=Path("data/external/airports.csv"))
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    start_date = date.fromisoformat(str(cfg["start_date"]))
    end_date = date.fromisoformat(str(cfg["end_date"]))
    top_n = int(cfg["top_n_airports"])
    cache_path = Path(cfg["cache_path"])
    manifest_path = Path(cfg["manifest_path"])

    pattern = str(args.data_root / "year=*" / "month=*" / "flights.parquet")
    dev = (
        pl.scan_parquet(pattern)
        .filter((pl.col("flight_date") >= start_date) & (pl.col("flight_date") <= end_date))
        .collect(engine="streaming")
    )
    top_airports = (
        dev.group_by("origin")
        .len()
        .sort("len", descending=True)
        .head(top_n)
        .get_column("origin")
        .to_list()
    )
    airports_ref = (
        pd.read_csv(args.airports)
        .loc[lambda d: d["iata_code"].isin(top_airports), ["iata_code", "latitude", "longitude"]]
        .dropna()
    )

    if not args.skip_fetch or not cache_path.exists():
        weather_pl, logs = fetch_weather_cache_for_airports(
            airports_ref,
            start_date=str(cfg["start_date"]),
            end_date=str(cfg["end_date"]),
            model=str(cfg["model"]),
            lead_days=int(cfg["lead_days"]),
            availability_lag_hours=int(cfg["assumed_availability_lag_hours"]),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        weather_pl.write_parquet(cache_path, compression="zstd")
        manifest_path.write_text(json.dumps({"config": cfg, "logs": logs}, indent=2), encoding="utf-8")
    else:
        weather_pl = pl.read_parquet(cache_path)

    if args.fetch_only:
        print(f"Fetched weather cache: {weather_pl.height} rows -> {cache_path}")
        return 0

    weather_pd = weather_pl.to_pandas()
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
    matched_mask = (with_both["origin_weather_missing"] == 0) & (
        with_both["dest_weather_missing"] == 0
    )
    matched_df = with_both.loc[matched_mask].copy().reset_index(drop=True)

    # Coverage report
    coverage_md = f"""# Weather Coverage & As-Of Provenance Report (V2)

- Provider: `{cfg['provider']}` (`{cfg['model']}`, lead `{cfg['lead_days']}` days, assumed availability lag `{cfg['assumed_availability_lag_hours']}`h -> available at `T-42h <= T-24h`)
- Archive gap: 2023-01..2024-03 is not covered by `{cfg['model']}` previous-runs archive; weather study is restricted to `{cfg['start_date']}..{cfg['end_date']}`.
- Top airports fetched: {len(airports_ref)} airports
- Flights in window `{cfg['start_date']}..{cfg['end_date']}`: {len(with_both):,}
- Origin weather available: {(with_both['origin_weather_missing'] == 0).sum():,} ({(with_both['origin_weather_missing'] == 0).mean():.2%})
- Destination weather available: {(with_both['dest_weather_missing'] == 0).sum():,} ({(with_both['dest_weather_missing'] == 0).mean():.2%})
- Matched-coverage (both origin and dest available): {len(matched_df):,} ({matched_mask.mean():.2%})
"""
    Path("docs/weather_coverage_report.md").write_text(coverage_md, encoding="utf-8")

    # Matched-sample temporal ablation: train 2024-04..2024-12, val 2025-02..2025-08 (excl Jan 2025 V1 ref and Sep 2025 holdout)
    folds = [
        TemporalFold(
            name="weather_matched_2025_02_to_05",
            train_start=date(2024, 4, 1),
            train_end_exclusive=date(2025, 1, 1),
            validation_start=date(2025, 2, 1),
            validation_end_exclusive=date(2025, 5, 1),
        ),
        TemporalFold(
            name="weather_matched_2025_05_to_08",
            train_start=date(2024, 4, 1),
            train_end_exclusive=date(2025, 5, 1),
            validation_start=date(2025, 5, 1),
            validation_end_exclusive=date(2025, 9, 1),
        ),
    ]
    labels = matched_df["dep_del15"].to_numpy().astype(int)
    dates = pd.to_datetime(matched_df["flight_date"])
    f3_features = build_v2_feature_frame(matched_df, "F3")
    f5_features = pd.concat([f3_features, matched_df[WEATHER_FEATURES]], axis=1)

    xgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.06,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.05,
    }
    f3_results = evaluate_params(
        model_name="xgboost",
        params=xgb_params,
        features=f3_features,
        labels=labels,
        dates=dates,
        folds=folds,
        feature_group="F3",
        seed=42,
    )

    num_f3, cat_f3 = v2_feature_columns("F3")
    num_f5 = num_f3 + WEATHER_FEATURES

    f5_rows = []
    for fold in folds:
        tr_m = (dates >= pd.Timestamp(fold.train_start)) & (dates < pd.Timestamp(fold.train_end_exclusive))
        va_m = (dates >= pd.Timestamp(fold.validation_start)) & (dates < pd.Timestamp(fold.validation_end_exclusive))
        scores, fit_s = _fit_predict_onehot(
            "xgboost",
            xgb_params,
            num_f5,
            cat_f3,
            f5_features.loc[tr_m],
            labels[tr_m],
            f5_features.loc[va_m],
            42,
            dates_train=dates.loc[tr_m],
        )
        f5_rows.append(
            {
                "fold": fold.name,
                "rows_train": int(tr_m.sum()),
                "rows_validation": int(va_m.sum()),
                "positive_rate_validation": float(labels[va_m].mean()),
                "pr_auc": float(average_precision_score(labels[va_m], scores)),
                "roc_auc": float(roc_auc_score(labels[va_m], scores)),
                "brier": float(brier_score_loss(labels[va_m], scores)),
                "fit_seconds": fit_s,
            }
        )

    comp_rows = []
    for r in f3_results:
        comp_rows.append({"variant": "F3_flight_only_matched", **r.__dict__})
    for r in f5_rows:
        comp_rows.append({"variant": "F5_flight_plus_weather_matched", **r})
    comp_df = pd.DataFrame(comp_rows)
    Path("reports/ablation").mkdir(parents=True, exist_ok=True)
    comp_df.to_csv("reports/ablation/weather_comparison.csv", index=False)
    print(comp_df.groupby("variant")[["pr_auc", "roc_auc", "brier"]].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
