"""Generate support-aware V2 EDA without reading the final holdout."""

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
import seaborn as sns
import yaml
from sklearn.feature_selection import mutual_info_classif

from flight_delay.analysis.stats import (
    jensen_shannon_divergence,
    population_stability_index,
    total_variation_distance,
    wilson_interval,
)
from flight_delay.features.v2 import _nearest_holiday_distance


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def rate_table(lazy: pl.LazyFrame, groups: list[str]) -> pd.DataFrame:
    table = (
        lazy.group_by(groups)
        .agg(
            pl.len().alias("rows"),
            pl.col("dep_del15").sum().cast(pl.Int64).alias("positive"),
            pl.col("dep_del15").mean().alias("delay_rate"),
        )
        .sort(groups)
        .collect(engine="streaming")
        .to_pandas()
    )
    intervals = [wilson_interval(int(row.positive), int(row.rows)) for row in table.itertuples()]
    table["ci_low"] = [value[0] for value in intervals]
    table["ci_high"] = [value[1] for value in intervals]
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_v2.yaml"))
    parser.add_argument("--end-exclusive", type=date.fromisoformat, default=date(2025, 9, 1))
    parser.add_argument("--output-root", type=Path, default=Path("reports/eda_v2"))
    args = parser.parse_args()

    pattern = str(args.data_root / "year=*" / "month=*" / "flights.parquet")
    lazy = (
        pl.scan_parquet(pattern)
        .filter(pl.col("flight_date") < args.end_exclusive)
        .with_columns(
            pl.col("flight_date").dt.strftime("%Y-%m").alias("year_month"),
            pl.col("scheduled_departure_local").dt.hour().alias("departure_hour"),
            pl.col("scheduled_departure_local").dt.weekday().alias("weekday"),
            pl.when(pl.col("scheduled_departure_local").dt.month().is_in([12, 1, 2]))
            .then(pl.lit("winter"))
            .when(pl.col("scheduled_departure_local").dt.month().is_in([3, 4, 5]))
            .then(pl.lit("spring"))
            .when(pl.col("scheduled_departure_local").dt.month().is_in([6, 7, 8]))
            .then(pl.lit("summer"))
            .otherwise(pl.lit("autumn"))
            .alias("season"),
            pl.when(pl.col("distance") <= 500)
            .then(pl.lit("short"))
            .when(pl.col("distance") <= 1_000)
            .then(pl.lit("medium"))
            .when(pl.col("distance") <= 2_000)
            .then(pl.lit("long"))
            .otherwise(pl.lit("very_long"))
            .alias("distance_bin"),
            pl.when(pl.col("origin_departures_planned_1h") <= 10)
            .then(pl.lit("1-10"))
            .when(pl.col("origin_departures_planned_1h") <= 25)
            .then(pl.lit("11-25"))
            .when(pl.col("origin_departures_planned_1h") <= 50)
            .then(pl.lit("26-50"))
            .otherwise(pl.lit("51+"))
            .alias("origin_load_bin"),
        )
    )
    sns.set_theme(style="whitegrid")
    figures = args.output_root / "figures"
    stats_dir = args.output_root / "stats"
    figures.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    monthly = rate_table(lazy, ["year_month"])
    monthly.to_csv(stats_dir / "monthly_label_rates.csv", index=False)
    plt.figure(figsize=(10, 4.8))
    plt.plot(monthly["year_month"], monthly["delay_rate"], marker="o")
    plt.fill_between(monthly["year_month"], monthly["ci_low"], monthly["ci_high"], alpha=0.2)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("DepDel15 rate")
    plt.title("Monthly delay prevalence with 95% Wilson intervals")
    save(figures / "01_monthly_delay_rate.png")

    carriers = rate_table(lazy, ["reporting_airline"]).sort_values("rows", ascending=False).head(15)
    carriers.to_csv(stats_dir / "carrier_rates.csv", index=False)
    sns.barplot(data=carriers, x="delay_rate", y="reporting_airline", color="#39728c")
    plt.title("Delay rate for the 15 most supported carriers")
    save(figures / "02_carrier_delay_rate.png")

    origins = rate_table(lazy, ["origin"])
    origins = origins.loc[origins["rows"] >= 2_000].sort_values("delay_rate", ascending=False).head(20)
    origins.to_csv(stats_dir / "origin_rates_supported.csv", index=False)
    sns.barplot(data=origins, x="delay_rate", y="origin", color="#996633")
    plt.title("Highest origin delay rates with at least 2,000 flights")
    save(figures / "03_origin_delay_rate.png")

    hourly = rate_table(lazy, ["departure_hour"])
    hourly.to_csv(stats_dir / "hourly_rates.csv", index=False)
    sns.lineplot(data=hourly, x="departure_hour", y="delay_rate", marker="o")
    plt.title("Delay rate by scheduled local departure hour")
    save(figures / "04_departure_hour.png")

    weekday = rate_table(lazy, ["weekday"])
    sns.barplot(data=weekday, x="weekday", y="delay_rate", color="#668855")
    plt.title("Delay rate by local weekday")
    save(figures / "05_weekday.png")

    season = rate_table(lazy, ["season"])
    sns.barplot(data=season, x="season", y="delay_rate", color="#8064a2")
    plt.title("Delay rate by season")
    save(figures / "06_season.png")

    distance = rate_table(lazy, ["distance_bin"])
    sns.barplot(data=distance, x="distance_bin", y="delay_rate", color="#4f81bd")
    plt.title("Delay rate by scheduled distance")
    save(figures / "07_distance_bin.png")

    load = rate_table(lazy, ["origin_load_bin"])
    sns.barplot(data=load, x="origin_load_bin", y="delay_rate", color="#c0504d")
    plt.title("Delay rate by retrospective hourly origin schedule load")
    save(figures / "08_origin_load_proxy.png")

    top_carriers = carriers["reporting_airline"].head(10).tolist()
    heatmap = rate_table(
        lazy.filter(pl.col("reporting_airline").is_in(top_carriers)),
        ["reporting_airline", "departure_hour"],
    ).pivot(index="reporting_airline", columns="departure_hour", values="delay_rate")
    sns.heatmap(heatmap, cmap="YlOrRd", vmin=0, vmax=float(np.nanquantile(heatmap, 0.95)))
    plt.title("Carrier and departure-hour delay rate")
    save(figures / "09_carrier_hour_heatmap.png")

    routes = rate_table(lazy, ["origin", "dest"])
    routes.to_csv(stats_dir / "route_rates.csv", index=False)
    supported_routes = routes.loc[routes["rows"] >= 500]
    plt.scatter(supported_routes["rows"], supported_routes["delay_rate"], alpha=0.35, s=12)
    plt.xscale("log")
    plt.xlabel("Route support, log scale")
    plt.ylabel("Delay rate")
    plt.title("Route support and delay prevalence")
    save(figures / "10_route_support.png")

    elevation = (
        lazy.with_columns(
            pl.when(pl.col("origin_elevation") < 100)
            .then(pl.lit("<100 ft"))
            .when(pl.col("origin_elevation") < 1_000)
            .then(pl.lit("100-999 ft"))
            .when(pl.col("origin_elevation") < 3_000)
            .then(pl.lit("1000-2999 ft"))
            .otherwise(pl.lit("3000+ ft"))
            .alias("elevation_bin")
        )
    )
    elevation_rate = rate_table(elevation, ["elevation_bin"])
    sns.barplot(data=elevation_rate, x="elevation_bin", y="delay_rate", color="#77933c")
    plt.xticks(rotation=20)
    plt.title("Delay rate by origin elevation")
    save(figures / "11_origin_elevation.png")

    # 12. Temporal drift (PSI, JSD, TVD, unseen rates) comparing 2023 reference against later periods
    dev_df = (
        lazy.select(
            "flight_date",
            "dep_del15",
            "reporting_airline",
            "origin",
            "dest",
            "distance",
            "crs_elapsed_time",
            "origin_elevation",
            "origin_departures_planned_1h",
            "season",
            "weekday",
            "scheduled_departure_local",
        )
        .collect(engine="streaming")
        .with_columns(
            (pl.col("origin") + pl.lit("|") + pl.col("dest")).alias("route"),
            pl.when(pl.col("flight_date") < date(2024, 1, 1))
            .then(pl.lit("2023_ref"))
            .when(pl.col("flight_date") < date(2024, 7, 1))
            .then(pl.lit("2024_h1"))
            .when(pl.col("flight_date") < date(2025, 1, 1))
            .then(pl.lit("2024_h2"))
            .when(pl.col("flight_date") < date(2025, 2, 1))
            .then(pl.lit("2025_01_v1_ref"))
            .otherwise(pl.lit("2025_02_to_08"))
            .alias("period"),
        )
    )
    ref_df = dev_df.filter(pl.col("period") == "2023_ref")
    ref_origins = set(ref_df.get_column("origin").unique().to_list())
    ref_routes = set(ref_df.get_column("route").unique().to_list())

    def _freq_map(frame: pl.DataFrame, col: str) -> dict[str, float]:
        counts = frame.group_by(col).len()
        total = max(1, frame.height)
        return {str(row[col]): float(row["len"]) / total for row in counts.to_dicts()}

    ref_carrier = _freq_map(ref_df, "reporting_airline")
    ref_origin = _freq_map(ref_df, "origin")
    drift_rows: list[dict[str, object]] = []
    for period_name in ["2024_h1", "2024_h2", "2025_01_v1_ref", "2025_02_to_08"]:
        cur = dev_df.filter(pl.col("period") == period_name)
        if cur.height == 0:
            continue
        cur_origins = cur.get_column("origin").to_list()
        cur_routes = cur.get_column("route").to_list()
        drift_rows.append(
            {
                "period": period_name,
                "rows": cur.height,
                "delay_rate": float(cur.get_column("dep_del15").mean()),
                "psi_distance": population_stability_index(
                    ref_df.get_column("distance").to_numpy(),
                    cur.get_column("distance").to_numpy(),
                ),
                "psi_elapsed": population_stability_index(
                    ref_df.get_column("crs_elapsed_time").to_numpy(),
                    cur.get_column("crs_elapsed_time").to_numpy(),
                ),
                "psi_origin_load_1h": population_stability_index(
                    ref_df.get_column("origin_departures_planned_1h").to_numpy(),
                    cur.get_column("origin_departures_planned_1h").to_numpy(),
                ),
                "jsd_carrier": jensen_shannon_divergence(
                    ref_carrier, _freq_map(cur, "reporting_airline")
                ),
                "tvd_origin": total_variation_distance(ref_origin, _freq_map(cur, "origin")),
                "unseen_origin_rate": float(
                    np.mean([o not in ref_origins for o in cur_origins])
                ),
                "unseen_route_rate": float(
                    np.mean([r not in ref_routes for r in cur_routes])
                ),
            }
        )
    drift_frame = pd.DataFrame(drift_rows)
    drift_frame.to_csv(stats_dir / "temporal_drift.csv", index=False)
    if not drift_frame.empty:
        _fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(drift_frame["period"], drift_frame["psi_distance"], marker="o", label="PSI distance")
        ax[0].plot(drift_frame["period"], drift_frame["psi_origin_load_1h"], marker="s", label="PSI origin load 1h")
        ax[0].axhline(0.1, linestyle="--", color="orange", alpha=0.6)
        ax[0].axhline(0.25, linestyle="--", color="red", alpha=0.6)
        ax[0].set_title("Numeric PSI vs 2023 baseline")
        ax[0].tick_params(axis="x", rotation=25)
        ax[0].legend()
        ax[1].plot(drift_frame["period"], drift_frame["jsd_carrier"], marker="o", label="JSD carrier")
        ax[1].plot(drift_frame["period"], drift_frame["tvd_origin"], marker="s", label="TVD origin")
        ax[1].plot(drift_frame["period"], drift_frame["unseen_route_rate"], marker="^", label="Unseen route rate")
        ax[1].set_title("Categorical & route shift vs 2023 baseline")
        ax[1].tick_params(axis="x", rotation=25)
        ax[1].legend()
        save(figures / "12_temporal_drift.png")

    # 13. Holiday proximity
    sched_series = pd.Series(dev_df.get_column("scheduled_departure_local").to_list())
    holiday_dist = _nearest_holiday_distance(sched_series)
    holiday_df = pd.DataFrame(
        {
            "dep_del15": dev_df.get_column("dep_del15").to_numpy(),
            "holiday_dist": holiday_dist,
        }
    )
    holiday_df["holiday_bin"] = pd.cut(
        holiday_df["holiday_dist"],
        bins=[-0.1, 0.5, 3.5, 7.5, 14.5, 400.0],
        labels=["0d (holiday)", "1-3d", "4-7d", "8-14d", "15+d"],
    )
    holiday_stats = (
        holiday_df.groupby("holiday_bin", observed=True)["dep_del15"]
        .agg(rows="count", positive="sum", delay_rate="mean")
        .reset_index()
    )
    h_intervals = [
        wilson_interval(int(r.positive), int(r.rows)) for r in holiday_stats.itertuples()
    ]
    holiday_stats["ci_low"] = [v[0] for v in h_intervals]
    holiday_stats["ci_high"] = [v[1] for v in h_intervals]
    holiday_stats.to_csv(stats_dir / "holiday_rates.csv", index=False)
    plt.figure(figsize=(7.5, 4.4))
    sns.barplot(data=holiday_stats, x="holiday_bin", y="delay_rate", color="#5a7d9a")
    plt.title("Delay rate by distance to nearest U.S. federal holiday")
    save(figures / "13_holiday_proximity.png")

    # 14. Weekend x Season
    weekend_season = (
        dev_df.with_columns((pl.col("weekday") >= 6).alias("is_weekend"))
        .group_by(["season", "is_weekend"])
        .agg(pl.len().alias("rows"), pl.col("dep_del15").mean().alias("delay_rate"))
        .sort(["season", "is_weekend"])
        .to_pandas()
    )
    weekend_season.to_csv(stats_dir / "weekend_season_rates.csv", index=False)
    plt.figure(figsize=(7.5, 4.4))
    sns.barplot(data=weekend_season, x="season", y="delay_rate", hue="is_weekend")
    plt.title("Delay rate by season and weekend indicator")
    save(figures / "14_weekend_season.png")

    # Numeric correlation & Mutual Information on deterministic 50k sample
    sample_mi = dev_df.sample(n=min(50_000, dev_df.height), seed=42, shuffle=True).to_pandas()
    sample_mi["dep_hour"] = pd.to_datetime(sample_mi["scheduled_departure_local"]).dt.hour
    num_cols = [
        "distance",
        "crs_elapsed_time",
        "origin_elevation",
        "origin_departures_planned_1h",
        "dep_hour",
    ]
    x_mi = sample_mi[num_cols].fillna(sample_mi[num_cols].median())
    y_mi = sample_mi["dep_del15"].to_numpy(dtype=int)
    mi_values = mutual_info_classif(x_mi, y_mi, random_state=42)
    mi_rows = []
    for idx, col in enumerate(num_cols):
        corr = float(np.corrcoef(x_mi[col].to_numpy(dtype=float), y_mi)[0, 1])
        mi_rows.append({"feature": col, "pearson_r": corr, "mutual_info": float(mi_values[idx])})
    pd.DataFrame(mi_rows).sort_values("mutual_info", ascending=False).to_csv(
        stats_dir / "numeric_mi_corr.csv", index=False
    )

    # Fold prevalence baselines (strictly excluding final_holdout)
    if args.splits.exists():
        splits_cfg = yaml.safe_load(args.splits.read_text(encoding="utf-8"))
        fold_base_rows = []
        for fold in splits_cfg.get("rolling_folds", []):
            t_start = date.fromisoformat(str(fold["train_start"]))
            t_end = date.fromisoformat(str(fold["train_end_exclusive"]))
            v_start = date.fromisoformat(str(fold["validation_start"]))
            v_end = date.fromisoformat(str(fold["validation_end_exclusive"]))
            if v_end > args.end_exclusive:
                continue
            tr = dev_df.filter((pl.col("flight_date") >= t_start) & (pl.col("flight_date") < t_end))
            va = dev_df.filter((pl.col("flight_date") >= v_start) & (pl.col("flight_date") < v_end))
            fold_base_rows.append(
                {
                    "window": fold["name"],
                    "train_rows": tr.height,
                    "train_positive_rate": float(tr.get_column("dep_del15").mean()) if tr.height else np.nan,
                    "eval_rows": va.height,
                    "eval_positive_rate_pr_auc_baseline": float(va.get_column("dep_del15").mean()) if va.height else np.nan,
                }
            )
        for win_name in ["historical_reference", "calibration", "threshold_selection", "pre_holdout_validation"]:
            win = splits_cfg.get(win_name)
            if not win:
                continue
            w_start = date.fromisoformat(str(win["start"]))
            w_end = date.fromisoformat(str(win["end_exclusive"]))
            if w_end > args.end_exclusive:
                continue
            sub = dev_df.filter((pl.col("flight_date") >= w_start) & (pl.col("flight_date") < w_end))
            fold_base_rows.append(
                {
                    "window": win_name,
                    "train_rows": 0,
                    "train_positive_rate": np.nan,
                    "eval_rows": sub.height,
                    "eval_positive_rate_pr_auc_baseline": float(sub.get_column("dep_del15").mean()) if sub.height else np.nan,
                }
            )
        pd.DataFrame(fold_base_rows).to_csv(stats_dir / "fold_baselines.csv", index=False)

    counts = {
        "rows": int(monthly["rows"].sum()),
        "months": len(monthly),
        "start_month": str(monthly["year_month"].min()),
        "end_month": str(monthly["year_month"].max()),
        "positive_rate": float(
            monthly["positive"].sum() / monthly["rows"].sum()
        ),
        "final_holdout_read": False,
        "schedule_load_status": "retrospective_bts_proxy_not_production_eligible",
    }
    (args.output_root / "eda_stats.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )
    summary = f"""# V2 EDA Summary

The development EDA covers {counts['rows']:,} sampled eligible flights across
{counts['months']} continuous months from {counts['start_month']} through
{counts['end_month']}. The final September 2025 holdout was not read.

The global development positive rate is {counts['positive_rate']:.3%}. Every
categorical rate chart includes support filtering or a companion support table.
Monthly confidence bands use Wilson intervals. The schedule-load plots are
retrospective BTS proxies and are not production-eligible without a versioned
schedule snapshot published before T-24h.

The plots are hypothesis generators. Feature promotion requires rolling-fold
ablation; no plot alone authorizes a V2 performance claim.
"""
    (args.output_root / "eda_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

