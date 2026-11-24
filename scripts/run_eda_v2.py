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

from flight_delay.analysis.stats import wilson_interval


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

