"""Generate reproducible EDA figures and data-backed findings from cleaned BTS data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns


def save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/clean_flights.parquet"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures/eda"))
    parser.add_argument("--findings", type=Path, default=Path("reports/eda_findings.md"))
    args = parser.parse_args()

    frame = pl.read_parquet(args.data).with_columns(
        pl.col("flight_date").dt.strftime("%Y-%m").alias("year_month"),
        pl.col("scheduled_departure_local").dt.hour().alias("departure_hour"),
        pl.when(pl.col("distance") <= 500)
        .then(pl.lit("short"))
        .when(pl.col("distance") <= 1_000)
        .then(pl.lit("medium"))
        .when(pl.col("distance") <= 2_000)
        .then(pl.lit("long"))
        .otherwise(pl.lit("very_long"))
        .alias("distance_bin"),
    )
    args.figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    label_counts = frame.group_by("dep_del15").len().sort("dep_del15").to_pandas()
    sns.barplot(data=label_counts, x="dep_del15", y="len", color="#0f6b78")
    plt.title("Departure Delay Label Distribution")
    plt.xlabel("DepDel15")
    plt.ylabel("Flights")
    save_current(args.figures / "label_distribution.png")

    monthly = frame.group_by("year_month").agg(pl.mean("dep_del15").alias("delay_rate")).sort("year_month").to_pandas()
    sns.lineplot(data=monthly, x="year_month", y="delay_rate", marker="o", color="#0f6b78")
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Departure Delay Rate")
    save_current(args.figures / "delay_rate_monthly.png")

    hourly = frame.group_by("departure_hour").agg(pl.mean("dep_del15").alias("delay_rate")).sort("departure_hour").to_pandas()
    sns.lineplot(data=hourly, x="departure_hour", y="delay_rate", marker="o", color="#b44b35")
    plt.title("Delay Rate by Scheduled Departure Hour")
    save_current(args.figures / "delay_by_hour.png")

    carriers = (
        frame.group_by("reporting_airline")
        .agg(pl.len().alias("flights"), pl.mean("dep_del15").alias("delay_rate"))
        .sort("flights", descending=True)
        .head(15)
        .to_pandas()
    )
    sns.barplot(data=carriers, x="flights", y="reporting_airline", color="#4477aa")
    plt.title("Top Carrier Support")
    save_current(args.figures / "carrier_support.png")

    distance = frame.group_by("distance_bin").agg(pl.len().alias("flights"), pl.mean("dep_del15").alias("delay_rate")).sort("distance_bin").to_pandas()
    sns.barplot(data=distance, x="distance_bin", y="delay_rate", color="#66a061")
    plt.title("Delay Rate by Distance Bin")
    save_current(args.figures / "delay_by_distance.png")

    split = frame.with_columns(
        pl.when(pl.col("flight_date") < pl.date(2024, 1, 1)).then(pl.lit("train"))
        .when(pl.col("flight_date") < pl.date(2024, 10, 1)).then(pl.lit("validation"))
        .when(pl.col("flight_date") < pl.date(2025, 1, 1)).then(pl.lit("calibration"))
        .otherwise(pl.lit("test")).alias("split")
    )
    split_stats = split.group_by("split").agg(pl.len().alias("flights"), pl.mean("dep_del15").alias("delay_rate")).to_pandas()
    sns.barplot(data=split_stats, x="split", y="delay_rate", color="#8c6bb1")
    plt.title("Target Rate by Chronological Split")
    save_current(args.figures / "temporal_shift.png")

    missing = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_rate": [frame.get_column(column).null_count() / frame.height for column in frame.columns],
        }
    ).sort_values("missing_rate", ascending=False).head(20)
    sns.barplot(data=missing, x="missing_rate", y="column", color="#999999")
    plt.title("Top Missingness Rates After Cleaning")
    save_current(args.figures / "missingness.png")

    overall_rate = float(frame.get_column("dep_del15").mean())
    peak_month = monthly.loc[monthly["delay_rate"].idxmax()]
    low_month = monthly.loc[monthly["delay_rate"].idxmin()]
    peak_hour = hourly.loc[hourly["delay_rate"].idxmax()]
    top_carrier = carriers.iloc[0]
    findings = [
        "# EDA Findings",
        "",
        f"The benchmark contains **{frame.height:,} eligible flights** from {frame['flight_date'].min()} to {frame['flight_date'].max()}.",
        f"The overall DepDel15 positive rate is **{overall_rate:.2%}**.",
        f"The highest sampled monthly delay rate is **{peak_month.delay_rate:.2%}** in {peak_month.year_month}.",
        f"The lowest sampled monthly delay rate is **{low_month.delay_rate:.2%}** in {low_month.year_month}.",
        f"Scheduled hour {int(peak_hour.departure_hour):02d}:00 has the highest observed hourly rate at **{peak_hour.delay_rate:.2%}**.",
        f"The largest carrier group is **{top_carrier.reporting_airline}** with {int(top_carrier.flights):,} sampled flights.",
        f"The dataset contains **{frame.get_column('origin').n_unique()} origins**, **{frame.get_column('dest').n_unique()} destinations**, and **{frame.select(pl.struct(['origin','dest']).n_unique()).item():,} routes**.",
        f"The train split contains {int(split_stats.loc[split_stats['split']=='train','flights'].iloc[0]):,} rows; test contains {int(split_stats.loc[split_stats['split']=='test','flights'].iloc[0]):,} rows.",
        "All figures describe the bounded monthly benchmark sample, not the full BTS population.",
        "Cancelled, diverted, missing-label, invalid-time and unknown-airport records are excluded under the documented policy and remain visible in the data-quality report.",
        "Post-departure fields are retained only for audit and are not included in the model feature allowlist.",
        "The temporal target-rate chart is used as a drift warning; it is not causal evidence.",
        "Weather ablation remains not evaluated because the complete 2023-2025 window does not have uniform issued-run coverage at T-24h.",
    ]
    args.findings.parent.mkdir(parents=True, exist_ok=True)
    args.findings.write_text("\n\n".join(findings) + "\n", encoding="utf-8")
    print("\n".join(findings[:7]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

