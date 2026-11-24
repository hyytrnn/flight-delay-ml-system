"""Generate locked-test slice metrics and false-positive/false-negative analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score

from flight_delay.features.build import build_feature_frame
from flight_delay.models.predictor import load_bundle


def slice_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(column, dropna=False, observed=True):
        if len(group) < 200:
            continue
        rows.append(
            {
                "slice": column,
                "value": str(value),
                "rows": len(group),
                "positive_rate": group["target"].mean(),
                "predicted_positive_rate": group["prediction"].mean(),
                "precision": precision_score(
                    group["target"], group["prediction"], zero_division=0
                ),
                "recall": recall_score(group["target"], group["prediction"], zero_division=0),
                "f1": f1_score(group["target"], group["prediction"], zero_division=0),
                "false_negatives": int(
                    ((group["target"] == 1) & (group["prediction"] == 0)).sum()
                ),
                "false_positives": int(
                    ((group["target"] == 0) & (group["prediction"] == 1)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/clean_flights.parquet"))
    parser.add_argument("--model", type=Path, default=Path("models/flight_delay_bundle.joblib"))
    parser.add_argument("--csv", type=Path, default=Path("reports/metrics/error_slices.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/error_analysis.md"))
    args = parser.parse_args()

    test = pl.read_parquet(args.data).filter(pl.col("flight_date") >= pl.date(2025, 1, 1))
    bundle = load_bundle(args.model)
    features = build_feature_frame(test)
    scores = bundle.predict_proba(features)
    output = test.select(
        "reporting_airline", "origin", "dest", "distance", "scheduled_departure_local", "dep_del15"
    ).to_pandas()
    output["target"] = output.pop("dep_del15").astype(int)
    output["score"] = scores
    output["prediction"] = (scores >= bundle.threshold).astype(int)
    output["departure_hour"] = pd.to_datetime(output["scheduled_departure_local"]).dt.hour
    output["distance_bin"] = pd.cut(
        output["distance"],
        bins=[0, 500, 1_000, 2_000, np.inf],
        labels=["short", "medium", "long", "very_long"],
    )

    slices = pd.concat(
        [slice_metrics(output, column) for column in ["reporting_airline", "origin", "departure_hour", "distance_bin"]],
        ignore_index=True,
    ).sort_values(["slice", "rows"], ascending=[True, False])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    slices.to_csv(args.csv, index=False)

    worst_recall = slices.loc[slices["rows"] >= 1_000].sort_values("recall").iloc[0]
    most_fn = slices.sort_values("false_negatives", ascending=False).iloc[0]
    most_fp = slices.sort_values("false_positives", ascending=False).iloc[0]
    report = f"""# Locked Test Error Analysis

The analysis covers {len(output):,} flights from the locked January 2025 test period.
It reports correlations and operational slices, not causal explanations.

- Lowest recall among slices with at least 1,000 rows: `{worst_recall['slice']}={worst_recall['value']}` at {worst_recall['recall']:.3f} across {int(worst_recall['rows']):,} flights.
- Largest false-negative count: `{most_fn['slice']}={most_fn['value']}` with {int(most_fn['false_negatives']):,} false negatives.
- Largest false-positive count: `{most_fp['slice']}={most_fp['value']}` with {int(most_fp['false_positives']):,} false positives.
- Unknown routes use global median distance and duration; the API exposes this as `route_status=fallback_global_median`.
- Weather-missing slices are not applicable because v1 is explicitly flight-only.

Detailed carrier, origin, scheduled-hour and distance-bin metrics are in `reports/metrics/error_slices.csv`.
"""
    args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
