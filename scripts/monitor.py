"""Generate a weekly-style drift report from training and locked-test periods."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from flight_delay.monitoring.report import build_report, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/clean_flights.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/monitoring/weekly_report.json"))
    args = parser.parse_args()
    frame = pl.read_parquet(args.data)
    reference = frame.filter(pl.col("flight_date") < pl.date(2024, 1, 1))
    current = frame.filter(pl.col("flight_date") >= pl.date(2025, 1, 1))
    report = build_report(reference, current)
    write_report(report, args.output)
    print(args.output.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
