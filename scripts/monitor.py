"""Generate a weekly-style drift report from training and locked-test periods."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from flight_delay.monitoring.report import build_report, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/clean_flights.parquet"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--reference-end", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--current-start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--current-end", type=date.fromisoformat, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/monitoring/weekly_report.json"))
    args = parser.parse_args()
    if args.data_root is not None:
        pattern = str(args.data_root / "year=*" / "month=*" / "flights.parquet")
        frame = pl.scan_parquet(pattern).collect(engine="streaming")
    else:
        frame = pl.read_parquet(args.data)
    reference = frame.filter(pl.col("flight_date") < args.reference_end)
    current_filter = pl.col("flight_date") >= args.current_start
    if args.current_end is not None:
        current_filter = current_filter & (pl.col("flight_date") < args.current_end)
    current = frame.filter(current_filter)
    report = build_report(reference, current)
    write_report(report, args.output)
    print(args.output.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
