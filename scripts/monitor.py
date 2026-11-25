"""Generate data, score, and delayed-outcome monitoring reports."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from flight_delay.monitoring.report import build_report, write_report


def _load_frame(data_path: Path, data_root: Path | None) -> pl.DataFrame:
    if data_root is not None:
        pattern = str(data_root / "year=*" / "month=*" / "flights.parquet")
        return pl.scan_parquet(pattern).collect(engine="streaming")
    return pl.read_parquet(data_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/clean_flights.parquet"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--current-data-root", type=Path, default=None)
    parser.add_argument("--reference-end", type=date.fromisoformat, default=date(2025, 2, 1))
    parser.add_argument("--current-start", type=date.fromisoformat, default=date(2025, 10, 1))
    parser.add_argument("--current-end", type=date.fromisoformat, default=None)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    parser.add_argument("--window-label", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/monitoring/v2_1_latest.json"))
    args = parser.parse_args()

    ref_frame = _load_frame(args.data, args.data_root)
    reference = ref_frame.filter(pl.col("flight_date") < args.reference_end)

    cur_source = (
        _load_frame(args.data, args.current_data_root)
        if args.current_data_root is not None
        else ref_frame
    )
    current_filter = pl.col("flight_date") >= args.current_start
    if args.current_end is not None:
        current_filter = current_filter & (pl.col("flight_date") < args.current_end)
    current = cur_source.filter(current_filter)

    report = build_report(
        reference,
        current,
        bundle_path=args.bundle,
        config_path=args.config if args.config.exists() else None,
        window_label=args.window_label,
    )
    write_report(report, args.output)
    print(args.output.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
