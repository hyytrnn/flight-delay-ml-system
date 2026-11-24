"""Prepare partitioned continuous-month BTS data without concatenating raw months."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from flight_delay.data.clean import clean_flights
from flight_delay.data.load_bts import read_bts_file
from flight_delay.features.v2 import P2_PROXY_NUMERIC, add_schedule_load_proxy


def parse_month(value: str) -> tuple[int, int]:
    year_text, month_text = value.split("-", maxsplit=1)
    year, month = int(year_text), int(month_text)
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"invalid month: {value}")
    return year, month


def month_index(value: tuple[int, int]) -> int:
    return value[0] * 12 + value[1] - 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample(frame: pl.DataFrame, limit: int | None, seed: int) -> pl.DataFrame:
    if limit is None or frame.height <= limit:
        return frame
    return frame.sample(n=limit, seed=seed, shuffle=True)


def discover_monthly_files(
    root: Path,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    allow_missing: bool = False,
    excluded_months: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int, Path]]:
    start_index, end_index = month_index(start), month_index(end)
    discovered: list[tuple[int, int, Path]] = []
    seen_months: set[tuple[int, int]] = set()
    duplicates: set[tuple[int, int]] = set()
    for path in root.rglob("*.zip"):
        try:
            year = int(path.parent.parent.name)
            month = int(path.parent.name)
        except ValueError:
            continue
        if not 1 <= month <= 12:
            continue
        ym = (year, month)
        if start_index <= month_index(ym) <= end_index:
            if excluded_months and ym in excluded_months:
                raise ValueError(
                    f"excluded final-holdout month present in range: {year}-{month:02d}"
                )
            if ym in seen_months:
                duplicates.add(ym)
            seen_months.add(ym)
            discovered.append((year, month, path))
    if duplicates:
        formatted = ", ".join(f"{y}-{m:02d}" for y, m in sorted(duplicates))
        raise ValueError(f"duplicate monthly partitions found: {formatted}")
    if not allow_missing:
        expected = {
            (idx // 12, idx % 12 + 1)
            for idx in range(start_index, end_index + 1)
            if not (excluded_months and (idx // 12, idx % 12 + 1) in excluded_months)
        }
        missing = sorted(expected - seen_months)
        if missing:
            formatted = ", ".join(f"{y}-{m:02d}" for y, m in missing)
            raise FileNotFoundError(f"missing monthly BTS ZIP files: {formatted}")
    return sorted(discovered)


def prepare_month(
    *,
    year: int,
    month: int,
    path: Path,
    airports: pl.DataFrame,
    output_root: Path,
    quarantine_root: Path,
    max_rows: int | None,
    seed: int,
) -> dict[str, object]:
    raw = read_bts_file(path)
    result = clean_flights(raw, airports, run_kind="real")
    schedule_population = (
        pl.concat([result.clean, result.quarantine], how="diagonal_relaxed")
        .filter(
            pl.col("scheduled_departure_local").is_not_null()
            & pl.col("origin_timezone").is_not_null()
            & pl.col("dest_timezone").is_not_null()
        )
        .unique(subset=["flight_key"], keep="first")
    )
    load_features = (
        add_schedule_load_proxy(schedule_population)
        .group_by("flight_key")
        .agg(
            *[pl.col(column).first() for column in P2_PROXY_NUMERIC],
            pl.col("schedule_load_proxy_status").first(),
        )
    )
    eligible_with_load = result.clean.join(
        load_features, on="flight_key", how="left", validate="1:1"
    )
    sampled = stable_sample(
        eligible_with_load,
        max_rows,
        seed=seed + year * 100 + month,
    ).with_columns(
        pl.lit(year).alias("source_year"),
        pl.lit(month).alias("source_month"),
    )
    quarantine = stable_sample(
        result.quarantine,
        2_000,
        seed=seed + year * 100 + month + 1,
    )

    partition = output_root / f"year={year}" / f"month={month:02d}"
    quarantine_partition = quarantine_root / f"year={year}" / f"month={month:02d}"
    partition.mkdir(parents=True, exist_ok=True)
    quarantine_partition.mkdir(parents=True, exist_ok=True)
    sampled.write_parquet(partition / "flights.parquet", compression="zstd")
    quarantine.write_parquet(
        quarantine_partition / "quarantine_sample.parquet", compression="zstd"
    )

    raw_positive_rate = float(result.clean.get_column("dep_del15").mean())
    sample_positive_rate = float(sampled.get_column("dep_del15").mean())
    return {
        "year": year,
        "month": month,
        "filename": path.name,
        "source_url": f"https://www.transtats.bts.gov/PREZIP/{path.name}",
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "raw_rows": int(result.report["input_rows"]),
        "eligible_rows": int(result.report["clean_rows"]),
        "sample_rows": sampled.height,
        "quarantine_rows": int(result.report["quarantine_rows"]),
        "eligible_positive_rate": raw_positive_rate,
        "sample_positive_rate": sample_positive_rate,
        "positive_rate_delta": sample_positive_rate - raw_positive_rate,
        "seed": seed + year * 100 + month,
        "schema": sampled.columns,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/raw/bts"))
    parser.add_argument("--airports", type=Path, default=Path("data/external/airports.csv"))
    parser.add_argument("--start-month", type=parse_month, required=True)
    parser.add_argument("--end-month", type=parse_month, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--quarantine-root", type=Path, default=Path("data/interim/v2"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/bts_v2.json"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/data/v2"))
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--max-rows-per-month", type=int, default=100_000)
    sampling.add_argument("--all-rows", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-months", action="store_true")
    parser.add_argument("--allow-holdout", action="store_true")
    args = parser.parse_args()

    if month_index(args.end_month) < month_index(args.start_month):
        parser.error("end month must not be before start month")
    excluded = None if args.allow_holdout else {(2025, 9)}
    if excluded and (
        month_index(args.start_month) <= month_index((2025, 9)) <= month_index(args.end_month)
    ):
        parser.error("2025-09 final holdout is excluded unless --allow-holdout is explicitly set")
    inputs = discover_monthly_files(
        args.input_root,
        args.start_month,
        args.end_month,
        allow_missing=args.allow_missing_months,
        excluded_months=excluded,
    )
    if not inputs:
        raise FileNotFoundError("no monthly BTS ZIP files found for requested range")

    airports = pl.read_csv(args.airports)
    monthly_records: list[dict[str, object]] = []
    max_rows = None if args.all_rows else args.max_rows_per_month

    for year, month, path in inputs:
        record = prepare_month(
            year=year,
            month=month,
            path=path,
            airports=airports,
            output_root=args.output_root,
            quarantine_root=args.quarantine_root,
            max_rows=max_rows,
            seed=args.seed,
        )
        monthly_records.append(record)
        print(
            f"{year}-{month:02d}: raw={record['raw_rows']} "
            f"eligible={record['eligible_rows']} sample={record['sample_rows']}"
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.reports_root.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(monthly_records, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = pl.DataFrame(monthly_records).sort("year", "month")
    report.select(
        "year",
        "month",
        "raw_rows",
        "eligible_rows",
        "sample_rows",
        "quarantine_rows",
    ).write_csv(args.reports_root / "monthly_counts.csv")
    report.select(
        "year",
        "month",
        "eligible_positive_rate",
        "sample_positive_rate",
        "positive_rate_delta",
    ).write_csv(args.reports_root / "monthly_label_rates.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

