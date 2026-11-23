"""Prepare a bounded real benchmark dataset from monthly BTS ZIP archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flight_delay.data.clean import clean_flights
from flight_delay.data.load_bts import read_bts_file


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_frame(frame: pl.DataFrame, limit: int, seed: int) -> pl.DataFrame:
    if frame.height <= limit:
        return frame
    return frame.sample(n=limit, seed=seed, shuffle=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/raw/bts"))
    parser.add_argument("--airports", type=Path, default=Path("data/external/airports.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/clean_flights.parquet"))
    parser.add_argument("--quarantine", type=Path, default=Path("data/interim/quarantine_sample.parquet"))
    parser.add_argument("--report", type=Path, default=Path("reports/data_quality.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/download_manifest.json"))
    parser.add_argument("--max-clean-per-file", type=int, default=120_000)
    parser.add_argument("--max-quarantine-per-file", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inputs = sorted(args.input_root.rglob("*.zip"))
    if not inputs:
        raise FileNotFoundError(f"No BTS ZIP files under {args.input_root}")
    airports = pl.read_csv(args.airports)
    clean_parts: list[pl.DataFrame] = []
    quarantine_parts: list[pl.DataFrame] = []
    file_reports: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []

    for index, path in enumerate(inputs):
        flights = read_bts_file(path)
        result = clean_flights(flights, airports, run_kind="real")
        clean_parts.append(sample_frame(result.clean, args.max_clean_per_file, args.seed + index))
        quarantine_parts.append(
            sample_frame(result.quarantine, args.max_quarantine_per_file, args.seed + index)
        )
        report = dict(result.report)
        report["filename"] = path.name
        file_reports.append(report)
        manifest.append(
            {
                "filename": path.name,
                "source_url": f"https://www.transtats.bts.gov/PREZIP/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "selected_fields": flights.columns,
                "license_notes": "U.S. DOT BTS public data; verify current BTS terms",
            }
        )
        print(
            f"{path.name}: input={result.report['input_rows']} clean={result.report['clean_rows']} "
            f"sampled={clean_parts[-1].height} quarantine={result.report['quarantine_rows']}"
        )

    clean = pl.concat(clean_parts, how="diagonal_relaxed").sort("flight_date")
    quarantine = pl.concat(quarantine_parts, how="diagonal_relaxed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.quarantine.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(b"") if False else None
    clean.write_parquet(args.output, compression="zstd")
    quarantine.write_parquet(args.quarantine, compression="zstd")

    total_input = sum(int(item["input_rows"]) for item in file_reports)
    total_clean = sum(int(item["clean_rows"]) for item in file_reports)
    report = {
        "status": "executed_real",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": len(inputs),
        "raw_input_rows": total_input,
        "eligible_rows_before_sampling": total_clean,
        "benchmark_rows_after_sampling": clean.height,
        "quarantine_sample_rows": quarantine.height,
        "label_coverage": total_clean / total_input if total_input else 0.0,
        "file_reports": file_reports,
        "sampling_policy": {
            "max_clean_per_file": args.max_clean_per_file,
            "max_quarantine_per_file": args.max_quarantine_per_file,
            "seed": args.seed,
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "file_reports"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
