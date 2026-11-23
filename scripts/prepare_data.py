"""CLI for BTS ingestion, manifesting, cleaning and quarantine outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from glob import glob
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flight_delay.data.clean import clean_flights
from flight_delay.data.load_bts import read_bts_files

MANIFEST_FIELDS = [
    "filename",
    "source_url",
    "flight_year",
    "flight_month",
    "downloaded_at_utc",
    "bytes",
    "sha256",
    "selected_fields",
    "license_notes",
    "run_kind",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="CSV/ZIP paths or glob patterns")
    parser.add_argument("--airports", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quarantine", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-kind", choices=["real", "synthetic-smoke"], default="real")
    parser.add_argument("--source-url", default="local-file")
    parser.add_argument("--license-notes", default="Verify BTS terms and source metadata")
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(value) for value in glob(pattern)]
        if matches:
            paths.extend(matches)
        elif Path(pattern).is_file():
            paths.append(Path(pattern))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError(f"No input files matched: {patterns}")
    return unique


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: Path,
    inputs: list[Path],
    columns: list[str],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    append = args.run_kind == "real" and path.exists() and path.stat().st_size > 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if not append:
            writer.writeheader()
        for source in inputs:
            writer.writerow(
                {
                    "filename": source.name,
                    "source_url": args.source_url,
                    "flight_year": "",
                    "flight_month": "",
                    "downloaded_at_utc": datetime.fromtimestamp(
                        source.stat().st_mtime, tz=UTC
                    ).isoformat(),
                    "bytes": source.stat().st_size,
                    "sha256": sha256(source),
                    "selected_fields": ";".join(columns),
                    "license_notes": args.license_notes,
                    "run_kind": args.run_kind,
                }
            )


def main() -> int:
    args = parse_args()
    inputs = expand_inputs(args.input)
    flights = read_bts_files(inputs)
    airports = pl.read_csv(args.airports)
    result = clean_flights(flights, airports, run_kind=args.run_kind)

    for path in (args.output, args.quarantine, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    result.clean.write_parquet(args.output)
    result.quarantine.write_parquet(args.quarantine)
    args.report.write_text(json.dumps(result.report, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(args.manifest, inputs, flights.columns, args)

    print(json.dumps(result.report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

