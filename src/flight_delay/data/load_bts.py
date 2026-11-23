"""Load BTS CSV or ZIP files and normalize source column names."""

from __future__ import annotations

import csv
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import polars as pl

COLUMN_ALIASES = {
    "flightdate": "flight_date",
    "flight_date": "flight_date",
    "reporting_airline": "reporting_airline",
    "uniquecarrier": "reporting_airline",
    "unique_carrier": "reporting_airline",
    "flight_number_reporting_airline": "flight_number",
    "flightnum": "flight_number",
    "flight_num": "flight_number",
    "origin": "origin",
    "dest": "dest",
    "crsdeptime": "crs_dep_time",
    "crs_dep_time": "crs_dep_time",
    "crsarrtime": "crs_arr_time",
    "crs_arr_time": "crs_arr_time",
    "crselapsedtime": "crs_elapsed_time",
    "crs_elapsed_time": "crs_elapsed_time",
    "distance": "distance",
    "depdel15": "dep_del15",
    "dep_del15": "dep_del15",
    "cancelled": "cancelled",
    "diverted": "diverted",
}

BTS_KEEP_SOURCE_COLUMNS = frozenset(
    {
        "FlightDate",
        "Reporting_Airline",
        "UniqueCarrier",
        "Flight_Number_Reporting_Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "Distance",
        "DepDel15",
        "Cancelled",
        "Diverted",
        "DepDelay",
        "DepDelayMinutes",
        "DepTime",
        "ArrTime",
        "ActualElapsedTime",
        "TaxiOut",
        "TaxiIn",
        "WheelsOff",
        "WheelsOn",
        "ArrDel15",
        "ArrDelay",
        "ArrDelayMinutes",
        "CancellationCode",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
    }
)


def snake_case(name: str) -> str:
    """Convert BTS-style column names to stable snake_case."""

    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def canonical_column_name(name: str) -> str:
    normalized = snake_case(name)
    compact = normalized.replace("_", "")
    return COLUMN_ALIASES.get(normalized, COLUMN_ALIASES.get(compact, normalized))


def normalize_columns(frame: pl.DataFrame) -> pl.DataFrame:
    mapping = {column: canonical_column_name(column) for column in frame.columns}
    targets = list(mapping.values())
    duplicates = sorted({name for name in targets if targets.count(name) > 1})
    if duplicates:
        raise ValueError(f"Multiple source columns map to canonical columns: {duplicates}")
    return frame.rename(mapping)


def _selected_columns(source: str | Path | BytesIO) -> list[str]:
    """Read only the header and keep contract plus leakage-audit fields."""

    if isinstance(source, BytesIO):
        position = source.tell()
        source.seek(0)
        header = source.readline().decode("utf-8-sig", errors="replace")
        source.seek(position)
    else:
        with Path(source).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            header = handle.readline()
    names = next(csv.reader([header]))
    selected = [name for name in names if name in BTS_KEEP_SOURCE_COLUMNS]
    if not selected:
        raise ValueError("No recognized BTS columns found in CSV header")
    return selected


def _read_csv(source: str | Path | BytesIO) -> pl.DataFrame:
    return pl.read_csv(
        source,
        columns=_selected_columns(source),
        infer_schema_length=10_000,
        null_values=["", "NA", "N/A", "NULL", "null"],
        ignore_errors=False,
    )


def read_bts_file(path: str | Path) -> pl.DataFrame:
    """Read one CSV or every CSV member in a ZIP archive."""

    source = Path(path)
    if source.suffix.lower() == ".csv":
        return normalize_columns(_read_csv(source))
    if source.suffix.lower() == ".zip":
        frames: list[pl.DataFrame] = []
        with ZipFile(source) as archive:
            members = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
            if not members:
                raise ValueError(f"ZIP archive contains no CSV files: {source}")
            for member in members:
                buffer = BytesIO(archive.read(member))
                frames.append(normalize_columns(_read_csv(buffer)))
        return pl.concat(frames, how="diagonal_relaxed")
    raise ValueError(f"Unsupported input type (expected .csv or .zip): {source}")


def read_bts_files(paths: list[str | Path]) -> pl.DataFrame:
    if not paths:
        raise ValueError("At least one BTS CSV or ZIP input is required")
    return pl.concat([read_bts_file(path) for path in paths], how="diagonal_relaxed")

