"""Leakage-safe BTS cleaning with explicit quarantine reasons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfoNotFoundError

import polars as pl
from polars.exceptions import ComputeError

from flight_delay.data.validate import validate_input_contract


@dataclass(frozen=True)
class CleanResult:
    clean: pl.DataFrame
    quarantine: pl.DataFrame
    report: dict[str, object]


def _normalize_airports(airports: pl.DataFrame) -> pl.DataFrame:
    normalized = airports.with_columns(
        pl.col("iata_code").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("timezone").cast(pl.String).str.strip_chars(),
    )
    available = [
        column
        for column in [
            "iata_code",
            "timezone",
            "latitude",
            "longitude",
            "elevation",
            "country",
            "subdivision",
        ]
        if column in normalized.columns
    ]
    return normalized.select(available)


def _airport_reference(airports: pl.DataFrame, code_column: str, prefix: str) -> pl.DataFrame:
    reference = _normalize_airports(airports)
    return reference.rename(
        {
            column: code_column if column == "iata_code" else f"{prefix}_{column}"
            for column in reference.columns
        }
    )


def _parse_schedule(frame: pl.DataFrame) -> pl.DataFrame:
    prepared = frame.with_columns(
        pl.col("flight_date").cast(pl.String).alias("_flight_date_raw"),
        pl.col("crs_dep_time").cast(pl.String).alias("_crs_dep_time_raw"),
        pl.col("flight_date").cast(pl.String).str.strptime(pl.Date, strict=False),
        pl.col("crs_dep_time").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        pl.col("crs_arr_time").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        pl.col("crs_elapsed_time").cast(pl.Float64, strict=False),
        pl.col("distance").cast(pl.Float64, strict=False),
        pl.col("dep_del15").cast(pl.Float64, strict=False).cast(pl.Int8, strict=False),
        pl.col("cancelled").cast(pl.Float64, strict=False).cast(pl.Int8, strict=False),
        pl.col("diverted").cast(pl.Float64, strict=False).cast(pl.Int8, strict=False),
        pl.col("reporting_airline").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("flight_number").cast(pl.String).str.strip_chars(),
        pl.col("origin").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("dest").cast(pl.String).str.strip_chars().str.to_uppercase(),
    )

    valid_hhmm = (
        ((pl.col("crs_dep_time") >= 0) & (pl.col("crs_dep_time") <= 2359))
        & ((pl.col("crs_dep_time") % 100) < 60)
    ) | (pl.col("crs_dep_time") == 2400)

    return (
        prepared.with_columns(valid_hhmm.fill_null(False).alias("_valid_crs_dep_time"))
        .with_columns(
            pl.when(~pl.col("_valid_crs_dep_time"))
            .then(None)
            .when(pl.col("crs_dep_time") == 2400)
            .then(0)
            .otherwise(pl.col("crs_dep_time") // 100)
            .alias("_dep_hour"),
            pl.when(~pl.col("_valid_crs_dep_time"))
            .then(None)
            .when(pl.col("crs_dep_time") == 2400)
            .then(0)
            .otherwise(pl.col("crs_dep_time") % 100)
            .alias("_dep_minute"),
            (pl.col("crs_dep_time") == 2400).cast(pl.Int32).alias("_day_shift"),
        )
        .with_columns(
            (
                pl.datetime(
                    pl.col("flight_date").dt.year(),
                    pl.col("flight_date").dt.month(),
                    pl.col("flight_date").dt.day(),
                    pl.col("_dep_hour"),
                    pl.col("_dep_minute"),
                )
                + pl.duration(days=pl.col("_day_shift"))
            ).alias("scheduled_departure_local")
        )
    )


def _localize_by_timezone(frame: pl.DataFrame) -> pl.DataFrame:
    """Localize schedule times in native Polars groups and convert them to UTC."""

    timezones = frame.get_column("origin_timezone").drop_nulls().unique().to_list()
    localized: list[pl.DataFrame] = []

    for timezone in sorted(timezones):
        group = frame.filter(pl.col("origin_timezone") == timezone)
        try:
            earliest = pl.col("scheduled_departure_local").dt.replace_time_zone(
                timezone, ambiguous="earliest", non_existent="null"
            )
            latest = pl.col("scheduled_departure_local").dt.replace_time_zone(
                timezone, ambiguous="latest", non_existent="null"
            )
            localized.append(
                group.with_columns(
                    earliest.dt.convert_time_zone("UTC").alias("scheduled_departure_utc"),
                    (earliest != latest).fill_null(False).alias("ambiguous_time"),
                )
            )
        except (ComputeError, ZoneInfoNotFoundError):
            localized.append(
                group.with_columns(
                    pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias(
                        "scheduled_departure_utc"
                    ),
                    pl.lit(False).alias("ambiguous_time"),
                ).with_columns(pl.lit(True).alias("_invalid_timezone"))
            )

    unknown = frame.filter(pl.col("origin_timezone").is_null()).with_columns(
        pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("scheduled_departure_utc"),
        pl.lit(False).alias("ambiguous_time"),
    )
    localized.append(unknown)
    return pl.concat(localized, how="diagonal_relaxed").sort("_row_id")


def clean_flights(flights: pl.DataFrame, airports: pl.DataFrame, run_kind: str) -> CleanResult:
    """Validate, normalize and split BTS rows into clean and quarantine datasets."""

    validate_input_contract(flights, airports)
    if run_kind not in {"real", "synthetic-smoke"}:
        raise ValueError("run_kind must be 'real' or 'synthetic-smoke'")

    frame = _parse_schedule(flights.with_row_index("_row_id"))
    airport_ref = _airport_reference(airports, "origin", "origin")
    frame = frame.join(airport_ref, on="origin", how="left")
    destination_ref = _airport_reference(airports, "dest", "dest")
    frame = frame.join(destination_ref, on="dest", how="left")
    frame = _localize_by_timezone(frame)
    if "_invalid_timezone" not in frame.columns:
        frame = frame.with_columns(pl.lit(False).alias("_invalid_timezone"))
    else:
        frame = frame.with_columns(pl.col("_invalid_timezone").fill_null(False))

    frame = frame.with_columns(
        (pl.col("scheduled_departure_utc") - pl.duration(hours=24)).alias(
            "prediction_at_utc"
        ),
        pl.concat_str(
            [
                pl.col("flight_date").cast(pl.String),
                pl.col("reporting_airline"),
                pl.col("flight_number"),
                pl.col("origin"),
                pl.col("dest"),
                pl.col("crs_dep_time").cast(pl.String),
            ],
            separator="|",
        ).alias("flight_key"),
    ).with_columns((pl.len().over("flight_key") > 1).alias("_duplicate_flight_key"))

    valid_disruption = (
        pl.col("cancelled").is_in([0, 1]).fill_null(False)
        & pl.col("diverted").is_in([0, 1]).fill_null(False)
    )
    eligible_label = (
        (pl.col("cancelled") == 0)
        & (pl.col("diverted") == 0)
        & pl.col("dep_del15").is_in([0, 1])
    )
    reasons = pl.concat_list(
        [
            pl.when(pl.col("flight_date").is_null()).then(pl.lit("invalid_flight_date")),
            pl.when(~pl.col("_valid_crs_dep_time")).then(pl.lit("invalid_crs_dep_time")),
            pl.when(pl.col("origin_timezone").is_null()).then(pl.lit("unknown_origin_airport")),
            pl.when(pl.col("dest_timezone").is_null()).then(
                pl.lit("unknown_destination_airport")
            ),
            pl.when(pl.col("_invalid_timezone")).then(pl.lit("invalid_origin_timezone")),
            pl.when(pl.col("scheduled_departure_utc").is_null()).then(
                pl.lit("nonexistent_or_unresolved_local_time")
            ),
            pl.when(~valid_disruption).then(pl.lit("invalid_disruption_code")),
            pl.when(pl.col("cancelled") == 1).then(pl.lit("cancelled_excluded")),
            pl.when(pl.col("diverted") == 1).then(pl.lit("diverted_excluded")),
            pl.when(~pl.col("dep_del15").is_in([0, 1]).fill_null(False)).then(
                pl.lit("invalid_or_missing_label")
            ),
            pl.when(pl.col("distance").is_null() | (pl.col("distance") <= 0)).then(
                pl.lit("invalid_distance")
            ),
            pl.when(pl.col("crs_elapsed_time").is_null() | (pl.col("crs_elapsed_time") <= 0)).then(
                pl.lit("invalid_crs_elapsed_time")
            ),
            pl.when(pl.col("_duplicate_flight_key")).then(pl.lit("duplicate_flight_key")),
        ]
    ).list.drop_nulls()
    frame = frame.with_columns(reasons.list.join("|").alias("quarantine_reason"))

    clean = frame.filter((pl.col("quarantine_reason") == "") & eligible_label)
    quarantine = frame.filter((pl.col("quarantine_reason") != "") | ~eligible_label)

    reason_counts: dict[str, int] = {}
    for reason_text in quarantine.get_column("quarantine_reason").to_list():
        for reason in filter(None, str(reason_text).split("|")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    report: dict[str, object] = {
        "status": "executed_synthetic_only" if run_kind == "synthetic-smoke" else "executed_real",
        "run_kind": run_kind,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_rows": frame.height,
        "clean_rows": clean.height,
        "quarantine_rows": quarantine.height,
        "label_coverage": clean.height / frame.height if frame.height else 0.0,
        "duplicate_rows": int(frame.get_column("_duplicate_flight_key").sum()),
        "ambiguous_time_rows": int(frame.get_column("ambiguous_time").sum()),
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        "policy": {
            "target": "dep_del15",
            "prediction_horizon_hours": 24,
            "cancelled": "excluded",
            "diverted": "excluded",
            "missing_label": "excluded",
            "crs_dep_time_2400": "next_day_midnight",
            "ambiguous_dst": "earliest_fold_and_flag",
            "nonexistent_dst": "quarantine",
        },
    }

    internal = [column for column in clean.columns if column.startswith("_")]
    return CleanResult(
        clean=clean.drop(internal),
        quarantine=quarantine.drop(internal),
        report=report,
    )

