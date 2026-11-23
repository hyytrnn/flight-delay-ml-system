"""Schema-level contracts for BTS inputs and airport references."""

from __future__ import annotations

import polars as pl

REQUIRED_BTS_COLUMNS = frozenset(
    {
        "flight_date",
        "reporting_airline",
        "flight_number",
        "origin",
        "dest",
        "crs_dep_time",
        "crs_arr_time",
        "crs_elapsed_time",
        "distance",
        "dep_del15",
        "cancelled",
        "diverted",
    }
)

REQUIRED_AIRPORT_COLUMNS = frozenset({"iata_code", "timezone"})


class DataContractError(ValueError):
    """Raised when a dataset cannot enter row-level cleaning safely."""


def require_columns(frame: pl.DataFrame, required: frozenset[str], dataset: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"{dataset} is missing required columns: {missing}")


def validate_input_contract(flights: pl.DataFrame, airports: pl.DataFrame) -> None:
    require_columns(flights, REQUIRED_BTS_COLUMNS, "BTS input")
    require_columns(airports, REQUIRED_AIRPORT_COLUMNS, "airport reference")

    duplicated_airports = (
        airports.group_by(pl.col("iata_code").cast(pl.String).str.to_uppercase())
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicated_airports.height:
        raise DataContractError("airport reference contains duplicate IATA codes")

