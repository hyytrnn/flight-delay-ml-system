from __future__ import annotations

from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl
import pytest

from flight_delay.data.load_bts import normalize_columns, read_bts_file
from flight_delay.data.validate import DataContractError, validate_input_contract


def test_bts_aliases_are_canonicalized() -> None:
    frame = pl.DataFrame(
        {
            "FlightDate": ["2025-01-01"],
            "Reporting_Airline": ["AA"],
            "Flight_Number_Reporting_Airline": ["1"],
            "Origin": ["JFK"],
            "Dest": ["LAX"],
            "CRSDepTime": [830],
            "CRSArrTime": [1130],
            "CRSElapsedTime": [360],
            "Distance": [2475],
            "DepDel15": [1],
            "Cancelled": [0],
            "Diverted": [0],
        }
    )
    normalized = normalize_columns(frame)
    assert "flight_date" in normalized.columns
    assert "flight_number" in normalized.columns
    assert "dep_del15" in normalized.columns


def test_missing_required_column_fails_closed(valid_flights: pl.DataFrame, airports: pl.DataFrame) -> None:
    with pytest.raises(DataContractError, match="dep_del15"):
        validate_input_contract(valid_flights.drop("dep_del15"), airports)


def test_zip_loader_reads_csv_members(tmp_path) -> None:
    csv_text = (
        "FlightDate,Reporting_Airline,Flight_Number_Reporting_Airline,Origin,Dest,"
        "CRSDepTime,CRSArrTime,CRSElapsedTime,Distance,DepDel15,Cancelled,Diverted\n"
        "2025-01-01,AA,1,JFK,LAX,830,1130,360,2475,1,0,0\n"
    )
    archive_path = tmp_path / "bts.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("month.csv", csv_text)

    frame = read_bts_file(archive_path)

    assert frame.height == 1
    assert "dep_del15" in frame.columns


@pytest.fixture
def airports() -> pl.DataFrame:
    return pl.DataFrame({"iata_code": ["JFK", "LAX"], "timezone": ["America/New_York", "America/Los_Angeles"]})


@pytest.fixture
def valid_flights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "flight_date": ["2025-01-01"],
            "reporting_airline": ["AA"],
            "flight_number": ["1"],
            "origin": ["JFK"],
            "dest": ["LAX"],
            "crs_dep_time": [830],
            "crs_arr_time": [1130],
            "crs_elapsed_time": [360],
            "distance": [2475],
            "dep_del15": [1],
            "cancelled": [0],
            "diverted": [0],
        }
    )

