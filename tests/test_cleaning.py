from __future__ import annotations

from datetime import datetime

import polars as pl

from flight_delay.data.clean import clean_flights


def airports() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "iata_code": ["JFK", "LAX"],
            "timezone": ["America/New_York", "America/Los_Angeles"],
        }
    )


def rows(**overrides: object) -> pl.DataFrame:
    values: dict[str, list[object]] = {
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
    for key, value in overrides.items():
        values[key] = [value]
    return pl.DataFrame(values)


def test_valid_row_builds_t24_timestamp() -> None:
    result = clean_flights(rows(), airports(), "synthetic-smoke")
    assert result.clean.height == 1
    scheduled = result.clean.item(0, "scheduled_departure_utc")
    prediction = result.clean.item(0, "prediction_at_utc")
    assert (scheduled - prediction).total_seconds() == 24 * 3600


def test_2400_rolls_to_next_day_midnight() -> None:
    result = clean_flights(rows(crs_dep_time=2400), airports(), "synthetic-smoke")
    local = result.clean.item(0, "scheduled_departure_local")
    assert local == datetime(2025, 1, 2, 0, 0)


def test_cancelled_flight_is_quarantined_not_labeled_on_time() -> None:
    result = clean_flights(rows(cancelled=1, dep_del15=None), airports(), "synthetic-smoke")
    assert result.clean.height == 0
    reason = result.quarantine.item(0, "quarantine_reason")
    assert "cancelled_excluded" in reason
    assert "invalid_or_missing_label" in reason


def test_unknown_airport_is_quarantined() -> None:
    result = clean_flights(rows(origin="XXX"), airports(), "synthetic-smoke")
    assert result.clean.height == 0
    assert "unknown_origin_airport" in result.quarantine.item(0, "quarantine_reason")


def test_unknown_destination_is_quarantined() -> None:
    result = clean_flights(rows(dest="XXX"), airports(), "synthetic-smoke")
    assert result.clean.height == 0
    assert "unknown_destination_airport" in result.quarantine.item(0, "quarantine_reason")


def test_invalid_hhmm_is_quarantined_explicitly() -> None:
    result = clean_flights(rows(crs_dep_time=1260), airports(), "synthetic-smoke")
    assert result.clean.height == 0
    assert "invalid_crs_dep_time" in result.quarantine.item(0, "quarantine_reason")


def test_nonexistent_dst_time_is_quarantined() -> None:
    result = clean_flights(
        rows(flight_date="2025-03-09", crs_dep_time=230), airports(), "synthetic-smoke"
    )
    assert result.clean.height == 0
    assert "nonexistent_or_unresolved_local_time" in result.quarantine.item(
        0, "quarantine_reason"
    )


def test_ambiguous_dst_time_uses_earliest_fold_and_is_flagged() -> None:
    result = clean_flights(
        rows(flight_date="2025-11-02", crs_dep_time=130), airports(), "synthetic-smoke"
    )
    assert result.clean.height == 1
    assert result.clean.item(0, "ambiguous_time") is True


def test_duplicate_flight_keys_are_quarantined() -> None:
    duplicated = pl.concat([rows(), rows()])
    result = clean_flights(duplicated, airports(), "synthetic-smoke")
    assert result.clean.height == 0
    assert result.report["duplicate_rows"] == 2

