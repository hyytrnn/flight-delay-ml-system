from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from flight_delay.features.v2 import (
    P2_PROXY_NUMERIC,
    add_schedule_load_proxy,
    build_v2_feature_frame,
    v2_feature_columns,
)


def v2_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "flight_key": ["a", "b", "c"],
            "flight_date": [date(2025, 1, 1)] * 3,
            "scheduled_departure_local": [
                datetime(2025, 1, 1, 8, 0),
                datetime(2025, 1, 1, 8, 30),
                datetime(2025, 1, 1, 10, 0),
            ],
            "reporting_airline": ["AA", "AA", "UA"],
            "origin": ["JFK", "JFK", "JFK"],
            "dest": ["LAX", "LAX", "ORD"],
            "distance": [2475.0, 2475.0, 740.0],
            "crs_elapsed_time": [360.0, 360.0, 130.0],
            "origin_latitude": [40.64] * 3,
            "origin_longitude": [-73.78] * 3,
            "origin_elevation": [13.0] * 3,
            "dest_latitude": [33.94, 33.94, 41.97],
            "dest_longitude": [-118.41, -118.41, -87.91],
            "dest_elevation": [125.0, 125.0, 672.0],
            "origin_subdivision": ["New York"] * 3,
            "dest_subdivision": ["California", "California", "Illinois"],
        }
    )


def test_schedule_load_proxy_uses_schedule_rows_not_target() -> None:
    result = add_schedule_load_proxy(v2_rows())
    assert result.filter(pl.col("flight_key") == "a").item(
        0, "origin_departures_planned_1h"
    ) == 2.0
    assert result.filter(pl.col("flight_key") == "a").item(
        0, "route_planned_load_24h"
    ) == 2.0
    assert "dep_del15" not in result.columns


def test_v2_f3_features_include_calendar_interactions_and_geography() -> None:
    features = build_v2_feature_frame(v2_rows(), "F3")
    numeric, categorical = v2_feature_columns("F3")
    assert features.columns.tolist() == categorical + numeric
    assert features.loc[0, "carrier_route"] == "AA|JFK|LAX"
    assert features.loc[0, "haversine_distance_km"] == pytest.approx(3974, rel=0.02)
    assert features.loc[0, "is_us_federal_holiday"] == 1


def test_proxy_group_requires_precomputed_load_columns() -> None:
    with pytest.raises(KeyError):
        build_v2_feature_frame(v2_rows(), "P2_PROXY")


def test_proxy_column_contract_is_explicit() -> None:
    numeric, _ = v2_feature_columns("P2_PROXY")
    assert set(P2_PROXY_NUMERIC).issubset(numeric)
