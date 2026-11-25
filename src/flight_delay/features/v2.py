"""V2 feature groups with explicit paper-inspired and as-of boundaries."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from pandas.tseries.holiday import USFederalHolidayCalendar

from flight_delay.features.build import MODEL_FEATURES, build_feature_frame
from flight_delay.features.contracts import assert_feature_columns_are_eligible

FeatureGroup = Literal["F0", "F1", "F2", "F3", "P2_PROXY"]

F1_NUMERIC = [
    "day_of_year_sin",
    "day_of_year_cos",
    "is_us_federal_holiday",
    "days_to_nearest_us_federal_holiday",
]
F1_CATEGORICAL = ["season"]
F2_CATEGORICAL = ["carrier_origin", "carrier_route", "origin_hour", "route_season"]
F3_NUMERIC = [
    "origin_latitude",
    "origin_longitude",
    "origin_elevation",
    "dest_latitude",
    "dest_longitude",
    "dest_elevation",
    "haversine_distance_km",
    "elevation_delta",
]
F3_CATEGORICAL = ["origin_subdivision", "dest_subdivision"]
P2_PROXY_NUMERIC = [
    "origin_departures_planned_1h",
    "origin_departures_planned_3h_block",
    "origin_departures_planned_6h_block",
    "origin_carrier_load_3h_block",
    "route_planned_load_24h",
]


def _nearest_holiday_distance(dates: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(dates, errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return np.full(len(parsed), np.nan)
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(
        start=valid.min() - pd.Timedelta(days=370),
        end=valid.max() + pd.Timedelta(days=370),
    ).to_numpy(dtype="datetime64[D]")
    values = parsed.to_numpy(dtype="datetime64[D]")
    positions = np.searchsorted(holidays, values)
    left = holidays[np.clip(positions - 1, 0, len(holidays) - 1)]
    right = holidays[np.clip(positions, 0, len(holidays) - 1)]
    left_distance = np.abs((values - left).astype("timedelta64[D]").astype(float))
    right_distance = np.abs((right - values).astype("timedelta64[D]").astype(float))
    result = np.minimum(left_distance, right_distance)
    result[pd.isna(parsed)] = np.nan
    return result


def _haversine_km(frame: pd.DataFrame) -> pd.Series:
    lat1 = np.radians(pd.to_numeric(frame["origin_latitude"], errors="coerce"))
    lon1 = np.radians(pd.to_numeric(frame["origin_longitude"], errors="coerce"))
    lat2 = np.radians(pd.to_numeric(frame["dest_latitude"], errors="coerce"))
    lon2 = np.radians(pd.to_numeric(frame["dest_longitude"], errors="coerce"))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        delta_lon / 2
    ) ** 2
    return pd.Series(2 * 6371.0088 * np.arcsin(np.sqrt(a)), index=frame.index)


def add_schedule_load_proxy(schedule_population: pl.DataFrame) -> pl.DataFrame:
    """Add retrospective planned-load proxies without using labels or outcomes."""

    required = {
        "flight_key",
        "flight_date",
        "scheduled_departure_local",
        "reporting_airline",
        "origin",
        "dest",
    }
    missing = sorted(required.difference(schedule_population.columns))
    if missing:
        raise ValueError(f"schedule population missing columns: {missing}")
    return (
        schedule_population.with_columns(
            pl.col("scheduled_departure_local").dt.hour().alias("_dep_hour"),
        )
        .with_columns(
            (pl.col("_dep_hour") // 3).alias("_dep_3h_block"),
            (pl.col("_dep_hour") // 6).alias("_dep_6h_block"),
        )
        .with_columns(
            pl.len()
            .over("origin", "flight_date", "_dep_hour")
            .cast(pl.Float64)
            .alias("origin_departures_planned_1h"),
            pl.len()
            .over("origin", "flight_date", "_dep_3h_block")
            .cast(pl.Float64)
            .alias("origin_departures_planned_3h_block"),
            pl.len()
            .over("origin", "flight_date", "_dep_6h_block")
            .cast(pl.Float64)
            .alias("origin_departures_planned_6h_block"),
            pl.len()
            .over("origin", "reporting_airline", "flight_date", "_dep_3h_block")
            .cast(pl.Float64)
            .alias("origin_carrier_load_3h_block"),
            pl.len()
            .over("origin", "dest", "flight_date")
            .cast(pl.Float64)
            .alias("route_planned_load_24h"),
            pl.lit("retrospective_bts_proxy").alias("schedule_load_proxy_status"),
        )
        .drop("_dep_hour", "_dep_3h_block", "_dep_6h_block")
    )


def v2_feature_columns(group: FeatureGroup) -> tuple[list[str], list[str]]:
    numeric = [column for column in MODEL_FEATURES if column not in {"reporting_airline", "origin", "dest"}]
    categorical = ["reporting_airline", "origin", "dest"]
    if group in {"F1", "F2", "F3", "P2_PROXY"}:
        numeric += F1_NUMERIC
        categorical += F1_CATEGORICAL
    if group in {"F2", "F3", "P2_PROXY"}:
        categorical += F2_CATEGORICAL
    if group in {"F3", "P2_PROXY"}:
        numeric += F3_NUMERIC
        categorical += F3_CATEGORICAL
    if group == "P2_PROXY":
        numeric += P2_PROXY_NUMERIC
    assert_feature_columns_are_eligible(numeric + categorical)
    return numeric, categorical


def build_v2_feature_frame(
    frame: pl.DataFrame | pd.DataFrame, group: FeatureGroup
) -> pd.DataFrame:
    pandas_frame = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame.copy()
    result = build_feature_frame(pandas_frame)
    if group == "F0":
        return result

    scheduled = pd.to_datetime(pandas_frame["scheduled_departure_local"], errors="coerce")
    day_of_year = scheduled.dt.dayofyear.astype(float)
    result["day_of_year_sin"] = (2.0 * math.pi * day_of_year / 365.25).map(math.sin)
    result["day_of_year_cos"] = (2.0 * math.pi * day_of_year / 365.25).map(math.cos)
    result["days_to_nearest_us_federal_holiday"] = _nearest_holiday_distance(scheduled)
    result["is_us_federal_holiday"] = (
        result["days_to_nearest_us_federal_holiday"] == 0
    ).astype(int)
    result["season"] = pd.cut(
        scheduled.dt.month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=["winter", "spring", "summer", "autumn", "winter_end"],
        include_lowest=True,
    ).astype("string")
    result.loc[result["season"] == "winter_end", "season"] = "winter"

    if group in {"F2", "F3", "P2_PROXY"}:
        hour = scheduled.dt.hour.astype("Int64").astype("string")
        result["carrier_origin"] = (
            pandas_frame["reporting_airline"].astype("string")
            + "|"
            + pandas_frame["origin"].astype("string")
        )
        result["carrier_route"] = (
            pandas_frame["reporting_airline"].astype("string")
            + "|"
            + pandas_frame["origin"].astype("string")
            + "|"
            + pandas_frame["dest"].astype("string")
        )
        result["origin_hour"] = pandas_frame["origin"].astype("string") + "|" + hour
        result["route_season"] = (
            pandas_frame["origin"].astype("string")
            + "|"
            + pandas_frame["dest"].astype("string")
            + "|"
            + result["season"]
        )

    if group in {"F3", "P2_PROXY"}:
        for column in [
            "origin_latitude",
            "origin_longitude",
            "origin_elevation",
            "dest_latitude",
            "dest_longitude",
            "dest_elevation",
        ]:
            result[column] = pd.to_numeric(pandas_frame[column], errors="coerce")
        result["haversine_distance_km"] = _haversine_km(pandas_frame)
        result["elevation_delta"] = result["dest_elevation"] - result["origin_elevation"]
        result["origin_subdivision"] = pandas_frame["origin_subdivision"].astype("string")
        result["dest_subdivision"] = pandas_frame["dest_subdivision"].astype("string")

    if group == "P2_PROXY":
        for column in P2_PROXY_NUMERIC:
            result[column] = pd.to_numeric(pandas_frame[column], errors="coerce")

    numeric, categorical = v2_feature_columns(group)
    return result[categorical + numeric]

