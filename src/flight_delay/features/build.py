"""Shared leakage-safe feature construction for training and inference."""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import polars as pl

from flight_delay.features.contracts import assert_feature_columns_are_eligible

CATEGORICAL_FEATURES = ["reporting_airline", "origin", "dest"]
NUMERIC_FEATURES = [
    "distance",
    "crs_elapsed_time",
    "dep_hour_sin",
    "dep_hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def scheduled_local_from_aware(
    scheduled_departure: datetime, origin_timezone: str
) -> pd.Timestamp:
    """Convert one instant to origin wall time before calendar feature extraction."""

    if scheduled_departure.tzinfo is None:
        raise ValueError("scheduled_departure must be timezone-aware")
    try:
        origin_zone = ZoneInfo(origin_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown origin timezone: {origin_timezone}") from exc
    local = scheduled_departure.astimezone(origin_zone).replace(tzinfo=None)
    return pd.Timestamp(local)


def _add_calendar_features_pandas(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    scheduled = pd.to_datetime(result["scheduled_departure_local"], errors="coerce")
    hour = scheduled.dt.hour.astype(float)
    weekday = scheduled.dt.dayofweek.astype(float)
    month = scheduled.dt.month.astype(float)
    result["dep_hour_sin"] = (2.0 * math.pi * hour / 24.0).map(math.sin)
    result["dep_hour_cos"] = (2.0 * math.pi * hour / 24.0).map(math.cos)
    result["weekday_sin"] = (2.0 * math.pi * weekday / 7.0).map(math.sin)
    result["weekday_cos"] = (2.0 * math.pi * weekday / 7.0).map(math.cos)
    result["month_sin"] = (2.0 * math.pi * (month - 1.0) / 12.0).map(math.sin)
    result["month_cos"] = (2.0 * math.pi * (month - 1.0) / 12.0).map(math.cos)
    result["is_weekend"] = (weekday >= 5).astype(int)
    return result


def build_feature_frame(frame: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    """Build the exact feature table consumed by every model and API request."""

    pandas_frame = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame.copy()
    featured = _add_calendar_features_pandas(pandas_frame)
    assert_feature_columns_are_eligible(MODEL_FEATURES)
    return featured[MODEL_FEATURES]


def build_single_feature(
    *,
    reporting_airline: str,
    origin: str,
    dest: str,
    scheduled_departure_local: pd.Timestamp,
    distance: float,
    crs_elapsed_time: float,
) -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "reporting_airline": reporting_airline.upper(),
                "origin": origin.upper(),
                "dest": dest.upper(),
                "scheduled_departure_local": scheduled_departure_local,
                "distance": distance,
                "crs_elapsed_time": crs_elapsed_time,
            }
        ]
    )
    return build_feature_frame(raw)

