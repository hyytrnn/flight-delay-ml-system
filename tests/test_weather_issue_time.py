from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from flight_delay.data.weather_asof import join_weather_asof


def test_future_available_forecast_run_is_rejected_even_if_init_before_t24() -> None:
    sched_dep = datetime(2024, 7, 1, 12, 0, tzinfo=UTC)
    pred_at = sched_dep - timedelta(hours=24)  # 2024-06-30 12:00 UTC
    flights = pd.DataFrame(
        [
            {
                "origin": "JFK",
                "dest": "LAX",
                "scheduled_departure_utc": sched_dep,
                "prediction_at_utc": pred_at,
                "crs_elapsed_time": 360.0,
            }
        ]
    )
    # Run initialized at 2024-06-30 10:00 UTC (<= pred_at),
    # BUT with 6h processing lag it is available at 16:00 UTC (> pred_at): MUST BE REJECTED!
    # Older run initialized 2024-06-29 12:00 UTC (available 18:00 UTC) MUST BE SELECTED!
    weather = pd.DataFrame(
        [
            {
                "airport": "JFK",
                "forecast_valid_time_utc": sched_dep,
                "run_initialization_utc": datetime(2024, 6, 30, 10, 0, tzinfo=UTC),
                "model_availability_time_utc": datetime(2024, 6, 30, 16, 0, tzinfo=UTC),
                "temperature_2m": 99.0,
                "precipitation": 0.0,
                "wind_speed_10m": 0.0,
                "wind_gusts_10m": 0.0,
                "cloud_cover": 0.0,
            },
            {
                "airport": "JFK",
                "forecast_valid_time_utc": sched_dep,
                "run_initialization_utc": datetime(2024, 6, 29, 12, 0, tzinfo=UTC),
                "model_availability_time_utc": datetime(2024, 6, 29, 18, 0, tzinfo=UTC),
                "temperature_2m": 21.5,
                "precipitation": 0.0,
                "wind_speed_10m": 5.0,
                "wind_gusts_10m": 8.0,
                "cloud_cover": 10.0,
            },
        ]
    )
    out = join_weather_asof(flights, weather, role="origin")
    assert out.loc[0, "origin_weather_missing"] == 0
    assert out.loc[0, "origin_temperature_2m"] == pytest.approx(21.5)


def test_only_unavailable_runs_yield_missing_flag() -> None:
    sched_dep = datetime(2024, 7, 1, 12, 0, tzinfo=UTC)
    pred_at = sched_dep - timedelta(hours=24)
    flights = pd.DataFrame(
        [
            {
                "origin": "JFK",
                "dest": "LAX",
                "scheduled_departure_utc": sched_dep,
                "prediction_at_utc": pred_at,
                "crs_elapsed_time": 360.0,
            }
        ]
    )
    weather = pd.DataFrame(
        [
            {
                "airport": "JFK",
                "forecast_valid_time_utc": sched_dep,
                "run_initialization_utc": datetime(2024, 6, 30, 10, 0, tzinfo=UTC),
                "model_availability_time_utc": None,
                "temperature_2m": 30.0,
                "precipitation": 0.0,
                "wind_speed_10m": 5.0,
                "wind_gusts_10m": 8.0,
                "cloud_cover": 10.0,
            }
        ]
    )
    out = join_weather_asof(flights, weather, role="origin", default_availability_lag_hours=6)
    assert out.loc[0, "origin_weather_missing"] == 1
