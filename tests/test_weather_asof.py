from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from flight_delay.data.weather_asof import join_weather_asof


def test_join_weather_asof_matches_origin_and_dest_scheduled_arrival() -> None:
    sched_dep = datetime(2024, 7, 1, 14, 10, tzinfo=UTC)
    pred_at = sched_dep - timedelta(hours=24)
    flights = pd.DataFrame(
        [
            {
                "origin": "JFK",
                "dest": "LAX",
                "scheduled_departure_utc": sched_dep,
                "prediction_at_utc": pred_at,
                "crs_elapsed_time": 360.0,  # scheduled arrival 20:10 UTC
            }
        ]
    )
    weather = pd.DataFrame(
        [
            {
                "airport": "JFK",
                "forecast_valid_time_utc": datetime(2024, 7, 1, 14, 0, tzinfo=UTC),
                "run_initialization_utc": datetime(2024, 6, 29, 14, 0, tzinfo=UTC),
                "model_availability_time_utc": datetime(2024, 6, 29, 20, 0, tzinfo=UTC),
                "temperature_2m": 25.0,
                "precipitation": 0.0,
                "wind_speed_10m": 12.0,
                "wind_gusts_10m": 18.0,
                "cloud_cover": 20.0,
            },
            {
                "airport": "LAX",
                "forecast_valid_time_utc": datetime(2024, 7, 1, 20, 0, tzinfo=UTC),
                "run_initialization_utc": datetime(2024, 6, 29, 20, 0, tzinfo=UTC),
                "model_availability_time_utc": datetime(2024, 6, 30, 2, 0, tzinfo=UTC),
                "temperature_2m": 22.0,
                "precipitation": 0.1,
                "wind_speed_10m": 9.0,
                "wind_gusts_10m": 14.0,
                "cloud_cover": 40.0,
            },
        ]
    )
    out = join_weather_asof(flights, weather, role="origin", tolerance_minutes=30)
    out = join_weather_asof(out, weather, role="dest", tolerance_minutes=30)
    assert out.loc[0, "origin_weather_missing"] == 0
    assert out.loc[0, "dest_weather_missing"] == 0
    assert out.loc[0, "origin_temperature_2m"] == pytest.approx(25.0)
    assert out.loc[0, "dest_temperature_2m"] == pytest.approx(22.0)
    assert out.loc[0, "origin_weather_mismatch_min"] == pytest.approx(10.0)
    assert out.loc[0, "dest_weather_mismatch_min"] == pytest.approx(10.0)
