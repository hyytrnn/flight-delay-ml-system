from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from flight_delay.data.weather_client import WEATHER_FEATURES, resolve_t24_weather_features
from flight_delay.features.v2 import build_v2_feature_frame, v2_feature_columns
from flight_delay.models.predictor import CategoricalFrameTransformer, ModelBundle


def _make_f5_bundle() -> ModelBundle:
    rows = pd.DataFrame(
        [
            {
                "reporting_airline": "AA",
                "origin": "JFK",
                "dest": "LAX",
                "scheduled_departure_local": pd.Timestamp("2025-01-01T08:00:00"),
                "distance": 2475.0,
                "crs_elapsed_time": 360.0,
                "origin_latitude": 40.64,
                "origin_longitude": -73.78,
                "origin_elevation": 13.0,
                "origin_subdivision": "New York",
                "dest_latitude": 33.94,
                "dest_longitude": -118.41,
                "dest_elevation": 125.0,
                "dest_subdivision": "California",
            },
            {
                "reporting_airline": "UA",
                "origin": "ORD",
                "dest": "JFK",
                "scheduled_departure_local": pd.Timestamp("2025-07-02T18:00:00"),
                "distance": 740.0,
                "crs_elapsed_time": 130.0,
                "origin_latitude": 41.97,
                "origin_longitude": -87.91,
                "origin_elevation": 672.0,
                "origin_subdivision": "Illinois",
                "dest_latitude": 40.64,
                "dest_longitude": -73.78,
                "dest_elevation": 13.0,
                "dest_subdivision": "New York",
            },
        ]
    )
    base = build_v2_feature_frame(rows, "F3")
    for col in WEATHER_FEATURES:
        base[col] = [15.0, 0.2] if not col.endswith("_weather_missing") else [0.0, 1.0]
    numeric, categorical = v2_feature_columns("F3")
    pipe = Pipeline(
        [
            (
                "prep",
                CategoricalFrameTransformer(numeric + WEATHER_FEATURES, categorical),
            ),
            (
                "clf",
                XGBClassifier(
                    n_estimators=4,
                    max_depth=2,
                    enable_categorical=True,
                    tree_method="hist",
                    random_state=42,
                ),
            ),
        ]
    ).fit(base, np.array([0, 1]))
    return ModelBundle(
        model=pipe,
        calibrator=None,
        threshold=0.5,
        metadata={
            "model_version": "experimental-f5-v2.1.0",
            "feature_contract_version": "t24-v2.1-F5",
            "require_origin_timezone": True,
        },
        route_lookup={"JFK|LAX": {"distance": 2475.0, "crs_elapsed_time": 360.0}},
        defaults={"distance": 1000.0, "crs_elapsed_time": 150.0},
        airport_timezones={"JFK": "America/New_York", "LAX": "America/Los_Angeles"},
        airport_lookup={
            "JFK": {
                "timezone": "America/New_York",
                "latitude": 40.64,
                "longitude": -73.78,
                "elevation": 13.0,
                "subdivision": "New York",
            },
            "LAX": {
                "timezone": "America/Los_Angeles",
                "latitude": 33.94,
                "longitude": -118.41,
                "elevation": 125.0,
                "subdivision": "California",
            },
        },
        feature_group="F5",
        calibration_method="none",
    )


def test_f5_live_weather_ok_and_t24_availability_guard() -> None:
    bundle = _make_f5_bundle()
    dep = datetime(2027, 2, 10, 13, 30, tzinfo=UTC)
    pred_at = dep - timedelta(hours=24)

    def ok_fetcher(_lat: float, _lon: float) -> dict[str, object]:
        return {
            "availability_time_utc": (pred_at - timedelta(hours=18)).isoformat(),
            "hourly": {
                "time": ["2027-02-10T14:00:00Z", "2027-02-10T20:00:00Z"],
                "temperature_2m": [12.5, 21.0],
                "precipitation": [0.0, 0.4],
                "wind_speed_10m": [10.0, 8.0],
                "wind_gusts_10m": [16.0, 12.0],
                "cloud_cover": [30.0, 10.0],
            },
        }

    prob, _, route_status, weather_status = bundle.predict_request_with_weather(
        reporting_airline="AA",
        origin="JFK",
        dest="LAX",
        scheduled_departure=dep,
        prediction_at=pred_at,
        fetcher=ok_fetcher,
    )
    assert 0.0 <= prob <= 1.0
    assert route_status == "known"
    assert weather_status == "live_weather_ok"

    # Future-issued payload (availability > prediction_at) must be blocked from leaking
    def leaky_fetcher(_lat: float, _lon: float) -> dict[str, object]:
        return {
            "availability_time_utc": (pred_at + timedelta(hours=2)).isoformat(),
            "hourly": {
                "time": ["2027-02-10T14:00:00Z", "2027-02-10T20:00:00Z"],
                "temperature_2m": [12.5, 21.0],
                "precipitation": [0.0, 0.4],
                "wind_speed_10m": [10.0, 8.0],
                "wind_gusts_10m": [16.0, 12.0],
                "cloud_cover": [30.0, 10.0],
            },
        }

    feats, status_leaky = resolve_t24_weather_features(
        origin_meta={"latitude": 40.64, "longitude": -73.78},
        dest_meta={"latitude": 33.94, "longitude": -118.41},
        scheduled_departure_utc=dep,
        scheduled_arrival_utc=dep + timedelta(hours=6),
        prediction_at_utc=pred_at,
        fetcher=leaky_fetcher,
    )
    assert status_leaky == "degraded_fallback_missing_weather"
    assert feats["origin_weather_missing"] == 1.0
    assert feats["dest_weather_missing"] == 1.0
    assert np.isnan(feats["origin_temperature_2m"])


def test_f5_graceful_fallback_on_timeout_429_and_missing_station() -> None:
    bundle = _make_f5_bundle()
    dep = datetime(2027, 2, 10, 13, 30, tzinfo=UTC)
    pred_at = dep - timedelta(hours=24)

    def timeout_fetcher(_lat: float, _lon: float) -> dict[str, object]:
        raise httpx.ReadTimeout("timed out")

    prob_t, _, _, status_t = bundle.predict_request_with_weather(
        reporting_airline="AA",
        origin="JFK",
        dest="LAX",
        scheduled_departure=dep,
        prediction_at=pred_at,
        fetcher=timeout_fetcher,
    )
    assert 0.0 <= prob_t <= 1.0
    assert status_t == "degraded_fallback_missing_weather"

    def rate_limit_fetcher(_lat: float, _lon: float) -> dict[str, object]:
        req = httpx.Request("GET", "https://api.open-meteo.com/v1/forecast")
        resp = httpx.Response(429, request=req)
        raise httpx.HTTPStatusError("rate limited", request=req, response=resp)

    prob_429, _, _, status_429 = bundle.predict_request_with_weather(
        reporting_airline="AA",
        origin="JFK",
        dest="LAX",
        scheduled_departure=dep,
        prediction_at=pred_at,
        fetcher=rate_limit_fetcher,
    )
    assert 0.0 <= prob_429 <= 1.0
    assert status_429 == "degraded_fallback_missing_weather"

    # Missing station coordinates on unknown destination -> degrades without crashing
    prob_missing, _, route_status, status_missing = bundle.predict_request_with_weather(
        reporting_airline="AA",
        origin="JFK",
        dest="ZZZ",
        scheduled_departure=dep,
        prediction_at=pred_at,
        fetcher=timeout_fetcher,
    )
    assert 0.0 <= prob_missing <= 1.0
    assert route_status == "fallback_global_median"
    assert status_missing == "degraded_fallback_missing_weather"
