from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd

from flight_delay.features.build import scheduled_local_from_aware
from flight_delay.features.v2 import build_v2_feature_frame
from flight_delay.models.predictor import ModelBundle


def test_post_outcome_mutations_do_not_alter_v2_features() -> None:
    base = pd.DataFrame(
        [
            {
                "reporting_airline": "AA",
                "origin": "JFK",
                "dest": "LAX",
                "scheduled_departure_local": pd.Timestamp("2024-07-04T08:30:00"),
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
                "dep_delay": 0.0,
                "dep_time": 830,
                "arr_delay": -5.0,
                "taxi_out": 15.0,
                "cancelled": 0,
                "diverted": 0,
                "carrier_delay": 0.0,
            }
        ]
    )
    mutated = base.copy()
    mutated["dep_delay"] = 240.0
    mutated["dep_time"] = 1230
    mutated["arr_delay"] = 250.0
    mutated["taxi_out"] = 95.0
    mutated["cancelled"] = 1
    mutated["diverted"] = 1
    mutated["carrier_delay"] = 200.0

    for group in ["F0", "F1", "F2", "F3"]:
        f_base = build_v2_feature_frame(base, group)  # type: ignore[arg-type]
        f_mut = build_v2_feature_frame(mutated, group)  # type: ignore[arg-type]
        pd.testing.assert_frame_equal(f_base, f_mut)


def test_v2_train_serving_parity_and_unseen_route_timezone() -> None:
    instant_utc = datetime(2024, 11, 3, 6, 30, tzinfo=UTC)  # DST fallback day in US Eastern
    instant_plus2 = instant_utc.astimezone(timezone(timedelta(hours=2)))
    local_a = scheduled_local_from_aware(instant_utc, "America/New_York")
    local_b = scheduled_local_from_aware(instant_plus2, "America/New_York")
    assert local_a == local_b

    row = pd.DataFrame(
        [
            {
                "reporting_airline": "ZZ",
                "origin": "ORD",
                "dest": "SFO",
                "scheduled_departure_local": local_a,
                "distance": 1000.0,
                "crs_elapsed_time": 150.0,
                "origin_latitude": 41.97,
                "origin_longitude": -87.91,
                "origin_elevation": 672.0,
                "origin_subdivision": "Illinois",
                "dest_latitude": 37.62,
                "dest_longitude": -122.38,
                "dest_elevation": 13.0,
                "dest_subdivision": "California",
            }
        ]
    )
    expected = build_v2_feature_frame(row, "F3")
    bundle = ModelBundle(
        model=None,
        calibrator=None,
        threshold=0.5,
        metadata={"require_origin_timezone": True},
        route_lookup={},
        defaults={"distance": 1000.0, "crs_elapsed_time": 150.0},
        airport_timezones={"ORD": "America/Chicago", "SFO": "America/Los_Angeles"},
        airport_lookup={
            "ORD": {
                "timezone": "America/New_York",
                "latitude": 41.97,
                "longitude": -87.91,
                "elevation": 672.0,
                "subdivision": "Illinois",
            },
            "SFO": {
                "timezone": "America/Los_Angeles",
                "latitude": 37.62,
                "longitude": -122.38,
                "elevation": 13.0,
                "subdivision": "California",
            },
        },
        feature_group="F3",
        calibration_method="none",
    )
    actual = bundle._build_request_features(
        reporting_airline="ZZ",
        origin="ORD",
        dest="SFO",
        scheduled_local=local_a,
        values={"distance": 1000.0, "crs_elapsed_time": 150.0},
    )
    pd.testing.assert_frame_equal(expected, actual)
