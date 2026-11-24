from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from flight_delay.features.build import build_feature_frame
from flight_delay.features.contracts import (
    BASELINE_FEATURE_COLUMNS,
    assert_feature_columns_are_eligible,
)


def test_baseline_feature_allowlist_has_no_leakage() -> None:
    assert_feature_columns_are_eligible(BASELINE_FEATURE_COLUMNS)


@pytest.mark.parametrize(
    "column",
    ["dep_del15", "dep_delay", "dep_time", "taxi_out", "arr_del15", "weather_delay"],
)
def test_known_post_departure_columns_are_rejected(column: str) -> None:
    with pytest.raises(ValueError, match="Leakage columns"):
        assert_feature_columns_are_eligible(["origin", column])


def test_future_outcomes_cannot_change_feature_vector() -> None:
    common = {
        "reporting_airline": "AA",
        "origin": "JFK",
        "dest": "LAX",
        "distance": 2475.0,
        "crs_elapsed_time": 360.0,
        "scheduled_departure_local": pd.Timestamp("2025-01-15T08:30:00"),
    }
    before = pd.DataFrame([{**common, "dep_delay": 0.0, "taxi_out": 10.0}])
    mutated_future = pd.DataFrame([{**common, "dep_delay": 180.0, "taxi_out": 75.0}])
    assert_frame_equal(build_feature_frame(before), build_feature_frame(mutated_future))

