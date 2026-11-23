from __future__ import annotations

import pytest

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

