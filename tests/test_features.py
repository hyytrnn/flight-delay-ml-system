from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from flight_delay.features.build import (
    MODEL_FEATURES,
    build_single_feature,
    scheduled_local_from_aware,
)


def test_single_feature_uses_scheduled_time_and_expected_schema() -> None:
    result = build_single_feature(
        reporting_airline="aa",
        origin="jfk",
        dest="lax",
        scheduled_departure_local=pd.Timestamp(datetime(2025, 1, 1, 6, 0)),
        distance=2475,
        crs_elapsed_time=360,
    )
    assert result.columns.tolist() == MODEL_FEATURES
    assert result.loc[0, "reporting_airline"] == "AA"
    assert result.loc[0, "dep_hour_sin"] == pytest.approx(1.0)


def test_same_instant_with_different_offsets_has_identical_origin_local_time() -> None:
    instant_utc = datetime(2027, 2, 10, 13, 30, tzinfo=UTC)
    same_instant_plus_five = instant_utc.astimezone(timezone(timedelta(hours=5)))
    expected = scheduled_local_from_aware(instant_utc, "America/New_York")
    observed = scheduled_local_from_aware(
        same_instant_plus_five, "America/New_York"
    )
    assert observed == expected
    assert expected == pd.Timestamp("2027-02-10T08:30:00")


def test_origin_timezone_fails_closed_when_unknown() -> None:
    with pytest.raises(ValueError, match="unknown origin timezone"):
        scheduled_local_from_aware(
            datetime(2027, 2, 10, 13, 30, tzinfo=UTC),
            "Invalid/Timezone",
        )

