from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from flight_delay.features.build import MODEL_FEATURES, build_single_feature


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

