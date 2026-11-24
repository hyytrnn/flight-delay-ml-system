from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from scripts.feature_flow_v2 import build_asof_hourly_flow_table


def test_asof_hourly_flow_uses_only_lags_at_least_24h_prior() -> None:
    base = datetime(2024, 5, 1, 10, 0, tzinfo=UTC)
    rows = []
    # Day 0: 2 flights at 10:00 UTC
    for _ in range(2):
        rows.append({"origin": "JFK", "scheduled_departure_utc": base})
    # Day 1 (t+24h): 5 flights at 10:00 UTC
    for _ in range(5):
        rows.append({"origin": "JFK", "scheduled_departure_utc": base + timedelta(hours=24)})
    # Day 7 (t+168h): 3 flights at 10:00 UTC
    for _ in range(3):
        rows.append({"origin": "JFK", "scheduled_departure_utc": base + timedelta(hours=168)})

    flow = build_asof_hourly_flow_table(pl.DataFrame(rows))
    day1 = flow.filter(pl.col("hour_utc") == base + timedelta(hours=24))
    assert day1.item(0, "hourly_flow") == pytest.approx(5.0)
    assert day1.item(0, "flow_lag_24h") == pytest.approx(2.0)
    assert day1.item(0, "flow_lag_7d") is None

    day7 = flow.filter(pl.col("hour_utc") == base + timedelta(hours=168))
    assert day7.item(0, "flow_lag_7d") == pytest.approx(2.0)
    assert day7.item(0, "flow_lag_24h") is None
