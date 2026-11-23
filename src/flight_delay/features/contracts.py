"""T-24h feature eligibility contract.

This module is intentionally allowlist-based. New fields do not become model
features merely because they appear in a BTS extract.
"""

from __future__ import annotations

LEAKAGE_COLUMNS = frozenset(
    {
        "dep_del15",
        "dep_delay",
        "dep_delay_minutes",
        "dep_time",
        "arr_time",
        "actual_elapsed_time",
        "taxi_out",
        "taxi_in",
        "wheels_off",
        "wheels_on",
        "arr_del15",
        "arr_delay",
        "arr_delay_minutes",
        "cancelled",
        "cancellation_code",
        "diverted",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    }
)

BASELINE_FEATURE_COLUMNS = (
    "reporting_airline",
    "origin",
    "dest",
    "distance",
    "crs_elapsed_time",
    "scheduled_departure_local",
)


def assert_feature_columns_are_eligible(columns: list[str] | tuple[str, ...]) -> None:
    """Reject post-decision and target fields from a proposed feature set."""

    blocked = sorted(set(columns).intersection(LEAKAGE_COLUMNS))
    if blocked:
        raise ValueError(f"Leakage columns are not eligible at T-24h: {blocked}")

