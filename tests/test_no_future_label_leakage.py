from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from flight_delay.features.historical_priors import (
    HISTORICAL_PRIOR_COLUMNS,
    compute_asof_historical_priors,
    compute_label_available_utc,
)


def _sample_history() -> pd.DataFrame:
    rows = []
    # Jan 2024 (available 2024-04-01), Mar 2024 (available 2024-06-01), May 2024 (available 2024-08-01)
    for idx in range(20):
        rows.append(
            {
                "flight_date": pd.Timestamp("2024-01-15").date(),
                "origin": "JFK",
                "dest": "LAX",
                "reporting_airline": "AA",
                "dep_del15": 1 if idx < 10 else 0,
            }
        )
    for idx in range(20):
        rows.append(
            {
                "flight_date": pd.Timestamp("2024-03-10").date(),
                "origin": "JFK",
                "dest": "LAX",
                "reporting_airline": "AA",
                "dep_del15": 1 if idx < 18 else 0,
            }
        )
    for idx in range(20):
        rows.append(
            {
                "flight_date": pd.Timestamp("2024-05-10").date(),
                "origin": "JFK",
                "dest": "LAX",
                "reporting_airline": "AA",
                "dep_del15": 0,
            }
        )
    return pd.DataFrame(rows)


def test_publication_lag_blocks_future_and_unreleased_recent_labels() -> None:
    history = _sample_history()
    avail = compute_label_available_utc(history["flight_date"], publication_lag_months=2)
    assert avail.iloc[0] == pd.Timestamp("2024-04-01T00:00:00Z")
    assert avail.iloc[25] == pd.Timestamp("2024-06-01T00:00:00Z")

    # Target predicted on 2024-05-15 UTC: ONLY Jan 2024 is available; Mar & May must NOT affect it!
    targets = pd.DataFrame(
        [
            {
                "origin": "JFK",
                "dest": "LAX",
                "reporting_airline": "AA",
                "prediction_at_utc": datetime(2024, 5, 15, 12, 0, tzinfo=UTC),
                "scheduled_departure_utc": datetime(2024, 5, 16, 12, 0, tzinfo=UTC),
            },
            {
                "origin": "ORD",
                "dest": "SFO",
                "reporting_airline": "UA",
                "prediction_at_utc": datetime(2024, 5, 15, 12, 0, tzinfo=UTC),
                "scheduled_departure_utc": datetime(2024, 5, 16, 12, 0, tzinfo=UTC),
            },
        ]
    )

    baseline_priors = compute_asof_historical_priors(
        history, targets, alpha=10.0, min_support=15, publication_lag_months=2
    )

    mutated = history.copy()
    # Flip labels inside unreleased lag window (2024-03) and future window (2024-05)
    mutated.loc[mutated["flight_date"] >= pd.Timestamp("2024-03-01").date(), "dep_del15"] = 1 - (
        mutated.loc[mutated["flight_date"] >= pd.Timestamp("2024-03-01").date(), "dep_del15"]
    )
    mutated_priors = compute_asof_historical_priors(
        mutated, targets, alpha=10.0, min_support=15, publication_lag_months=2
    )

    pd.testing.assert_frame_equal(baseline_priors, mutated_priors)
    assert list(baseline_priors.columns) == HISTORICAL_PRIOR_COLUMNS

    # Row 0 (JFK|LAX, 20 support from Jan 2024, 10 pos -> rate = (10 + 10*0.5)/(20+10) = 0.5)
    assert baseline_priors.loc[0, "route_hist_delay_rate_asof"] == pytest.approx(0.5)
    assert baseline_priors.loc[0, "route_hist_unseen_asof"] == 0
    assert baseline_priors.loc[0, "hist_prior_freshness_age_days"] > 100.0

    # Row 1 (unseen ORD|SFO shrinks to global_prior = 0.5 and marked unseen)
    assert baseline_priors.loc[1, "route_hist_delay_rate_asof"] == pytest.approx(0.5)
    assert baseline_priors.loc[1, "route_hist_unseen_asof"] == 1
    assert np.isclose(baseline_priors.loc[1, "route_hist_support_asof"], 0.0)


def test_bayesian_smoothing_shrinks_low_support_to_prior() -> None:
    history = pd.DataFrame(
        [{"flight_date": pd.Timestamp("2024-01-05").date(), "origin": "JFK", "dest": "LAX", "reporting_airline": "AA", "dep_del15": 0}] * 100
        + [{"flight_date": pd.Timestamp("2024-01-05").date(), "origin": "BOS", "dest": "SFO", "reporting_airline": "B6", "dep_del15": 1}] * 2
    )
    targets = pd.DataFrame(
        [
            {"origin": "BOS", "dest": "SFO", "reporting_airline": "B6", "prediction_at_utc": datetime(2024, 5, 1, 12, 0, tzinfo=UTC)},
        ]
    )
    priors = compute_asof_historical_priors(history, targets, alpha=50.0, min_support=10)
    # 2 positives / 102 global = 0.0196; route has 2/2 -> smoothed = (2 + 50*(2/102)) / 52 = ~0.0573
    assert 0.0196 < priors.loc[0, "route_hist_delay_rate_asof"] < 0.15
    assert priors.loc[0, "route_hist_unseen_asof"] == 1
