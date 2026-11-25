from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import httpx
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from flight_delay.api.main import create_app
from flight_delay.features.build import build_single_feature
from flight_delay.models.predictor import ModelBundle, save_bundle


def make_bundle(path) -> None:
    x = pd.concat(
        [
            build_single_feature(
                reporting_airline="AA",
                origin="JFK",
                dest="LAX",
                scheduled_departure_local=pd.Timestamp("2025-01-01T08:00:00"),
                distance=2475,
                crs_elapsed_time=360,
            ),
            build_single_feature(
                reporting_airline="UA",
                origin="ORD",
                dest="JFK",
                scheduled_departure_local=pd.Timestamp("2025-01-02T18:00:00"),
                distance=740,
                crs_elapsed_time=130,
            ),
        ],
        ignore_index=True,
    )
    model = DummyClassifier(strategy="prior").fit(x, np.array([0, 1]))
    logits = np.array([[-1.0], [1.0], [-0.5], [0.5]])
    calibrator = LogisticRegression().fit(logits, np.array([0, 1, 0, 1]))
    bundle = ModelBundle(
        model=model,
        calibrator=calibrator,
        threshold=0.5,
        metadata={"model_version": "test-v1", "feature_contract_version": "t24-v1"},
        route_lookup={"JFK|LAX": {"distance": 2475.0, "crs_elapsed_time": 360.0}},
        defaults={"distance": 1000.0, "crs_elapsed_time": 150.0},
    )
    save_bundle(bundle, path)


@pytest.mark.asyncio
async def test_api_predict_and_health(tmp_path) -> None:
    artifact = tmp_path / "bundle.joblib"
    make_bundle(artifact)
    departure = datetime(2027, 2, 10, 8, 30, tzinfo=timezone(timedelta(hours=-5)))
    payload = {
        "airline": "AA",
        "origin": "JFK",
        "destination": "LAX",
        "scheduled_departure": departure.isoformat(),
        "prediction_at": (departure - timedelta(hours=24)).isoformat(),
    }
    transport = httpx.ASGITransport(app=create_app(artifact))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        response = await client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0 <= response.json()["delay_probability"] <= 1
    assert response.json()["weather_status"] == "not_used_flight_only_model"


@pytest.mark.asyncio
async def test_api_rejects_wrong_prediction_horizon(tmp_path) -> None:
    artifact = tmp_path / "bundle.joblib"
    make_bundle(artifact)
    departure = datetime(2027, 2, 10, 8, 30, tzinfo=UTC)
    transport = httpx.ASGITransport(app=create_app(artifact))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": departure.isoformat(),
                "prediction_at": (departure - timedelta(hours=2)).isoformat(),
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_is_not_ready_when_model_missing(tmp_path) -> None:
    transport = httpx.ASGITransport(app=create_app(tmp_path / "missing.joblib"))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 503


def _make_v2_f3_bundle(path) -> None:
    from sklearn.pipeline import Pipeline

    from flight_delay.features.v2 import build_v2_feature_frame, v2_feature_columns
    from flight_delay.models.tuning_v2 import _onehot_preprocessor

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
    features = build_v2_feature_frame(rows, "F3")
    numeric, categorical = v2_feature_columns("F3")
    pipe = Pipeline(
        [
            ("preprocessor", _onehot_preprocessor(numeric, categorical)),
            ("classifier", LogisticRegression(max_iter=200, random_state=42)),
        ]
    ).fit(features, np.array([0, 1]))
    bundle = ModelBundle(
        model=pipe,
        calibrator=None,
        threshold=0.5,
        metadata={
            "model_version": "flight-only-v2.0.0-test",
            "feature_contract_version": "t24-v2-F3",
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
        feature_group="F3",
        calibration_method="none",
    )
    save_bundle(bundle, path)


@pytest.mark.asyncio
async def test_v2_api_timezone_instant_parity_dst_and_fail_closed(tmp_path) -> None:
    artifact = tmp_path / "v2_bundle.joblib"
    _make_v2_f3_bundle(artifact)
    transport = httpx.ASGITransport(app=create_app(artifact))

    # Same instant represented in UTC vs EST (-05:00) must produce identical probability
    instant_utc = datetime(2027, 2, 10, 13, 30, tzinfo=UTC)
    instant_est = instant_utc.astimezone(timezone(timedelta(hours=-5)))
    # DST transition day (EDT -04:00)
    dst_instant = datetime(2027, 7, 15, 12, 30, tzinfo=UTC)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r_utc = await client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": instant_utc.isoformat(),
                "prediction_at": (instant_utc - timedelta(hours=24)).isoformat(),
            },
        )
        r_est = await client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": instant_est.isoformat(),
                "prediction_at": (instant_est - timedelta(hours=24)).isoformat(),
            },
        )
        r_dst = await client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "ZZZ",
                "scheduled_departure": dst_instant.isoformat(),
                "prediction_at": (dst_instant - timedelta(hours=24)).isoformat(),
            },
        )
        r_unknown_tz = await client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "ZZZ",
                "destination": "LAX",
                "scheduled_departure": instant_utc.isoformat(),
                "prediction_at": (instant_utc - timedelta(hours=24)).isoformat(),
            },
        )

    assert r_utc.status_code == 200
    assert r_est.status_code == 200
    assert r_utc.json()["delay_probability"] == pytest.approx(r_est.json()["delay_probability"])
    assert r_utc.json()["feature_contract_version"] == "t24-v2-F3"
    assert r_dst.status_code == 200
    assert r_dst.json()["route_status"] == "fallback_global_median"
    assert r_unknown_tz.status_code == 422


@pytest.mark.asyncio
async def test_v1_frozen_bundle_compatibility_if_present() -> None:
    from pathlib import Path

    v1_path = Path("models/flight_delay_bundle.joblib")
    if not v1_path.exists():
        pytest.skip("V1 artifact not present")
    departure = datetime(2027, 2, 10, 8, 30, tzinfo=timezone(timedelta(hours=-5)))
    transport = httpx.ASGITransport(app=create_app(v1_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": departure.isoformat(),
                "prediction_at": (departure - timedelta(hours=24)).isoformat(),
            },
        )
    assert response.status_code == 200
    assert response.json()["model_version"] == "flight-only-v1.0.0"


@pytest.mark.asyncio
async def test_v2_1_default_bundle_loads_and_serves_if_present() -> None:
    from pathlib import Path

    v21_path = Path("models/flight_delay_v2_1_bundle.joblib")
    if not v21_path.exists():
        pytest.skip("V2.1 artifact not present")
    departure = datetime(2027, 2, 10, 13, 30, tzinfo=UTC)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_version"] == "flight-only-v2.1.0-xgboost"
        assert health.json()["feature_contract_version"] == "t24-v2.1-F3"
        response = await client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": departure.isoformat(),
                "prediction_at": (departure - timedelta(hours=24)).isoformat(),
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "flight-only-v2.1.0-xgboost"
    assert body["feature_contract_version"] == "t24-v2.1-F3"
    assert 0.0 <= body["delay_probability"] <= 1.0


@pytest.mark.asyncio
async def test_reviewer_sample_flights_and_dst_spring_forward_parity() -> None:
    import json
    from pathlib import Path

    v21_path = Path("models/flight_delay_v2_1_bundle.joblib")
    if not v21_path.exists():
        pytest.skip("V2.1 artifact not present")

    samples = json.loads(Path("data/examples/sample_flights.json").read_text(encoding="utf-8"))
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for case in samples["examples"]:
            resp = await client.post("/predict", json=case["payload"])
            assert resp.status_code == case["expected_http"], case["name"]
            if case["expected_http"] == 200:
                assert resp.json()["model_version"] == "flight-only-v2.1.0-xgboost"

        # US Spring-forward DST day (2027-03-14: EDT -04:00 vs UTC)
        dep_utc = datetime(2027, 3, 14, 14, 0, tzinfo=UTC)
        dep_edt = dep_utc.astimezone(timezone(timedelta(hours=-4)))
        r_utc = await client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": dep_utc.isoformat(),
                "prediction_at": (dep_utc - timedelta(hours=24)).isoformat(),
            },
        )
        r_edt = await client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": dep_edt.isoformat(),
                "prediction_at": (dep_edt - timedelta(hours=24)).isoformat(),
            },
        )
        assert r_utc.status_code == 200 and r_edt.status_code == 200
        assert r_utc.json()["delay_probability"] == pytest.approx(
            r_edt.json()["delay_probability"]
        )


