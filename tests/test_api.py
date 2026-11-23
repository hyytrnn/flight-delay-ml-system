from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
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


def test_api_predict_and_health(tmp_path) -> None:
    artifact = tmp_path / "bundle.joblib"
    make_bundle(artifact)
    client = TestClient(create_app(artifact))
    departure = datetime(2027, 2, 10, 8, 30, tzinfo=timezone(timedelta(hours=-5)))
    payload = {
        "airline": "AA",
        "origin": "JFK",
        "destination": "LAX",
        "scheduled_departure": departure.isoformat(),
        "prediction_at": (departure - timedelta(hours=24)).isoformat(),
    }
    assert client.get("/health").status_code == 200
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0 <= response.json()["delay_probability"] <= 1
    assert response.json()["weather_status"] == "not_used_flight_only_model"


def test_api_rejects_wrong_prediction_horizon(tmp_path) -> None:
    artifact = tmp_path / "bundle.joblib"
    make_bundle(artifact)
    client = TestClient(create_app(artifact))
    departure = datetime(2027, 2, 10, 8, 30, tzinfo=UTC)
    response = client.post(
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


def test_health_is_not_ready_when_model_missing(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "missing.joblib"))
    assert client.get("/health").status_code == 503

