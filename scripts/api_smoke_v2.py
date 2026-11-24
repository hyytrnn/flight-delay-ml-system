"""Run local FastAPI V2 smoke checks against the frozen V2 bundle."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from flight_delay.api.main import create_app


def main() -> int:
    bundle_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "models/flight_delay_v2_bundle.joblib"
    )
    app = create_app(bundle_path)
    with TestClient(app) as client:
        health = client.get("/health")
        info = client.get("/model-info")
        instant = datetime(2027, 2, 10, 13, 30, tzinfo=UTC)
        est = instant.astimezone(timezone(timedelta(hours=-5)))
        r1 = client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": instant.isoformat(),
                "prediction_at": (instant - timedelta(hours=24)).isoformat(),
            },
        )
        r2 = client.post(
            "/predict",
            json={
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": est.isoformat(),
                "prediction_at": (est - timedelta(hours=24)).isoformat(),
            },
        )
        metrics = client.get("/metrics")
        summary = {
            "health_status": health.status_code,
            "health": health.json(),
            "model_version": info.json().get("model_version"),
            "feature_contract_version": info.json().get("feature_contract_version"),
            "predict_utc_status": r1.status_code,
            "predict_utc": r1.json(),
            "predict_est": r2.json(),
            "metrics_status": metrics.status_code,
        }
        print(json.dumps(summary, indent=2))
        assert health.status_code == 200
        assert r1.status_code == 200 and r2.status_code == 200
        assert abs(r1.json()["delay_probability"] - r2.json()["delay_probability"]) < 1e-9
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
