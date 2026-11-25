"""Verify V2.1/V2/V1 model bundles and FastAPI serving parity on a fresh checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from flight_delay.api.main import create_app


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("models/manifest_v2_1.json")
    )
    parser.add_argument(
        "--v2-manifest", type=Path, default=Path("models/locked_v2_manifest.json")
    )
    parser.add_argument(
        "--v1-bundle", type=Path, default=Path("models/flight_delay_bundle.joblib")
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"missing V2.1 manifest: {args.manifest}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bundle_path = Path(manifest["bundle_path"])
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing V2.1 bundle: {bundle_path}")
    actual_sha = _sha256(bundle_path)
    if actual_sha != manifest["bundle_sha256"]:
        raise AssertionError(
            f"V2.1 bundle SHA256 mismatch: expected {manifest['bundle_sha256']}, got {actual_sha}"
        )

    if args.v2_manifest.exists():
        v2_m = json.loads(args.v2_manifest.read_text(encoding="utf-8"))
        v2_path = Path(v2_m["v2_bundle_path"])
        if v2_path.exists() and _sha256(v2_path) != v2_m["v2_bundle_sha256"]:
            raise AssertionError("locked V2 bundle SHA256 mismatch")

    # Boot FastAPI with default bundle resolution (no path argument -> FLIGHT_DELAY_MODEL_PATH default)
    app = create_app()
    client = TestClient(app)

    health = client.get("/health")
    if health.status_code != 200:
        raise AssertionError(f"/health failed: {health.status_code} {health.text}")
    health_json = health.json()
    if health_json["model_version"] != manifest["model_version"]:
        raise AssertionError(
            f"unexpected default model_version: {health_json['model_version']} != {manifest['model_version']}"
        )

    payload_utc = {
        "airline": "AA",
        "origin": "JFK",
        "destination": "LAX",
        "scheduled_departure": "2027-02-10T13:30:00Z",
        "prediction_at": "2027-02-09T13:30:00Z",
    }
    payload_est = {
        "airline": "AA",
        "origin": "JFK",
        "destination": "LAX",
        "scheduled_departure": "2027-02-10T08:30:00-05:00",
        "prediction_at": "2027-02-09T08:30:00-05:00",
    }
    r_utc = client.post("/predict", json=payload_utc)
    r_est = client.post("/predict", json=payload_est)
    if r_utc.status_code != 200 or r_est.status_code != 200:
        raise AssertionError(f"/predict parity failed: {r_utc.text} | {r_est.text}")
    p_utc = r_utc.json()["delay_probability"]
    p_est = r_est.json()["delay_probability"]
    if abs(p_utc - p_est) > 1e-12:
        raise AssertionError(f"UTC vs offset parity mismatch: {p_utc} vs {p_est}")

    r_unseen = client.post(
        "/predict",
        json={
            "airline": "AA",
            "origin": "JFK",
            "destination": "ZZZ",
            "scheduled_departure": "2027-02-10T13:30:00Z",
            "prediction_at": "2027-02-09T13:30:00Z",
        },
    )
    if r_unseen.status_code != 200 or r_unseen.json()["route_status"] != "fallback_global_median":
        raise AssertionError(f"unseen route check failed: {r_unseen.text}")

    r_bad_tz = client.post(
        "/predict",
        json={
            "airline": "AA",
            "origin": "ZZZ",
            "destination": "LAX",
            "scheduled_departure": "2027-02-10T13:30:00Z",
            "prediction_at": "2027-02-09T13:30:00Z",
        },
    )
    if r_bad_tz.status_code != 422:
        raise AssertionError(f"unknown origin must fail closed with 422, got {r_bad_tz.status_code}")

    v1_status = None
    if args.v1_bundle.exists():
        v1_client = TestClient(create_app(args.v1_bundle))
        v1_h = v1_client.get("/health")
        if v1_h.status_code != 200:
            raise AssertionError(f"V1 compatibility failed: {v1_h.text}")
        v1_status = v1_h.json()["model_version"]

    summary = {
        "status": "PASS",
        "bundle_path": str(bundle_path),
        "bundle_sha256": actual_sha,
        "health": health_json,
        "predict_utc": r_utc.json(),
        "predict_est_probability": p_est,
        "unseen_route_status": r_unseen.json()["route_status"],
        "unknown_origin_http_status": r_bad_tz.status_code,
        "v1_backward_compat_model_version": v1_status,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FRESH_CLONE_CHECK_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
