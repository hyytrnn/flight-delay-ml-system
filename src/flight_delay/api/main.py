"""Production-shaped FastAPI service for the verified flight-only model."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import UTC
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from flight_delay.api.schemas import BatchRequest, FlightRequest, PredictionResponse
from flight_delay.models.predictor import ModelBundle, load_bundle

REQUESTS = Counter("flight_delay_requests_total", "Flight delay API requests", ["path", "status"])
LATENCY = Histogram("flight_delay_request_seconds", "Flight delay API request latency", ["path"])


def create_app(model_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Flight Delay ML API", version="2.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-request-id"],
    )
    configured = Path(
        model_path
        or os.environ.get("FLIGHT_DELAY_MODEL_PATH", "models/flight_delay_v2_1_bundle.joblib")
    )
    state: dict[str, object] = {"bundle": None, "error": None, "path": str(configured)}
    try:
        state["bundle"] = load_bundle(configured)
    except (OSError, ValueError, TypeError, EOFError, ImportError, AttributeError, KeyError) as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"

    @app.middleware("http")
    async def request_metrics(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        try:
            response = await call_next(request)
        except Exception:
            REQUESTS.labels(request.url.path, "500").inc()
            raise
        elapsed = time.perf_counter() - started
        REQUESTS.labels(request.url.path, str(response.status_code)).inc()
        LATENCY.labels(request.url.path).observe(elapsed)
        response.headers["x-request-id"] = request_id
        bundle_obj = state["bundle"]
        model_version = (
            bundle_obj.metadata.get("model_version")
            if isinstance(bundle_obj, ModelBundle)
            else None
        )
        logging.getLogger("flight_delay.api").info(
            json.dumps(
                {
                    "request_id": request_id,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round(elapsed * 1000, 3),
                    "model_version": model_version,
                }
            )
        )
        return response

    def bundle_or_503() -> ModelBundle:
        bundle = state["bundle"]
        if not isinstance(bundle, ModelBundle):
            raise HTTPException(status_code=503, detail="model unavailable")
        return bundle

    def predict_one(request: FlightRequest) -> PredictionResponse:
        bundle = bundle_or_503()
        try:
            probability, label, route_status = bundle.predict_request(
                reporting_airline=request.airline,
                origin=request.origin,
                dest=request.destination,
                scheduled_departure=request.scheduled_departure,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse(
            delay_probability=probability,
            predicted_delayed=label,
            threshold=bundle.threshold,
            model_version=str(bundle.metadata["model_version"]),
            weather_status=str(
                bundle.metadata.get("weather_status", "not_used_flight_only_model")
            ),
            route_status=route_status,
            prediction_at_utc=request.prediction_at.astimezone(UTC),
            feature_contract_version=str(
                bundle.metadata.get("feature_contract_version", "t24-v1")
            ),
            data_freshness=bundle.metadata.get("data_freshness"),
        )

    @app.get("/health")
    def health():
        if state["bundle"] is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "model_loaded": False,
                    "model_path": state["path"],
                    "error": state["error"],
                },
            )
        bundle = bundle_or_503()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": bundle.metadata["model_version"],
            "feature_contract_version": bundle.metadata.get(
                "feature_contract_version", "t24-v1"
            ),
            "bundle_path": state["path"],
            "weather_provider_status": "not_required",
        }

    @app.get("/model-info")
    def model_info():
        return bundle_or_503().metadata

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: FlightRequest) -> PredictionResponse:
        return predict_one(request)

    @app.post("/predict/batch", response_model=list[PredictionResponse])
    def predict_batch(request: BatchRequest) -> list[PredictionResponse]:
        return [predict_one(item) for item in request.flights]

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()

