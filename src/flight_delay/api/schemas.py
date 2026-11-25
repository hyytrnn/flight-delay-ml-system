"""Validated API request and response schemas."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator, model_validator


class FlightRequest(BaseModel):
    airline: str = Field(min_length=2, max_length=5)
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    scheduled_departure: datetime
    prediction_at: datetime

    @field_validator("airline", "origin", "destination")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        if not value.isalpha():
            raise ValueError("codes must contain letters only")
        return value.upper()

    @model_validator(mode="after")
    def validate_times(self) -> FlightRequest:
        if self.scheduled_departure.tzinfo is None or self.prediction_at.tzinfo is None:
            raise ValueError("timezone-aware datetimes required")
        horizon = self.scheduled_departure - self.prediction_at
        if abs(horizon - timedelta(hours=24)) > timedelta(minutes=5):
            raise ValueError("prediction_at must be T-24h within five minutes")
        return self


class PredictionResponse(BaseModel):
    delay_probability: float
    predicted_delayed: bool
    threshold: float
    model_version: str
    weather_status: str
    route_status: str
    prediction_at_utc: datetime
    feature_contract_version: str = "t24-v1"
    data_freshness: str | None = None


class BatchRequest(BaseModel):
    flights: list[FlightRequest] = Field(min_length=1, max_length=100)

