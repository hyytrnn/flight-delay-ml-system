"""Serializable model bundle and shared prediction behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flight_delay.features.build import build_single_feature, scheduled_local_from_aware


@dataclass
class ModelBundle:
    model: Any
    calibrator: Any
    threshold: float
    metadata: dict[str, Any]
    route_lookup: dict[str, dict[str, float]]
    defaults: dict[str, float]
    airport_timezones: dict[str, str] = field(default_factory=dict)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.model.predict_proba(features)[:, 1], dtype=float)
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return np.asarray(self.calibrator.predict_proba(logits)[:, 1], dtype=float)

    def predict_request(
        self,
        *,
        reporting_airline: str,
        origin: str,
        dest: str,
        scheduled_departure: datetime,
    ) -> tuple[float, bool, str]:
        route_key = f"{origin.upper()}|{dest.upper()}"
        route = self.route_lookup.get(route_key)
        route_status = "known" if route is not None else "fallback_global_median"
        values = route or self.defaults
        airport_timezones = getattr(self, "airport_timezones", {})
        origin_timezone = airport_timezones.get(origin.upper())
        if origin_timezone:
            scheduled_local = scheduled_local_from_aware(
                scheduled_departure, origin_timezone
            )
        else:
            # Compatibility path for the frozen V1 artifact, which predates the
            # airport-timezone map. V2 artifacts must populate airport_timezones.
            scheduled_local = pd.Timestamp(scheduled_departure.replace(tzinfo=None))
        features = build_single_feature(
            reporting_airline=reporting_airline,
            origin=origin,
            dest=dest,
            scheduled_departure_local=scheduled_local,
            distance=float(values["distance"]),
            crs_elapsed_time=float(values["crs_elapsed_time"]),
        )
        probability = float(self.predict_proba(features)[0])
        return probability, probability >= self.threshold, route_status


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)


def load_bundle(path: str | Path) -> ModelBundle:
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, ModelBundle):
        raise TypeError("Artifact is not a trusted ModelBundle")
    return loaded

