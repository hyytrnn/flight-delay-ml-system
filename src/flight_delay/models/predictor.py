"""Serializable model bundle and shared prediction behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from flight_delay.features.build import build_single_feature, scheduled_local_from_aware
from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame


class CategoricalFrameTransformer(BaseEstimator, TransformerMixin):
    """Prepare bounded integer-coded pandas category columns for native XGBoost hist."""

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        *,
        min_frequency: int = 100,
        max_categories: int = 300,
    ) -> None:
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def fit(self, X: pd.DataFrame, y: Any = None) -> CategoricalFrameTransformer:
        eff_min_freq = (
            min(self.min_frequency, max(1, len(X) // 20))
            if len(X) < 1_000
            else self.min_frequency
        )
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.int32,
            min_frequency=eff_min_freq,
            max_categories=self.max_categories,
        )
        cats = X[self.categorical_columns].astype("string").fillna("UNKNOWN").astype(str)
        self.encoder_.fit(cats)
        sample_out = self.encoder_.transform(cats.iloc[:1])
        self._max_code_ = [
            max(
                int(sample_out[:, i].max()),
                len(self.encoder_.categories_[i]) - 1,
            )
            for i in range(len(self.categorical_columns))
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num_df = X[self.numeric_columns].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        cats = X[self.categorical_columns].astype("string").fillna("UNKNOWN").astype(str)
        codes = self.encoder_.transform(cats)
        cat_cols: dict[str, pd.Categorical] = {}
        for idx, col in enumerate(self.categorical_columns):
            max_code = max(self._max_code_[idx], int(codes[:, idx].max()))
            cat_cols[col] = pd.Categorical(
                codes[:, idx], categories=range(-1, max_code + 1)
            )
        cat_df = pd.DataFrame(cat_cols, index=X.index)
        return pd.concat([cat_df, num_df], axis=1)


class CatBoostBundleEstimator:
    """Sklearn-like wrapper so CatBoost bundles share predict_proba parity."""

    def __init__(self, estimator: Any, categorical_columns: list[str]) -> None:
        self.estimator = estimator
        self.categorical_columns = list(categorical_columns)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        frame = features.copy()
        for column in self.categorical_columns:
            frame[column] = frame[column].astype("string").fillna("UNKNOWN")
        return np.asarray(self.estimator.predict_proba(frame), dtype=float)


@dataclass
class ModelBundle:
    model: Any
    calibrator: Any
    threshold: float
    metadata: dict[str, Any]
    route_lookup: dict[str, dict[str, float]]
    defaults: dict[str, float]
    airport_timezones: dict[str, str] = field(default_factory=dict)
    airport_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_group: str = "F0"
    calibration_method: str = "platt"

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.model.predict_proba(features)[:, 1], dtype=float)
        clipped = np.clip(raw, 1e-6, 1.0 - 1e-6)
        method = getattr(self, "calibration_method", "platt")
        if method == "none" or self.calibrator is None:
            return clipped
        if method == "isotonic":
            calibrated = np.asarray(self.calibrator.predict(clipped), dtype=float)
            return np.clip(calibrated, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return np.asarray(self.calibrator.predict_proba(logits)[:, 1], dtype=float)

    def _build_request_features(
        self,
        *,
        reporting_airline: str,
        origin: str,
        dest: str,
        scheduled_local: pd.Timestamp,
        values: dict[str, float],
    ) -> pd.DataFrame:
        group: FeatureGroup = getattr(self, "feature_group", "F0")
        if group == "F0":
            return build_single_feature(
                reporting_airline=reporting_airline,
                origin=origin,
                dest=dest,
                scheduled_departure_local=scheduled_local,
                distance=float(values["distance"]),
                crs_elapsed_time=float(values["crs_elapsed_time"]),
            )
        airport_lookup = getattr(self, "airport_lookup", {})
        origin_meta = airport_lookup.get(origin.upper(), {})
        dest_meta = airport_lookup.get(dest.upper(), {})
        raw = pd.DataFrame(
            [
                {
                    "reporting_airline": reporting_airline.upper(),
                    "origin": origin.upper(),
                    "dest": dest.upper(),
                    "scheduled_departure_local": scheduled_local,
                    "distance": float(values["distance"]),
                    "crs_elapsed_time": float(values["crs_elapsed_time"]),
                    "origin_latitude": origin_meta.get("latitude", np.nan),
                    "origin_longitude": origin_meta.get("longitude", np.nan),
                    "origin_elevation": origin_meta.get("elevation", np.nan),
                    "origin_subdivision": origin_meta.get("subdivision", "UNKNOWN"),
                    "dest_latitude": dest_meta.get("latitude", np.nan),
                    "dest_longitude": dest_meta.get("longitude", np.nan),
                    "dest_elevation": dest_meta.get("elevation", np.nan),
                    "dest_subdivision": dest_meta.get("subdivision", "UNKNOWN"),
                }
            ]
        )
        return build_v2_feature_frame(raw, group)

    def predict_request(
        self,
        *,
        reporting_airline: str,
        origin: str,
        dest: str,
        scheduled_departure: datetime,
    ) -> tuple[float, bool, str]:
        origin_code = origin.upper()
        dest_code = dest.upper()
        route_key = f"{origin_code}|{dest_code}"
        route = self.route_lookup.get(route_key)
        route_status = "known" if route is not None else "fallback_global_median"
        values = route or self.defaults

        airport_timezones = getattr(self, "airport_timezones", {})
        airport_lookup = getattr(self, "airport_lookup", {})
        origin_timezone = airport_timezones.get(origin_code) or airport_lookup.get(
            origin_code, {}
        ).get("timezone")
        require_tz = bool(
            self.metadata.get("require_origin_timezone", False)
            or getattr(self, "feature_group", "F0") != "F0"
        )
        if origin_timezone:
            scheduled_local = scheduled_local_from_aware(
                scheduled_departure, str(origin_timezone)
            )
        elif require_tz:
            raise ValueError(f"unknown origin airport timezone: {origin_code}")
        else:
            # Compatibility path for the frozen V1 artifact, which predates the
            # airport-timezone map. V2 artifacts must populate airport_timezones.
            scheduled_local = pd.Timestamp(scheduled_departure.replace(tzinfo=None))

        features = self._build_request_features(
            reporting_airline=reporting_airline,
            origin=origin_code,
            dest=dest_code,
            scheduled_local=scheduled_local,
            values=values,
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
