from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from flight_delay.features.v2 import build_v2_feature_frame, v2_feature_columns
from flight_delay.models.predictor import ModelBundle, save_bundle
from flight_delay.models.tuning_v2 import _onehot_preprocessor
from flight_delay.monitoring.report import build_report, population_stability_index


def test_psi_is_zero_for_identical_values() -> None:
    values = pl.Series([1.0, 2.0, 3.0, 4.0]).to_numpy()
    assert population_stability_index(values, values) == 0.0


def test_monitoring_report_separates_drift_from_performance() -> None:
    reference = pl.DataFrame(
        {"distance": [100.0, 200.0], "crs_elapsed_time": [60.0, 90.0], "dep_del15": [0, 1]}
    )
    current = pl.DataFrame(
        {"distance": [800.0, 900.0], "crs_elapsed_time": [180.0, 200.0], "dep_del15": [1, 1]}
    )
    report = build_report(reference, current)
    assert "not proof of performance drift" in str(report["interpretation"])
    assert isinstance(report["alerts"], dict)


def test_monitoring_report_with_bundle_outcome_and_alerts(tmp_path: Path) -> None:
    rows = []
    for idx in range(20):
        rows.append(
            {
                "reporting_airline": "AA" if idx % 2 == 0 else "UA",
                "origin": "JFK" if idx % 2 == 0 else "ORD",
                "dest": "LAX" if idx % 2 == 0 else "JFK",
                "scheduled_departure_local": pd.Timestamp("2025-01-01T08:00:00")
                if idx % 2 == 0
                else pd.Timestamp("2025-07-02T18:00:00"),
                "distance": 2475.0 if idx % 2 == 0 else 740.0,
                "crs_elapsed_time": 360.0 if idx % 2 == 0 else 130.0,
                "origin_latitude": 40.64,
                "origin_longitude": -73.78,
                "origin_elevation": 13.0 if idx % 2 == 0 else 672.0,
                "origin_subdivision": "New York" if idx % 2 == 0 else "Illinois",
                "dest_latitude": 33.94,
                "dest_longitude": -118.41,
                "dest_elevation": 125.0,
                "dest_subdivision": "California",
                "dep_del15": idx % 2,
            }
        )
    ref_df = pd.DataFrame(rows)
    cur_df = ref_df.copy()
    cur_df.loc[cur_df.index[:6], "dest"] = "HNL"
    feats = build_v2_feature_frame(ref_df, "F3")
    num, cat = v2_feature_columns("F3")
    pipe = Pipeline(
        [
            ("prep", _onehot_preprocessor(num, cat)),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    ).fit(feats, ref_df["dep_del15"].to_numpy())
    bundle_path = tmp_path / "b.joblib"
    save_bundle(
        ModelBundle(
            model=pipe,
            calibrator=None,
            threshold=0.5,
            metadata={"model_version": "test-v2.1", "threshold_precision_target_35": 0.6},
            route_lookup={},
            defaults={"distance": 1000.0, "crs_elapsed_time": 150.0},
            feature_group="F3",
            calibration_method="none",
        ),
        bundle_path,
    )

    report = build_report(
        pl.from_pandas(ref_df),
        pl.from_pandas(cur_df),
        bundle_path=bundle_path,
        config_path=Path("configs/monitoring.yaml"),
        window_label="unit_test",
    )
    assert report["model_version"] == "test-v2.1"
    assert report["score_histogram"] is not None
    assert sum(report["score_histogram"]["counts"]) == len(cur_df)
    assert report["outcome_metrics"] is not None
    assert "pr_auc" in report["outcome_metrics"]
    assert float(report["unseen_route_rate"]) == np.mean(cur_df["dest"] == "HNL")
    assert report["alerts"]["unseen_route_status"] == "critical"
    assert report["alerts"]["retraining_recommended"] is True
