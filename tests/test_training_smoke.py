from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from flight_delay.models.train import train_evaluate


def synthetic_partition(start: date, rows: int, offset: int) -> pl.DataFrame:
    records = []
    for index in range(rows):
        service_date = start + timedelta(days=index % 20)
        hour = (index * 3 + offset) % 24
        records.append(
            {
                "flight_date": service_date,
                "reporting_airline": "AA" if index % 2 else "UA",
                "origin": "JFK" if index % 3 else "ORD",
                "dest": "LAX" if index % 3 else "JFK",
                "distance": float(700 + (index % 5) * 300),
                "crs_elapsed_time": float(100 + (index % 7) * 20),
                "scheduled_departure_local": datetime.combine(
                    service_date, datetime.min.time()
                ).replace(hour=hour),
                "dep_del15": (index + offset) % 2,
            }
        )
    return pl.DataFrame(records)


def test_end_to_end_training_smoke(tmp_path) -> None:
    frame = pl.concat(
        [
            synthetic_partition(date(2023, 1, 1), 40, 0),
            synthetic_partition(date(2024, 1, 1), 30, 1),
            synthetic_partition(date(2024, 10, 1), 30, 2),
            synthetic_partition(date(2025, 1, 1), 30, 3),
        ]
    )
    data_path = tmp_path / "flights.parquet"
    frame.write_parquet(data_path)
    artifact = tmp_path / "bundle.joblib"
    output = train_evaluate(
        data_path=data_path,
        artifact_path=artifact,
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
        project_root=tmp_path,
        seed=42,
    )
    assert artifact.exists()
    assert output.selected_model in {
        "dummy_prior",
        "logistic_regression",
        "hist_gradient_boosting",
        "xgboost",
    }
    assert 0 <= output.test_metrics["pr_auc"] <= 1

