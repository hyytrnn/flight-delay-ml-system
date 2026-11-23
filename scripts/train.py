"""Train baselines, select on validation, calibrate and evaluate locked test once."""

from __future__ import annotations

import argparse
from pathlib import Path

from flight_delay.models.train import train_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/clean_flights.parquet"))
    parser.add_argument("--artifact", type=Path, default=Path("models/flight_delay_bundle.joblib"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = train_evaluate(
        data_path=args.data,
        artifact_path=args.artifact,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        project_root=Path(__file__).resolve().parents[1],
        seed=args.seed,
    )
    print(f"selected_model={output.selected_model}")
    print(f"artifact={output.artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
