"""Run reproducible Optuna pilot studies on V2 temporal folds."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import yaml

from flight_delay.models.tuning_v2 import (
    export_study,
    load_development_frame,
    load_splits,
    run_study,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tuning_v2.yaml"))
    parser.add_argument("--trials", type=int)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = args.models or config["models"]
    trials = args.trials or int(config["pilot_trials_per_model"])
    data = load_development_frame(
        Path(config["data_root"]), date.fromisoformat(config["development_end_exclusive"])
    )
    folds = load_splits(Path(config["splits"]))
    storage_path = Path(config["storage"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.as_posix()}"
    for model_name in models:
        study = run_study(
            model_name=model_name,
            frame=data,
            feature_group=config["feature_group"],
            folds=folds,
            storage=storage,
            trials=trials,
            timeout=int(config["timeout_seconds_per_model"]),
            seed=int(config["seed"]),
        )
        export_study(study, Path(config["output_root"]), model_name)
        print(model_name, study.best_value, study.best_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

