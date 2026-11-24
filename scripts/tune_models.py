"""Run V2 baseline comparisons and reproducible Optuna studies on expanding temporal folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from flight_delay.data.temporal import parse_date
from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame
from flight_delay.models.tuning_v2 import (
    AllModelName,
    evaluate_params,
    export_study,
    load_development_frame,
    load_splits,
    run_study,
    summarize_fold_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tuning_v2.yaml"))
    parser.add_argument("--trials", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--feature-group", type=str)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--include-class-weight-pilot", action="store_true")
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("reports/metrics/v2_model_comparison.csv"),
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=Path("reports/metrics/v2_model_comparison_folds.csv"),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = args.models or config["models"]
    trials = int(args.trials) if args.trials is not None else int(config["pilot_trials_per_model"])
    feature_group: FeatureGroup = args.feature_group or config["feature_group"]
    seed = int(config["seed"])

    data = load_development_frame(
        Path(config["data_root"]), parse_date(config["development_end_exclusive"])
    )
    folds = load_splits(Path(config["splits"]))
    features = build_v2_feature_frame(data, feature_group)
    labels = data.get_column("dep_del15").to_numpy().astype(int)
    dates = pd.to_datetime(data.get_column("flight_date").to_pandas())

    comparison_rows: list[dict[str, object]] = (
        pd.read_csv(args.comparison_output).to_dict("records")
        if args.comparison_output.exists()
        else []
    )
    fold_rows: list[dict[str, object]] = (
        pd.read_csv(args.fold_output).to_dict("records")
        if args.fold_output.exists()
        else []
    )

    def _upsert_summary(row: dict[str, object], new_folds: list[dict[str, object]]) -> None:
        key = (row["model"], row["class_weight"], row["source"])
        comparison_rows[:] = [
            r for r in comparison_rows if (r["model"], r["class_weight"], r["source"]) != key
        ]
        comparison_rows.append(row)
        fold_rows[:] = [
            r for r in fold_rows if (r["model"], r["class_weight"], r["source"]) != key
        ]
        fold_rows.extend(new_folds)
        args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(comparison_rows).sort_values("mean_pr_auc", ascending=False).to_csv(
            args.comparison_output, index=False
        )
        pd.DataFrame(fold_rows).to_csv(args.fold_output, index=False)

    if args.include_baselines:
        baselines: list[tuple[AllModelName, dict[str, object]]] = [
            ("dummy_prior", {}),
            ("logistic_regression", {"C": 1.0, "max_iter": 1_500}),
            (
                "hist_gradient_boosting",
                {"learning_rate": 0.08, "max_iter": 220, "max_leaf_nodes": 31, "min_samples_leaf": 50},
            ),
        ]
        for model_name, params in baselines:
            results = evaluate_params(
                model_name=model_name,
                params=params,
                features=features,
                labels=labels,
                dates=dates,
                folds=folds,
                feature_group=feature_group,
                seed=seed,
                class_weight_mode="none",
            )
            summary = summarize_fold_results(
                model_name=model_name,
                feature_group=feature_group,
                class_weight_mode="none",
                results=results,
                source="baseline",
            )
            _upsert_summary(
                summary,
                [
                    {
                        "model": model_name,
                        "class_weight": "none",
                        "source": "baseline",
                        **r.__dict__,
                    }
                    for r in results
                ],
            )
            print("baseline", summary, flush=True)

    storage_path = Path(config["storage"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.as_posix()}"

    if trials > 0:
        for model_name in models:
            study = run_study(
                model_name=model_name,
                frame=data,
                feature_group=feature_group,
                folds=folds,
                storage=storage,
                trials=trials,
                timeout=int(config["timeout_seconds_per_model"]),
                seed=seed,
            )
            export_study(study, Path(config["output_root"]), model_name)
            best_results = evaluate_params(
                model_name=model_name,
                params=study.best_params,
                features=features,
                labels=labels,
                dates=dates,
                folds=folds,
                feature_group=feature_group,
                seed=seed,
                class_weight_mode="none",
            )
            summary = summarize_fold_results(
                model_name=model_name,
                feature_group=feature_group,
                class_weight_mode="none",
                results=best_results,
                source=f"optuna_{len(study.trials)}_trials",
            )
            _upsert_summary(
                summary,
                [
                    {
                        "model": model_name,
                        "class_weight": "none",
                        "source": f"optuna_{len(study.trials)}_trials",
                        **r.__dict__,
                    }
                    for r in best_results
                ],
            )
            print("tuned", model_name, study.best_value, study.best_params, flush=True)

            if args.include_class_weight_pilot:
                weighted_results = evaluate_params(
                    model_name=model_name,
                    params=study.best_params,
                    features=features,
                    labels=labels,
                    dates=dates,
                    folds=folds,
                    feature_group=feature_group,
                    seed=seed,
                    class_weight_mode="balanced",
                )
                w_summary = summarize_fold_results(
                    model_name=model_name,
                    feature_group=feature_group,
                    class_weight_mode="balanced",
                    results=weighted_results,
                    source="class_weight_pilot",
                )
                _upsert_summary(
                    w_summary,
                    [
                        {
                            "model": model_name,
                            "class_weight": "balanced",
                            "source": "class_weight_pilot",
                            **r.__dict__,
                        }
                        for r in weighted_results
                    ],
                )
                print("weighted_pilot", w_summary, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
