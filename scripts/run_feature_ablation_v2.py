"""Run fixed-model feature-group ablations on identical temporal folds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame
from flight_delay.models.tuning_v2 import (
    evaluate_params,
    load_development_frame,
    load_splits,
)

XGBOOST_REFERENCE_PARAMS = {
    "n_estimators": 280,
    "max_depth": 6,
    "learning_rate": 0.08,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.05,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_v2.yaml"))
    parser.add_argument("--end-exclusive", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument(
        "--groups", nargs="+", default=["F0", "F1", "F2", "F3", "P2_PROXY"]
    )
    parser.add_argument("--output", type=Path, default=Path("reports/ablation/features_v2.csv"))
    parser.add_argument(
        "--fold-output", type=Path, default=Path("reports/ablation/features_v2_folds.csv")
    )
    args = parser.parse_args()

    frame = load_development_frame(args.data_root, args.end_exclusive)
    folds = load_splits(args.splits)
    labels = frame.get_column("dep_del15").to_numpy().astype(int)
    dates = pd.to_datetime(frame.get_column("flight_date").to_pandas())
    summaries = []
    fold_rows = []
    for group_text in args.groups:
        group: FeatureGroup = group_text
        features = build_v2_feature_frame(frame, group)
        results = evaluate_params(
            model_name="xgboost",
            params=XGBOOST_REFERENCE_PARAMS,
            features=features,
            labels=labels,
            dates=dates,
            folds=folds,
            feature_group=group,
            seed=42,
        )
        scores = [result.pr_auc for result in results]
        summaries.append(
            {
                "feature_group": group,
                "feature_count": features.shape[1],
                "mean_pr_auc": float(np.mean(scores)),
                "std_pr_auc": float(np.std(scores)),
                "worst_fold_pr_auc": float(np.min(scores)),
                "mean_brier": float(np.mean([result.brier for result in results])),
                "mean_fit_seconds": float(
                    np.mean([result.fit_seconds for result in results])
                ),
                "production_eligibility": (
                    "retrospective_proxy" if group == "P2_PROXY" else "candidate"
                ),
            }
        )
        fold_rows.extend(
            {"feature_group": group, **asdict(result)} for result in results
        )
        print(group, summaries[-1])

    summary_frame = pd.DataFrame(summaries).sort_values("mean_pr_auc", ascending=False)
    baseline = float(
        summary_frame.loc[summary_frame["feature_group"] == "F0", "mean_pr_auc"].iloc[0]
    )
    summary_frame["delta_vs_f0"] = summary_frame["mean_pr_auc"] - baseline
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.output, index=False)
    pd.DataFrame(fold_rows).to_csv(args.fold_output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

