"""Gate G1: Equal-budget, identical-fold feature group ablation (F0/F1/F2/F3) with paired CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame
from flight_delay.models.calibrate_v2 import paired_calendar_block_bootstrap
from flight_delay.models.tuning_v2 import (
    _evaluate_encoded_folds,
    _prepare_encoded_folds,
    export_study,
    load_development_frame,
    load_splits,
    run_study,
)

XGBOOST_FIXED_PARAMS = {
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


def _oof_sha256(oof: pd.DataFrame) -> str:
    arr = np.ascontiguousarray(oof["y_prob"].to_numpy(dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _eval_group_refit(
    *,
    stage: str,
    group: FeatureGroup,
    params: dict[str, object],
    encoded_folds: list,
    feature_count: int,
    seed: int,
    complete_trials: int,
    pruned_trials: int,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    results, oof = _evaluate_encoded_folds(
        model_name="xgboost",
        params=params,
        encoded_folds=encoded_folds,
        seed=seed,
        early_stopping_rounds=40,
        eval_strategy="refit_full_train",
    )
    pr_scores = [r.pr_auc for r in results]
    summary = {
        "stage": stage,
        "feature_group": group,
        "feature_count": feature_count,
        "model": "xgboost",
        "eval_strategy": "refit_full_train",
        "complete_trials": complete_trials,
        "pruned_trials": pruned_trials,
        "mean_pr_auc": float(np.mean(pr_scores)),
        "std_pr_auc": float(np.std(pr_scores)),
        "worst_fold_pr_auc": float(np.min(pr_scores)),
        "mean_roc_auc": float(np.mean([r.roc_auc for r in results])),
        "mean_brier": float(np.mean([r.brier for r in results])),
        "mean_fit_seconds": float(np.mean([r.fit_seconds for r in results])),
        "oof_predictions_sha256": _oof_sha256(oof),
    }
    fold_rows = [{"stage": stage, "feature_group": group, **asdict(r)} for r in results]
    return summary, fold_rows, oof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_v2_1.yaml"))
    parser.add_argument("--end-exclusive", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///reports/optuna_v2_1/v2_1_study.db",
    )
    parser.add_argument("--trials-per-group", type=int, default=10)
    parser.add_argument("--timeout-per-group", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = load_development_frame(args.data_root, args.end_exclusive)
    folds = load_splits(args.splits)
    labels = dev.get_column("dep_del15").to_numpy().astype(int)
    dates = pd.to_datetime(dev.get_column("flight_date").to_pandas())

    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    paired_ci: dict[str, object] = {}
    encoded_cache: dict[str, tuple[int, list]] = {}

    # Stage 1: Fixed hyperparameters across F0, F1, F2, F3 (with refit_full_train)
    fixed_oof: dict[str, pd.DataFrame] = {}
    for g_name in ["F0", "F1", "F2", "F3"]:
        group: FeatureGroup = g_name  # type: ignore[assignment]
        feats = build_v2_feature_frame(dev, group)
        encoded = _prepare_encoded_folds(
            model_name="xgboost",
            features=feats,
            labels=labels,
            dates=dates,
            folds=folds,
            feature_group=group,
            class_weight_mode="none",
            include_full_refit=True,
        )
        encoded_cache[g_name] = (feats.shape[1], encoded)
        summ, f_list, oof = _eval_group_refit(
            stage="fixed_params",
            group=group,
            params=XGBOOST_FIXED_PARAMS,
            encoded_folds=encoded,
            feature_count=feats.shape[1],
            seed=args.seed,
            complete_trials=0,
            pruned_trials=0,
        )
        fixed_oof[g_name] = oof
        fold_rows.extend(f_list)
        if g_name == "F0":
            summ["paired_delta_pr_auc_vs_F0"] = 0.0
            summ["paired_ci95_low"] = 0.0
            summ["paired_ci95_high"] = 0.0
        else:
            ci = paired_calendar_block_bootstrap(
                fixed_oof["F0"]["y_true"].to_numpy(),
                fixed_oof["F0"]["y_prob"].to_numpy(),
                oof["y_prob"].to_numpy(),
                oof["flight_date"],
                n_bootstrap=300,
                seed=args.seed,
            )
            paired_ci[f"fixed_{g_name}_minus_F0"] = ci
            summ["paired_delta_pr_auc_vs_F0"] = ci["delta_pr_auc"]["mean"]
            summ["paired_ci95_low"] = ci["delta_pr_auc"]["ci95_low"]
            summ["paired_ci95_high"] = ci["delta_pr_auc"]["ci95_high"]
        summary_rows.append(summ)
        print("FIXED", g_name, summ["mean_pr_auc"], summ["worst_fold_pr_auc"], flush=True)

    # Stage 2: Equal-budget Optuna (probe_only screening + full-train finalist refit) for F0, F2, F3
    Path("configs/best_v2_1").mkdir(parents=True, exist_ok=True)
    tuned_oof: dict[str, pd.DataFrame] = {}
    for g_name in ["F0", "F2", "F3"]:
        group = g_name  # type: ignore[assignment]
        study = run_study(
            model_name="xgboost",
            frame=dev,
            feature_group=group,
            folds=folds,
            storage=args.storage,
            trials=args.trials_per_group,
            timeout=args.timeout_per_group,
            seed=args.seed,
            class_weight_mode="none",
            study_name_prefix="v2_1_fair",
            eval_strategy="probe_only",
            use_pruner=True,
        )
        export_study(study, Path("reports/optuna_v2_1") / f"fair_{g_name}", f"xgboost_{g_name}")
        Path(f"configs/best_v2_1/xgboost_{g_name}.yaml").write_text(
            yaml.safe_dump(study.best_params, sort_keys=True), encoding="utf-8"
        )
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        fcount, encoded = encoded_cache[g_name]
        summ, f_list, oof = _eval_group_refit(
            stage="optuna_equal_budget",
            group=group,
            params=study.best_params,
            encoded_folds=encoded,
            feature_count=fcount,
            seed=args.seed,
            complete_trials=completed,
            pruned_trials=pruned,
        )
        tuned_oof[g_name] = oof
        fold_rows.extend(f_list)
        if g_name == "F0":
            summ["paired_delta_pr_auc_vs_F0"] = 0.0
            summ["paired_ci95_low"] = 0.0
            summ["paired_ci95_high"] = 0.0
        else:
            ci = paired_calendar_block_bootstrap(
                tuned_oof["F0"]["y_true"].to_numpy(),
                tuned_oof["F0"]["y_prob"].to_numpy(),
                oof["y_prob"].to_numpy(),
                oof["flight_date"],
                n_bootstrap=300,
                seed=args.seed,
            )
            paired_ci[f"tuned_{g_name}_minus_F0"] = ci
            summ["paired_delta_pr_auc_vs_F0"] = ci["delta_pr_auc"]["mean"]
            summ["paired_ci95_low"] = ci["delta_pr_auc"]["ci95_low"]
            summ["paired_ci95_high"] = ci["delta_pr_auc"]["ci95_high"]
        summary_rows.append(summ)
        print("TUNED", g_name, summ["mean_pr_auc"], summ["worst_fold_pr_auc"], flush=True)

    paired_ci["tuned_F3_minus_F2"] = paired_calendar_block_bootstrap(
        tuned_oof["F2"]["y_true"].to_numpy(),
        tuned_oof["F2"]["y_prob"].to_numpy(),
        tuned_oof["F3"]["y_prob"].to_numpy(),
        tuned_oof["F3"]["flight_date"],
        n_bootstrap=300,
        seed=args.seed,
    )

    summary_df = pd.DataFrame(summary_rows)
    out_dir = Path("reports/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "fair_feature_model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        out_dir / "fair_feature_model_comparison_folds.csv", index=False
    )
    (out_dir / "fair_feature_paired_ci.json").write_text(
        json.dumps(paired_ci, indent=2), encoding="utf-8"
    )

    tuned_df = summary_df.loc[summary_df["stage"] == "optuna_equal_budget"].copy()
    f0_row = tuned_df.loc[tuned_df["feature_group"] == "F0"].iloc[0]
    eligible = tuned_df[
        (tuned_df["feature_group"] == "F0")
        | (
            (tuned_df["mean_pr_auc"] >= float(f0_row["mean_pr_auc"]) + 0.0005)
            & (tuned_df["worst_fold_pr_auc"] >= float(f0_row["worst_fold_pr_auc"]) - 0.0020)
            & (tuned_df["mean_brier"] <= float(f0_row["mean_brier"]) + 0.0005)
        )
    ].sort_values("mean_pr_auc", ascending=False)
    selected_group = str(eligible.iloc[0]["feature_group"])
    Path("configs/best_v2_1/selection_summary.json").write_text(
        json.dumps(
            {
                "selected_feature_group": selected_group,
                "rule": ">= +0.0005 mean_pr_auc vs F0, worst_fold within -0.0020, brier within +0.0005; else F0 parsimony",
                "tuned_candidates": tuned_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("SELECTED_FEATURE_GROUP:", selected_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
