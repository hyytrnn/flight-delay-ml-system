"""Gate G3: Controlled multi-model Optuna tuning + refit comparison on SELECTED_GROUP."""

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
    cleanup_stale_running_trials,
    evaluate_params_with_oof,
    export_study,
    load_development_frame,
    load_splits,
    run_study,
)


def _oof_sha256(oof: pd.DataFrame) -> str:
    arr = np.ascontiguousarray(oof["y_prob"].to_numpy(dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _summarize(
    *,
    model: str,
    group: str,
    class_weight: str,
    source: str,
    results: list,
    oof: pd.DataFrame,
    complete_trials: int,
    pruned_trials: int,
) -> dict[str, object]:
    pr_scores = [r.pr_auc for r in results]
    return {
        "model": model,
        "feature_group": group,
        "class_weight": class_weight,
        "source": source,
        "eval_strategy": results[0].eval_strategy if results else "refit_full_train",
        "complete_trials": complete_trials,
        "pruned_trials": pruned_trials,
        "mean_pr_auc": float(np.mean(pr_scores)),
        "std_pr_auc": float(np.std(pr_scores)),
        "worst_fold_pr_auc": float(np.min(pr_scores)),
        "mean_roc_auc": float(np.mean([r.roc_auc for r in results])),
        "mean_brier": float(np.mean([r.brier for r in results])),
        "mean_log_loss": float(np.mean([r.log_loss for r in results])),
        "mean_prevalence_baseline": float(
            np.mean([r.positive_rate_validation for r in results])
        ),
        "mean_fit_seconds": float(np.mean([r.fit_seconds for r in results])),
        "oof_predictions_sha256": _oof_sha256(oof),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tuning_v2_1.yaml"))
    parser.add_argument("--feature-group", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.feature_group is None:
        sel_path = Path("configs/best_v2_1/selection_summary.json")
        group_str = (
            json.loads(sel_path.read_text(encoding="utf-8"))["selected_feature_group"]
            if sel_path.exists()
            else "F0"
        )
    else:
        group_str = args.feature_group

    feature_group: FeatureGroup = group_str  # type: ignore[assignment]
    seed = int(cfg["seed"])
    storage = str(cfg["storage"])
    budgets = cfg["budgets"]

    dev = load_development_frame(
        Path(cfg["data_root"]), date.fromisoformat(str(cfg["development_end_exclusive"]))
    )
    folds = load_splits(Path(cfg["splits"]))
    features = build_v2_feature_frame(dev, feature_group)
    labels = dev.get_column("dep_del15").to_numpy().astype(int)
    dates = pd.to_datetime(dev.get_column("flight_date").to_pandas())

    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    oof_map: dict[str, pd.DataFrame] = {}

    out_optuna = Path("reports/optuna_v2_1")
    out_optuna.mkdir(parents=True, exist_ok=True)
    Path("configs/best_v2_1").mkdir(parents=True, exist_ok=True)

    # 1. XGBoost: reuse fair study if already populated and expand up to xgboost_total_trials
    fair_study_name = f"v2_1_fair_xgboost_{feature_group}_none_probe_only"
    cleanup_stale_running_trials(storage, fair_study_name)
    existing_n = 0
    try:
        st_exist = optuna.load_study(study_name=fair_study_name, storage=storage)
        existing_n = sum(
            t.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
            for t in st_exist.trials
        )
    except KeyError:
        existing_n = 0
    needed_xgb = max(0, int(budgets["xgboost_total_trials"]) - existing_n)
    prefix_xgb = "v2_1_fair" if existing_n > 0 else "v2_1_optuna"
    xgb_study = run_study(
        model_name="xgboost",
        frame=dev,
        feature_group=feature_group,
        folds=folds,
        storage=storage,
        trials=needed_xgb,
        timeout=int(budgets["xgboost_timeout_seconds"]),
        seed=seed,
        class_weight_mode="none",
        study_name_prefix=prefix_xgb,
        eval_strategy="probe_only",
        use_pruner=True,
    )
    export_study(xgb_study, out_optuna, f"xgboost_{feature_group}")
    xgb_params = dict(xgb_study.best_params)
    Path(f"configs/best_v2_1/xgboost_{feature_group}_final.yaml").write_text(
        yaml.safe_dump(xgb_params, sort_keys=True), encoding="utf-8"
    )
    xgb_comp = sum(t.state == optuna.trial.TrialState.COMPLETE for t in xgb_study.trials)
    xgb_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in xgb_study.trials)
    xgb_res, xgb_oof = evaluate_params_with_oof(
        model_name="xgboost",
        params=xgb_params,
        features=features,
        labels=labels,
        dates=dates,
        folds=folds,
        feature_group=feature_group,
        seed=seed,
        eval_strategy="refit_full_train",
    )
    oof_map["xgboost"] = xgb_oof
    summary_rows.append(
        _summarize(
            model="xgboost",
            group=feature_group,
            class_weight="none",
            source=f"optuna_{len(xgb_study.trials)}_trials",
            results=xgb_res,
            oof=xgb_oof,
            complete_trials=xgb_comp,
            pruned_trials=xgb_pruned,
        )
    )
    fold_rows.extend({"model": "xgboost", "class_weight": "none", **asdict(r)} for r in xgb_res)
    print("XGBOOST", summary_rows[-1], flush=True)

    # 2. LightGBM: controlled Optuna search
    lgb_study = run_study(
        model_name="lightgbm",
        frame=dev,
        feature_group=feature_group,
        folds=folds,
        storage=storage,
        trials=int(budgets["lightgbm_trials"]),
        timeout=int(budgets["lightgbm_timeout_seconds"]),
        seed=seed,
        class_weight_mode="none",
        study_name_prefix="v2_1_optuna",
        eval_strategy="probe_only",
        use_pruner=True,
    )
    export_study(lgb_study, out_optuna, f"lightgbm_{feature_group}")
    lgb_params = dict(lgb_study.best_params)
    Path(f"configs/best_v2_1/lightgbm_{feature_group}_final.yaml").write_text(
        yaml.safe_dump(lgb_params, sort_keys=True), encoding="utf-8"
    )
    lgb_comp = sum(t.state == optuna.trial.TrialState.COMPLETE for t in lgb_study.trials)
    lgb_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in lgb_study.trials)
    lgb_res, lgb_oof = evaluate_params_with_oof(
        model_name="lightgbm",
        params=lgb_params,
        features=features,
        labels=labels,
        dates=dates,
        folds=folds,
        feature_group=feature_group,
        seed=seed,
        eval_strategy="refit_full_train",
    )
    oof_map["lightgbm"] = lgb_oof
    summary_rows.append(
        _summarize(
            model="lightgbm",
            group=feature_group,
            class_weight="none",
            source=f"optuna_{len(lgb_study.trials)}_trials",
            results=lgb_res,
            oof=lgb_oof,
            complete_trials=lgb_comp,
            pruned_trials=lgb_pruned,
        )
    )
    fold_rows.extend({"model": "lightgbm", "class_weight": "none", **asdict(r)} for r in lgb_res)
    print("LIGHTGBM", summary_rows[-1], flush=True)

    # 3. CatBoost: CPU-bounded Optuna search with explicit clean completion
    cat_study = run_study(
        model_name="catboost",
        frame=dev,
        feature_group=feature_group,
        folds=folds,
        storage=storage,
        trials=int(budgets["catboost_trials"]),
        timeout=int(budgets["catboost_timeout_seconds"]),
        seed=seed,
        class_weight_mode="none",
        study_name_prefix="v2_1_optuna",
        eval_strategy="probe_only",
        use_pruner=True,
    )
    export_study(cat_study, out_optuna, f"catboost_{feature_group}")
    cat_params = dict(cat_study.best_params)
    Path(f"configs/best_v2_1/catboost_{feature_group}_final.yaml").write_text(
        yaml.safe_dump(cat_params, sort_keys=True), encoding="utf-8"
    )
    cat_comp = sum(t.state == optuna.trial.TrialState.COMPLETE for t in cat_study.trials)
    cat_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in cat_study.trials)
    cat_res, cat_oof = evaluate_params_with_oof(
        model_name="catboost",
        params=cat_params,
        features=features,
        labels=labels,
        dates=dates,
        folds=folds,
        feature_group=feature_group,
        seed=seed,
        eval_strategy="refit_full_train",
    )
    oof_map["catboost"] = cat_oof
    summary_rows.append(
        _summarize(
            model="catboost",
            group=feature_group,
            class_weight="none",
            source=f"optuna_{len(cat_study.trials)}_trials",
            results=cat_res,
            oof=cat_oof,
            complete_trials=cat_comp,
            pruned_trials=cat_pruned,
        )
    )
    fold_rows.extend({"model": "catboost", "class_weight": "none", **asdict(r)} for r in cat_res)
    print("CATBOOST", summary_rows[-1], flush=True)

    # 4. Baselines + XGBoost balanced class-weight check on SELECTED_GROUP
    for b_model in ["dummy_prior", "logistic_regression", "hist_gradient_boosting"]:
        b_res, b_oof = evaluate_params_with_oof(
            model_name=b_model,  # type: ignore[arg-type]
            params={},
            features=features,
            labels=labels,
            dates=dates,
            folds=folds,
            feature_group=feature_group,
            seed=seed,
        )
        oof_map[b_model] = b_oof
        summary_rows.append(
            _summarize(
                model=b_model,
                group=feature_group,
                class_weight="none",
                source="baseline",
                results=b_res,
                oof=b_oof,
                complete_trials=0,
                pruned_trials=0,
            )
        )
        fold_rows.extend({"model": b_model, "class_weight": "none", **asdict(r)} for r in b_res)

    bal_res, bal_oof = evaluate_params_with_oof(
        model_name="xgboost",
        params=xgb_params,
        features=features,
        labels=labels,
        dates=dates,
        folds=folds,
        feature_group=feature_group,
        seed=seed,
        class_weight_mode="balanced",
        eval_strategy="refit_full_train",
    )
    oof_map["xgboost_balanced"] = bal_oof
    summary_rows.append(
        _summarize(
            model="xgboost",
            group=feature_group,
            class_weight="balanced",
            source="class_weight_check",
            results=bal_res,
            oof=bal_oof,
            complete_trials=0,
            pruned_trials=0,
        )
    )
    fold_rows.extend({"model": "xgboost", "class_weight": "balanced", **asdict(r)} for r in bal_res)

    # 5. Paired calendar-block bootstrap comparisons vs XGBoost
    paired_cis: dict[str, object] = {}
    for other_key in ["lightgbm", "catboost", "hist_gradient_boosting", "logistic_regression", "xgboost_balanced"]:
        paired_cis[f"xgboost_minus_{other_key}"] = paired_calendar_block_bootstrap(
            oof_map[other_key]["y_true"].to_numpy(),
            oof_map[other_key]["y_prob"].to_numpy(),
            oof_map["xgboost"]["y_prob"].to_numpy(),
            oof_map["xgboost"]["flight_date"],
            n_bootstrap=300,
            seed=seed,
        )

    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comp_df = pd.DataFrame(summary_rows).sort_values("mean_pr_auc", ascending=False)
    comp_df.to_csv(metrics_dir / "v2_1_model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(metrics_dir / "v2_1_model_comparison_folds.csv", index=False)
    (metrics_dir / "v2_1_model_paired_ci.json").write_text(
        json.dumps(paired_cis, indent=2), encoding="utf-8"
    )
    winner_row = comp_df.loc[comp_df["class_weight"] == "none"].iloc[0]
    winner_model = str(winner_row["model"])
    if winner_model in oof_map:
        Path("data/interim").mkdir(parents=True, exist_ok=True)
        oof_map[winner_model].to_parquet("data/interim/v2_1_champion_oof_2024.parquet", index=False)
    cleanup_stale_running_trials(storage)
    print(comp_df[["model", "class_weight", "source", "mean_pr_auc", "worst_fold_pr_auc", "mean_brier"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
