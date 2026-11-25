"""Gate G5: Evaluate strict publication-lagged BTS historical delay priors on temporal folds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flight_delay.features.historical_priors import (
    HISTORICAL_PRIOR_COLUMNS,
    compute_asof_historical_priors,
)
from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame, v2_feature_columns
from flight_delay.models.calibrate_v2 import paired_calendar_block_bootstrap
from flight_delay.models.tuning_v2 import (
    XGBOOST_FIXED_PARAMS,
    evaluate_params_with_oof,
    load_development_frame,
    load_splits,
)


def _eval_group_with_per_fold_priors(
    *,
    variant: str,
    group: FeatureGroup,
    dev_pl,
    dev_pd: pd.DataFrame,
    base_features: pd.DataFrame,
    labels: np.ndarray,
    dates: pd.Series,
    folds: list,
    params: dict[str, object],
    alpha: float | None,
    min_support: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    del dev_pl
    if alpha is None:
        results, oof = evaluate_params_with_oof(
            model_name="xgboost",
            params=params,
            features=base_features,
            labels=labels,
            dates=dates,
            folds=folds,
            feature_group=group,
            seed=seed,
            use_inner_early_stopping=True,
            eval_strategy="refit_full_train",
        )
        fold_rows = [{"variant": variant, "feature_group": group, **asdict(r)} for r in results]
    else:
        results = []
        fold_rows = []
        oof_parts: list[pd.DataFrame] = []
        for fold in folds:
            tr_mask = (dates >= pd.Timestamp(fold.train_start)) & (
                dates < pd.Timestamp(fold.train_end_exclusive)
            )
            va_mask = (dates >= pd.Timestamp(fold.validation_start)) & (
                dates < pd.Timestamp(fold.validation_end_exclusive)
            )
            both_mask = tr_mask | va_mask
            priors_df = compute_asof_historical_priors(
                train_history=dev_pd.loc[tr_mask],
                target_frame=dev_pd.loc[both_mask],
                alpha=alpha,
                min_support=min_support,
                publication_lag_months=2,
            )
            augmented = pd.concat(
                [base_features.loc[both_mask], priors_df[HISTORICAL_PRIOR_COLUMNS]], axis=1
            )
            sub_res, sub_oof = evaluate_params_with_oof(
                model_name="xgboost",
                params=params,
                features=augmented,
                labels=labels[both_mask],
                dates=dates.loc[both_mask],
                folds=[fold],
                feature_group=group,
                seed=seed,
                use_inner_early_stopping=True,
                eval_strategy="refit_full_train",
                extra_numeric=HISTORICAL_PRIOR_COLUMNS,
            )
            results.extend(sub_res)
            oof_parts.append(sub_oof)
            fold_rows.append({"variant": variant, "feature_group": group, **asdict(sub_res[0])})
        oof = pd.concat(oof_parts, ignore_index=True)

    pr_scores = [r.pr_auc for r in results]
    num_c, cat_c = v2_feature_columns(group)
    n_cols = len(num_c) + len(cat_c) + (len(HISTORICAL_PRIOR_COLUMNS) if alpha is not None else 0)
    summary = {
        "variant": variant,
        "feature_group": group,
        "alpha": alpha if alpha is not None else np.nan,
        "min_support": min_support if alpha is not None else 0,
        "publication_lag_months": 2 if alpha is not None else 0,
        "feature_count": n_cols,
        "mean_pr_auc": float(np.mean(pr_scores)),
        "std_pr_auc": float(np.std(pr_scores)),
        "worst_fold_pr_auc": float(np.min(pr_scores)),
        "mean_roc_auc": float(np.mean([r.roc_auc for r in results])),
        "mean_brier": float(np.mean([r.brier for r in results])),
        "mean_fit_seconds": float(np.mean([r.fit_seconds for r in results])),
        "production_status": (
            "static_t24_eligible"
            if alpha is None
            else "yellow_conservative_lag_policy_research_only"
        ),
    }
    return summary, fold_rows, oof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_v2_1.yaml"))
    parser.add_argument("--end-exclusive", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--selected-group", type=str, default=None)
    parser.add_argument("--params-yaml", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.selected_group is None:
        sel_path = Path("configs/best_v2_1/selection_summary.json")
        if sel_path.exists():
            args.selected_group = json.loads(sel_path.read_text(encoding="utf-8"))[
                "selected_feature_group"
            ]
        else:
            args.selected_group = "F0"

    selected_group: FeatureGroup = args.selected_group  # type: ignore[assignment]
    if args.params_yaml and args.params_yaml.exists():
        sel_params = yaml.safe_load(args.params_yaml.read_text(encoding="utf-8"))
    else:
        sel_params = XGBOOST_FIXED_PARAMS

    dev = load_development_frame(args.data_root, args.end_exclusive)
    folds = load_splits(args.splits)
    dev_pd = dev.select(
        ["flight_date", "prediction_at_utc", "scheduled_departure_utc", "origin", "dest", "reporting_airline", "dep_del15"]
    ).to_pandas()
    labels = dev.get_column("dep_del15").to_numpy().astype(int)
    dates = pd.to_datetime(dev.get_column("flight_date").to_pandas())

    summaries: list[dict[str, object]] = []
    all_fold_rows: list[dict[str, object]] = []
    paired_cis: dict[str, object] = {}

    group_configs: list[tuple[FeatureGroup, dict[str, object], list[tuple[float, int]]]] = [
        ("F0", XGBOOST_FIXED_PARAMS, [(100.0, 30)]),
    ]
    if selected_group != "F0":
        group_configs.append((selected_group, sel_params, [(50.0, 20), (200.0, 50)]))
    else:
        group_configs[0] = ("F0", sel_params, [(50.0, 20), (200.0, 50)])

    for grp, grp_params, alpha_grid in group_configs:
        base_feats = build_v2_feature_frame(dev, grp)
        base_name = f"{grp}_baseline"
        b_summ, b_folds, b_oof = _eval_group_with_per_fold_priors(
            variant=base_name,
            group=grp,
            dev_pl=dev,
            dev_pd=dev_pd,
            base_features=base_feats,
            labels=labels,
            dates=dates,
            folds=folds,
            params=grp_params,
            alpha=None,
            min_support=0,
            seed=args.seed,
        )
        b_summ["paired_delta_pr_auc"] = 0.0
        b_summ["paired_ci95_low"] = 0.0
        b_summ["paired_ci95_high"] = 0.0
        summaries.append(b_summ)
        all_fold_rows.extend(b_folds)
        print(base_name, b_summ["mean_pr_auc"], b_summ["mean_brier"], flush=True)

        for alpha, min_sup in alpha_grid:
            var_name = f"{grp}_plus_hist_priors_a{int(alpha)}_m{min_sup}"
            c_summ, c_folds, c_oof = _eval_group_with_per_fold_priors(
                variant=var_name,
                group=grp,
                dev_pl=dev,
                dev_pd=dev_pd,
                base_features=base_feats,
                labels=labels,
                dates=dates,
                folds=folds,
                params=grp_params,
                alpha=alpha,
                min_support=min_sup,
                seed=args.seed,
            )
            ci = paired_calendar_block_bootstrap(
                b_oof["y_true"].to_numpy(),
                b_oof["y_prob"].to_numpy(),
                c_oof["y_prob"].to_numpy(),
                c_oof["flight_date"],
                n_bootstrap=300,
                seed=args.seed,
            )
            paired_cis[f"{var_name}_minus_{base_name}"] = ci
            c_summ["paired_delta_pr_auc"] = ci["delta_pr_auc"]["mean"]
            c_summ["paired_ci95_low"] = ci["delta_pr_auc"]["ci95_low"]
            c_summ["paired_ci95_high"] = ci["delta_pr_auc"]["ci95_high"]
            summaries.append(c_summ)
            all_fold_rows.extend(c_folds)
            print(
                var_name,
                c_summ["mean_pr_auc"],
                "delta=",
                c_summ["paired_delta_pr_auc"],
                c_summ["mean_brier"],
                flush=True,
            )

    out_dir = Path("reports/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out_dir / "historical_priors_asof.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "historical_asof_ablation.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(out_dir / "historical_priors_asof_folds.csv", index=False)
    (out_dir / "historical_priors_paired_ci.json").write_text(
        json.dumps(paired_cis, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
