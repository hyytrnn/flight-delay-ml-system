"""Audit inner early-stopping probe_only vs full-outer-train refit across 4 temporal folds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame
from flight_delay.models.calibrate_v2 import paired_calendar_block_bootstrap
from flight_delay.models.tuning_v2 import (
    EvalStrategy,
    _evaluate_encoded_folds,
    _prepare_encoded_folds,
    load_development_frame,
    load_splits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_v2_1.yaml"))
    parser.add_argument("--end-exclusive", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--feature-group", type=str, default="F3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feature_group: FeatureGroup = args.feature_group  # type: ignore[assignment]
    dev = load_development_frame(args.data_root, args.end_exclusive)
    folds = load_splits(args.splits)
    features = build_v2_feature_frame(dev, feature_group)
    labels = dev.get_column("dep_del15").to_numpy().astype(int)
    dates = pd.to_datetime(dev.get_column("flight_date").to_pandas())

    candidates: list[tuple[Literal["xgboost", "lightgbm"], Path]] = [
        ("xgboost", Path("reports/optuna/xgboost_best_params.yaml")),
        ("lightgbm", Path("reports/optuna/lightgbm_best_params.yaml")),
    ]

    all_rows: list[dict[str, object]] = []
    paired_summary: dict[str, object] = {}

    for model_name, params_path in candidates:
        params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
        print(f"Preparing cached folds for {model_name}...", flush=True)
        encoded_folds = _prepare_encoded_folds(
            model_name=model_name,
            features=features,
            labels=labels,
            dates=dates,
            folds=folds,
            feature_group=feature_group,
            class_weight_mode="none",
            include_full_refit=True,
        )
        oof_by_strategy: dict[str, pd.DataFrame] = {}
        for strategy in ["probe_only", "refit_full_train"]:
            strat: EvalStrategy = strategy  # type: ignore[assignment]
            results, oof = _evaluate_encoded_folds(
                model_name=model_name,
                params=params,
                encoded_folds=encoded_folds,
                seed=args.seed,
                early_stopping_rounds=50,
                eval_strategy=strat,
            )
            oof_by_strategy[strategy] = oof
            for res in results:
                all_rows.append(
                    {"model": model_name, "feature_group": feature_group, **asdict(res)}
                )
            print(
                f"{model_name} [{strategy}]: "
                f"mean_pr_auc={sum(r.pr_auc for r in results) / len(results):.6f} "
                f"mean_brier={sum(r.brier for r in results) / len(results):.6f} "
                f"mean_fit_s={sum(r.fit_seconds for r in results) / len(results):.2f}",
                flush=True,
            )

        base_oof = oof_by_strategy["probe_only"]
        cand_oof = oof_by_strategy["refit_full_train"]
        paired_summary[model_name] = paired_calendar_block_bootstrap(
            base_oof["y_true"].to_numpy(),
            base_oof["y_prob"].to_numpy(),
            cand_oof["y_prob"].to_numpy(),
            cand_oof["flight_date"],
            n_bootstrap=300,
            seed=args.seed,
        )
        print(f"{model_name} paired delta PR-AUC CI:", paired_summary[model_name]["delta_pr_auc"], flush=True)

    out_csv = Path("reports/ablation/refit_vs_probe.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    out_json = Path("reports/ablation/refit_vs_probe_paired_ci.json")
    out_json.write_text(json.dumps(paired_summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
