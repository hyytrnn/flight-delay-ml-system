# V2.1 Early-Stopping & Refit Parity Audit

## 1. Code Finding & Clarification (Non-Leakage, Training Consistency)

In V2 (`src/flight_delay/models/tuning_v2.py`), Optuna trials and `evaluate_params` used `_fit_predict_onehot()` / `_evaluate_encoded_folds()` in **probe-only mode**: an inner chronological 85%/15% split (`_inner_chronological_split`) selected `best_iteration` via early stopping on the latest 15% of outer-train rows (`inner_val`), and returned predictions directly from that probe model (fitted on 85% of outer-train). Meanwhile, `scripts/evaluate_holdout_v2.py::_fit_frozen_model` refitted a fresh model on **100% of the training window** using the probe-selected `best_iteration`.

Important clarifications:
- Neither `probe_only` nor `refit_full_train` leaks outer-validation rows: preprocessors (`CategoricalFrameTransformer`, `ColumnTransformer`) fit solely inside `inner_train` (for `probe_only`) or `outer_train` (for `refit_full_train`), and outer-validation rows never enter fitting or early-stopping decisions.
- However, `probe_only` discarded the most recent 15% of pre-validation rows during fitting (e.g., `183,232` to `316,469` most recent rows per fold), creating a systematic training-data mismatch against the final frozen refit model.

## 2. Empirical Comparison Across 4 Expanding Folds (`F3`, Seed 42)

Sources: [`reports/ablation/refit_vs_probe.csv`](../reports/ablation/refit_vs_probe.csv), [`reports/ablation/refit_vs_probe_paired_ci.json`](../reports/ablation/refit_vs_probe_paired_ci.json).

### Fold-Level Summary

| Model | Fold | `rows_train` | `rows_inner_train` | `rows_inner_val` | `best_iter` | `probe_only` PR-AUC (`fit_s`) | `refit_full_train` PR-AUC (`fit_s`) |
|---|---|---:|---:|---:|---:|---:|---:|
| `xgboost` | `fold_1` | 1,200,000 | 1,016,768 | 183,232 | 257 | 0.31529 (39.3s) | **0.32120** (78.6s) |
| `xgboost` | `fold_2` | 1,500,000 | 1,274,081 | 225,919 | 287 | 0.37572 (58.6s) | **0.37678** (112.7s) |
| `xgboost` | `fold_3` | 1,800,000 | 1,527,067 | 272,933 | 156 | 0.40421 (48.5s) | **0.40866** (80.0s) |
| `xgboost` | `fold_4` | 2,100,000 | 1,783,531 | 316,469 | 500 | 0.26120 (129.6s) | **0.26443** (238.4s) |
| `lightgbm` | `fold_1` | 1,200,000 | 1,016,768 | 183,232 | 102 | 0.29855 (12.4s) | **0.30646** (23.7s) |
| `lightgbm` | `fold_2` | 1,500,000 | 1,274,081 | 225,919 | 56 | 0.37337 (11.8s) | **0.37438** (20.4s) |
| `lightgbm` | `fold_3` | 1,800,000 | 1,527,067 | 272,933 | 64 | 0.40491 (18.2s) | **0.40817** (28.2s) |
| `lightgbm` | `fold_4` | 2,100,000 | 1,783,531 | 316,469 | 86 | 0.25846 (26.1s) | **0.26132** (38.0s) |

### Paired Calendar-Block Bootstrap (`refit_full_train - probe_only`, `n_boot=300` over `1,200,000` validation rows)

- **XGBoost**: mean fold PR-AUC `0.339104 -> 0.342768`, mean Brier `0.151657 -> 0.151285`.
  Paired delta PR-AUC: **`+0.003083`** (95% CI `[+0.001598, +0.004919]`, `prob_positive = 1.000`);
  Paired delta Brier: **`-0.000370`** (95% CI `[-0.000580, -0.000170]`).
- **LightGBM**: mean fold PR-AUC `0.333825 -> 0.337582`, mean Brier `0.152160 -> 0.151745`.
  Paired delta PR-AUC: **`+0.003942`** (95% CI `[+0.002021, +0.006062]`, `prob_positive = 1.000`);
  Paired delta Brier: **`-0.000408`** (95% CI `[-0.000597, -0.000238]`).

## 3. Standard Policy Adopted for V2.1

1. During Optuna inner screening (`trials >= 10`), fast `probe_only` with `MedianPruner` may rank hyperparameter candidates without leaking outer-validation rows (~1.8x faster per trial).
2. For **every reported comparison table** (`fair_feature_model_comparison.csv`, `v2_1_model_comparison.csv`, `historical_priors_asof.csv`, `weather_v2_1.csv`) and final bundle training (`evaluate_holdout_v2_1.py`), all gradient boosters (`xgboost`, `lightgbm`, `catboost`) execute unified **`refit_full_train`**: early-stopping determines `best_iteration` on the chronological 85%/15% inner split, preprocessors fit on 100% of `outer_train`, and the model refits cleanly on 100% of `outer_train` with `rows_fit_actual == rows_train`.
