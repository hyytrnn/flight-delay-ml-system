# Model Card: Flight Delay V2 (`flight-only-v2.0.0-xgboost`)

## Model Details

- **Model ID**: `flight-only-v2.0.0-xgboost`
- **Contract**: `t24-v2-F3` (binary `dep_del15` departure delay `>= 15` minutes, predicted `24` hours before scheduled departure)
- **Estimator**: `Pipeline(CategoricalFrameTransformer(min_frequency=100, max_categories=300), XGBClassifier(tree_method="hist", enable_categorical=True, max_cat_to_onehot=16))`
- **Hyperparameters** ([`reports/optuna/xgboost_best_params.yaml`](optuna/xgboost_best_params.yaml)): `n_estimators=500` (with inner chronological early stopping), `max_depth=5`, `learning_rate=0.02196`, `subsample=0.7097`, `colsample_bytree=0.6728`, `min_child_weight=13`, `gamma=7.591`, `reg_alpha=1.1015`, `reg_lambda=67.3225`.
- **Calibration & Threshold**: `calibration_method = none` (raw log-loss XGBoost probabilities selected over Platt and Isotonic on `2025-02..2025-03`); classification threshold `0.185` (`max_f1` on `2025-04..2025-05`).
- **Weather Status**: `not_used_flight_only_v2` (flight-only T-24h serving contract; optional matched weather track documented in `reports/ablation/weather_comparison.csv`).

## Data Windows

- **Training**: `2023-01-01..2025-01-31` (`2,500,000` sampled BTS rows across 25 continuous months; positive rate `0.2041`)
- **Calibration**: `2025-02-01..2025-03-31` (`200,000` rows)
- **Threshold Selection**: `2025-04-01..2025-05-31` (`200,000` rows, positive rate `0.2061`)
- **Pre-Holdout Validation**: `2025-06-01..2025-08-31` (`300,000` rows, positive rate `0.2605`)
- **Final Untouched Holdout (Evaluated Once)**: `2025-09-01..2025-09-30` (`100,000` rows, positive rate `0.16545`)

## Quantitative Performance

| Window | Rows | Prevalence | PR-AUC | Uplift vs Prev | ROC-AUC | F1 (@thr) | Brier | ECE | P@Top5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2025-01` Ref (`thr=0.5`) | 100,000 | 0.1829 | 0.2396 | +0.0566 (+31.0%) | 0.6193 | — | 0.1498 | 0.0353 | 0.2578 |
| `2025-04..05` (`thr=0.185`) | 200,000 | 0.2061 | 0.3322 | +0.1261 (+61.2%) | 0.6718 | 0.4074 | 0.1539 | 0.0072 | 0.4259 |
| `2025-06..08` (`thr=0.185`) | 300,000 | 0.2605 | 0.4419 | +0.1815 (+69.7%) | 0.7092 | 0.4873 | 0.1731 | 0.0207 | 0.5598 |
| **`2025-09` Holdout (`thr=0.185`)** | **100,000** | **0.1655** | **0.2950** | **+0.1296 (+78.3%)** | **0.6868** | **0.3657** | **0.1298** | **0.0139** | **0.3952** |

Holdout 95% calendar-block-bootstrap CIs (`n_boot=300`, [`reports/holdout/v2_holdout_metrics.json`](holdout/v2_holdout_metrics.json)): PR-AUC `[0.2661, 0.3169]`, ROC-AUC `[0.6726, 0.6993]`, Brier `[0.1211, 0.1363]`, F1 `[0.3365, 0.3847]`.

### Holdout Slices (`2025-09`)

- Known routes (`99,341` rows, `0.1655` prev): `PR-AUC = 0.2954`, `ROC-AUC = 0.6869`, `P@Top5% = 0.3956`
- Unseen routes (`659` rows, `0.1578` prev): `PR-AUC = 0.2494`, `ROC-AUC = 0.6776`, `P@Top5% = 0.3333`
- Top airlines by volume: `WN` (`PR-AUC=0.3836`), `AA` (`0.3372`), `UA` (`0.2353`), `DL` (`0.2285`), `YX` (`0.2896`).

## Top Feature Drivers & Explainability

TreeSHAP (`pred_contribs=True`, `n=2,000`) and grouped permutation PR-AUC drop (`reports/explainability/`):

1. `dep_hour_sin` (mean `|SHAP|=0.3829`, PR-AUC drop `0.1061`)
2. `carrier_origin` (mean `|SHAP|=0.1126`, PR-AUC drop `0.0237`)
3. `reporting_airline` (mean `|SHAP|=0.1115`, PR-AUC drop `0.0212`)
4. `day_of_year_cos` / `dep_hour_cos` / `origin_hour` / `dest` / `weekday_sin`

## Limitations & Responsible Use

- Flight-only schedule priors cannot foresee sudden convective storms or real-time ground stops without live weather (`F5` track) and FAA Air Traffic Control advisories.
- Seasonal base-rate shifts (`16.5%` autumn vs `26.0%` summer) affect fixed-threshold precision/recall even when ranking ROC-AUC and calibration remain stable; pair fixed `0.185` classification with `Precision@TopK` alert budgets.
- Unknown origin timezones fail closed (`422 Unprocessable Entity` on V2 serving bundles) rather than silently falling back to UTC.
