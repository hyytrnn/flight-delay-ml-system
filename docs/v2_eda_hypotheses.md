# V2 EDA Hypotheses and Empirical Status

Each hypothesis is falsifiable and evaluated with support counts, 95% Wilson intervals,
and rolling temporal folds (`2023-01` through `2024-12`). The September 2025 final
holdout (`2025-09`) remained closed throughout EDA and feature ablation.

| ID | Hypothesis | Observed EDA & Ablation Evidence | Verdict | T-24h status |
|---|---|---|---|---|
| H1 | Late local departures have higher delay risk across seasons | Delay rate increases monotonically from `7.52%` at 05:00 (`n=91,611`) to `32.33%` at 20:00 (`n=143,406`); `dep_hour` has `r=0.189`, `MI=0.0245`. | **Supported** | GREEN schedule proxy |
| H2 | Carrier effects vary by origin rather than globally | Supported carriers span `12.73%` (`YX`, `n=140,282`) to `28.74%` (`F9`, `n=88,526`), and carrier×hour heatmap shows distinct evening amplification. | **Supported** | YELLOW BTS schedule proxy |
| H3 | Route effects persist after controlling for support | `F2` (`carrier_origin`, `carrier_route`, `origin_hour`, `route_season`) improves `F1` mean PR-AUC (`0.3355` vs `0.3333`) and lifts worst-fold PR-AUC (`0.2626` vs `0.2541` F0). | **Supported** (fold-dependent) | YELLOW |
| H4 | U.S. holiday proximity changes delay prevalence | Exact federal holidays (`0d`, `n=90,143`) have `20.33%` delay rate, lower than `1-3d` (`21.39%`) and `15+d` (`22.07%`, summer-dominated). `F1` lowers `fold_1` (`0.3006` vs `0.3191`) with 1yr history but improves `fold_3`/`fold_4`. | **Partially falsified / mixed** | GREEN calendar |
| H5 | High planned origin load raises delay risk | `P2_PROXY` achieves highest ablation mean PR-AUC (`0.3389`, `+0.0024` vs `F0`, `+0.0038` vs `F3`) and highest worst-fold PR-AUC (`0.2655`). | **Supported (offline proxy only)** | YELLOW retrospective proxy |
| H6 | Airport geography adds information beyond airport IDs | High-elevation `ASE` shows `32.22%` delay (`n=3,218`), `origin_elevation` `MI=0.0031`; `F3` lifts `fold_3` (`0.4067` vs `0.4059` F2) but adds variance on `fold_1`/`fold_4` (`mean_pr_auc=0.3351`). | **Weakly supported / nuanced** | GREEN static reference |
| H7 | Model quality varies more by time than headline mean suggests | Natural prevalence varies from `16.49%` (`fold_4` Q4 2024) to `26.05%` (`2025-06..08`); `unseen_route_rate` rises from `0.23%` to `1.37%`; `F0..F3` fold PR-AUC spans `0.2541` to `0.4067`. | **Supported** | GREEN evaluation protocol |
| H8 | Weather improves ranking on matched coverage | Deferred to Phase 5 (`configs/weather_v2.yaml` + `ecmwf_ifs025` `_previous_day2` matched `2024-04..2025-08`). | Pending Phase 5 | GREEN only when issue time verified |
| H9 | Class weighting improves recall but may harm calibration | Deferred to Phase 3 P5 pilot (`scale_pos_weight` vs unweighted on identical folds). | Pending Phase 3 | GREEN train-fold only |
| H10 | Flow forecasts help only if out-of-fold and T-24h | Deferred to Phase 5 (`scripts/feature_flow_v2.py` hourly lag >=24h/7d). | Pending Phase 5 | YELLOW until source timing proven |
