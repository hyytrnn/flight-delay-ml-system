# V2.1 Monitoring & Drift Report (`2025-06` through `2025-11`)

Reference baseline: `data/processed/v2` (`< 2025-02-01`, `2,500,000` rows, `positive_rate = 0.20408`, `reference_brier = 0.14803`) scored with frozen bundle `flight-only-v2.1.0-xgboost`.

## 1. Monthly Snapshot Progression

| Window | Rows | Positive Rate | `PR-AUC` | `ROC-AUC` | `Brier` (`delta`) | `ECE` | `score_psi` | `unseen_route_rate` | Alert Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `2025-06` ([2025-06.json](file:///D:/CV/flight-delay-ml-system/reports/monitoring/history/2025-06.json)) | 100,000 | `0.2754` | `0.4578` | `0.7080` | `0.1795` (`+0.0315`) | `0.0249` | `0.2036` | `0.0049` | `brier_increase_warning` (seasonal surge) |
| `2025-07` ([2025-07.json](file:///D:/CV/flight-delay-ml-system/reports/monitoring/history/2025-07.json)) | 100,000 | `0.2850` | `0.4688` | `0.7124` | `0.1818` (`+0.0338`) | `0.0246` | `0.3206` | `0.0056` | `brier_increase_warning` (seasonal surge) |
| `2025-08` ([2025-08.json](file:///D:/CV/flight-delay-ml-system/reports/monitoring/history/2025-08.json)) | 100,000 | `0.2210` | `0.3803` | `0.6959` | `0.1580` (`+0.0100`) | `0.0130` | `0.0292` | `0.0069` | All clear (`retraining_recommended = false`) |
| `2025-10` ([2025-10.json](file:///D:/CV/flight-delay-ml-system/reports/monitoring/history/2025-10.json)) | 100,000 | `0.1959` | `0.3384` | `0.6870` | `0.1491` (`+0.0011`) | **`0.0417`** | `0.2931` | `0.0067` | **`ece_warning`** |
| `2025-11` ([2025-11.json](file:///D:/CV/flight-delay-ml-system/reports/monitoring/history/2025-11.json)) | 100,000 | `0.1980` | `0.3122` | `0.6648` | `0.1532` (`+0.0051`) | **`0.0442`** | `0.4913` | `0.0071` | **`ece_warning`** |
| **`2025-10..11`** ([v2_1_latest.json](file:///D:/CV/flight-delay-ml-system/reports/monitoring/v2_1_latest.json)) | **200,000** | **`0.1969`** | **`0.3265`** | **`0.6759`** | **`0.1511` (`+0.0031`)** | **`0.0430`** | **`0.3562`** | **`0.0069`** | **`ece_warning` (`retraining_recommended = true`)** |

## 2. Feature Drift (`2025-10..11` Latest Snapshot)

- `distance` PSI: `0.00046`
- `origin_elevation` PSI: `0.00148`
- `crs_elapsed_time` PSI: `0.00226`
- `unseen_route_rate`: `0.00691` (`0.69%`, well below `0.03` warning and `0.06` critical cutoffs)
- **Predicted Score Distribution (`2025-10..11`)**: `mean = 0.1562`, `p50 = 0.1445`, `p90 = 0.2518`, `p95 = 0.2860`; alert rate `26.71%` at `0.185` (`max_f1`) and `5.11%` at `0.285` (`precision_target_35`).
- **Key Diagnostic Takeaway**: Schedule covariates (`distance`, `crs_elapsed_time`, `origin_elevation`) stayed almost stationary (`PSI <= 0.0023`), proving covariate monitoring alone would miss the autumn calibration shift (`ECE = 0.0430`) triggered once delayed BTS `DepDel15` labels arrived. That validates why [monitoring.yaml](file:///D:/CV/flight-delay-ml-system/configs/monitoring.yaml) pairs covariate PSI with delayed-label ECE/Brier monitoring.
