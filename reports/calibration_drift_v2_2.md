# Calibration Drift & Reliability Audit (`V2.2`)

- **Model Evaluated**: `flight-only-v2.1.0-xgboost` (`F3`, frozen SHA256 `8b48e8f0b730ac09d11536f4cf7e206962e116a0fcd4f762b02357c59a83d7f6`)
- **Audit Timestamp (UTC)**: `2026-11-25T15:26:05.997657+00:00` (`scripts/audit_calibration_v2_2.py`)
- **Fit / Evaluation Contract**: Calibrators (`platt`, `isotonic`, 1-parameter `temperature_scaling`) were fit strictly on `2025-02-01..2025-03-31` (`200,000` rows) and evaluated forward on `2025-04-01..2025-08-31` (`500,000` rows, `prevalence = 0.23872`). Prospective holdout `2025-10..2025-11` was **excluded** from calibrator fitting and selection.
- **Artifacts**: [calibration_audit_v2_2.json](monitoring/calibration_audit_v2_2.json) · [calibration_v2_2_reliability.png](figures/calibration_v2_2_reliability.png)

## 1. Before / After Calibrator Comparison (`2025-04..08`)

| Method | Brier | `delta_Brier vs none` (95% Block CI) | LogLoss | ECE (`10u`) | ECE (`15u`) | ECE (`20u`) | ECE (`10q`) | Reliability | Resolution | PR-AUC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`none_frozen` (Active)** | `0.16544` | `0.00000` (baseline) | `0.50455` | **`0.01312`** | `0.01312` | `0.01312` | `0.01388` | `0.000214` | `0.015564` | `0.40879` |
| `temperature_T1.007` (`T=1.0072`) | `0.16544` | `-0.000007` `[-0.000025, +0.000010]` | `0.50457` | `0.01316` | `0.01316` | `0.01316` | `0.01379` | `0.000205` | `0.015542` | `0.40879` |
| `platt_feb_mar` | `0.16579` | `+0.000354` `[+0.000203, +0.000536]` | `0.50548` | `0.02033` | `0.02033` | `0.02033` | `0.02111` | `0.000538` | `0.015373` | `0.40879` |
| `isotonic_feb_mar` | `0.16966` | `+0.004205` `[+0.003316, +0.005062]` | `0.60344` | `0.03141` | `0.03142` | `0.03151` | `0.01693` | `0.004243` | `0.015246` | `0.40314` |

**Decision**: Retain `none_frozen`. Neither `platt` nor `isotonic` improved forward calibration (both significantly degraded Brier and ECE across uniform and quantile bin schemes), and 1-parameter temperature scaling converged to `T = 1.0072` with zero practical difference (`delta_Brier = -0.000007`, CI crosses zero).

## 2. Monthly Brier Decomposition & Multi-Bin ECE (`2025-02..08`)

| Month | Rows | Prevalence | ECE (`10u`) | ECE (`15u`) | ECE (`10q`) | Brier | Reliability | Resolution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2025-02` | 100,000 | `0.1949` | `0.01636` | `0.01596` | `0.01646` | `0.15256` | `0.000356` | `0.003839` |
| `2025-03` | 100,000 | `0.1900` | `0.01680` | `0.01681` | `0.01814` | `0.14714` | `0.000469` | `0.006654` |
| `2025-04` | 100,000 | `0.1876` | `0.01707` | `0.01709` | `0.01874` | `0.14609` | `0.000566` | `0.006191` |
| `2025-05` | 100,000 | `0.2246` | `0.01273` | `0.01458` | `0.01421` | `0.16178` | `0.000256` | `0.011732` |
| `2025-06` | 100,000 | `0.2754` | `0.02491` | `0.02606` | `0.02637` | `0.17952` | `0.000923` | `0.019792` |
| `2025-07` | 100,000 | `0.2850` | `0.02465` | `0.02465` | `0.02417` | `0.18183` | `0.000839` | `0.021758` |
| `2025-08` | 100,000 | `0.2210` | `0.01304` | `0.01420` | `0.01260` | `0.15799` | `0.000192` | `0.013656` |

## 3. Subgroup Calibration (`2025-04..08`, Minimum Support `>= 500`)

| Slice | Rows | Prevalence | Mean Pred Prob | ECE (`10u`) | Brier | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `route_known = True` | 497,818 | `0.2388` | `0.2337` | `0.01311` | `0.16544` | `0.40886` |
| `route_known = False` | 2,182 | `0.2314` | `0.2383` | `0.02957` | `0.16529` | `0.39201` |
| `AA` | 69,249 | `0.2876` | `0.2753` | `0.01508` | `0.18597` | `0.45507` |
| `MQ` | 21,275 | `0.2094` | `0.1997` | `0.01196` | `0.15799` | `0.31579` |
| `WN` | 99,492 | `0.2723` | `0.2700` | `0.01997` | `0.17003` | `0.49171` |
| `DL` | 74,935 | `0.2105` | `0.2056` | `0.02075` | `0.15467` | `0.35841` |
| `UA` | 55,728 | `0.2218` | `0.2277` | `0.02087` | `0.15864` | `0.38335` |
| `OO` | 60,659 | `0.1922` | `0.1783` | `0.02275` | `0.14768` | `0.32760` |
| `AS` | 18,187 | `0.2329` | `0.2069` | `0.03334` | `0.17028` | `0.35834` |
| `YX` | 23,209 | `0.1880` | `0.1465` | `0.04585` | `0.14712` | `0.29713` |
| `B6` | 16,435 | `0.2518` | `0.3120` | `0.06076` | `0.17210` | `0.43400` |
| `OH` | 16,825 | `0.2758` | `0.1996` | `0.07626` | `0.19404` | `0.40857` |

While global forward calibration stays well below `0.04` (`0.0127..0.0249`), carrier-level subgroup calibration diverges meaningfully on `OH` (`0.0763`), `B6` (`0.0608`), and `YX` (`0.0459`), explaining why prospective autumn `2025-10..11` monitoring (`ECE = 0.0430`) triggered the retraining alert in `reports/monitoring/v2_1_latest.json`.
