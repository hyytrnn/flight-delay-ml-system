# V2.1 Operational Threshold Policy Study

Generated from `scripts/run_eda_error_v2_1.py` by fitting threshold policies on spring forward validation (`2025-04..05`) and evaluating out-of-sample on peak summer forward validation (`2025-06..08`, `300,000` flights). Prospective holdout (`2025-10..11`) and `2025-09` were never accessed during policy selection.

## 1. Policy Comparison (`2025-04..05` Selection -> `2025-06..08` Summer Forward)

Source artifact: `reports/eda_v2_1/threshold_policy_comparison.csv`

| Policy | Threshold (Selected on `2025-04..05`) | Spring `F1` (`2025-04..05`) | Summer Precision (`2025-06..08`) | Summer Recall (`2025-06..08`) | Summer `F1` (`2025-06..08`) | Summer Alert Rate | Summer False Alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_max_f1` | `0.185` | `0.4074` | `0.3453` | `0.8276` | `0.4873` | `0.624` | `122,622` |
| `precision_target_35` | `0.285` | — | `0.4304` | `0.5568` | `0.4855` | `0.337` | `57,577` |
| `top_10pct_budget` | — (rank top 10%) | — | `0.5254` | — | — | `0.100` | — |
| `top_5pct_budget` | — (rank top 5%) | — | `0.5598` | — | — | `0.050` | — |

Key operational finding:
- Under naive `fixed_max_f1` (`0.185`), a seasonal surge in summer base prevalence pushes the alert rate to **`62.4%`** of all flights and produces **`122,622` false alerts** across `300,000` summer flights while delivering virtually identical `F1` (`0.4873`) as `0.285` (`0.4855`).
- Raising the static decision threshold to `0.285` (`precision_target_35`) cuts false alerts by more than half (`122,622 -> 57,577`), reduces alert volume from `62.4%` to `33.7%`, lifts precision from `0.3453` to `0.4304`, and preserves `F1 = 0.4855`.

## 2. Asymmetric Cost Curve Analysis (`2025-06..08`)

Source artifact: `reports/eda_v2_1/cost_curve_v2_1.csv`

Expected cost per flight was swept from threshold `0.050` to `0.950` under three missed-delay vs. false-alert cost ratios (`C_FN : C_FP`):

| Cost Regime (`C_FN : C_FP`) | Optimal Threshold | Min Expected Cost / Flight | Notes |
| --- | ---: | ---: | --- |
| `cost_2_to_1` (`2:1`) | `0.290` | `0.4226` (`0.42256`) | At `0.290`: precision `0.4343`, recall `0.5415`, F1 `0.4820`, alert rate `0.3248`. At `0.285`: cost is `0.42280` (within `0.00024` per flight). |
| `cost_4_to_1` (`4:1`) | `0.200` | `0.5858` | Appropriate when missed-delay recovery costs 4x an advisory intervention. |
| `cost_8_to_1` (`8:1`) | `0.125` | `0.6984` | High-recall emergency staffing mode when missed delays carry severe penalty. |

## 3. Production & Budget-Constrained Policy Recommendation

1. **Precision-constrained default binary threshold (`>= 0.285` or V2.1 bundle-calibrated equivalent in `0.28–0.30` / ~`0.29`)**: Protects ops/traveler trust by avoiding >60% summer alert saturation and sits within `0.00024/flight` of the global `2:1` asymmetric expected-cost optimum.
2. **Capacity-capped alert queues (top-5% / top-10% score budgets)**: When gate-hold or proactive customer outreach capacity is fixed regardless of season, trigger on the top 10% (`precision = 0.5254` in summer) or top 5% (`precision = 0.5598` in summer) ranked probabilities rather than a fixed probability cutoff.
3. **Never tune threshold on test/holdout months**: The final threshold frozen into `models/flight_delay_v2_1_bundle.joblib` in Gate G7 will be derived purely on `2025-02..08` / OOF predictions prior to opening `2025-10..11`.
