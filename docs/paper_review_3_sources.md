# Three Paper Transfer Review

## Review objective

The papers provide research hypotheses and engineering patterns. None of their
reported scores transfers directly to the U.S. BTS `DepDel15` T-24h benchmark.
Every proposed transfer below requires the same temporal folds, target, feature
availability contract, and matched flight population as its baseline.

## Paper A Multiple data sources and explainable ML

Pineda-Jaramillo et al. study 68,206 flights departing Santiago in 2017 and retain
67,730 after cleaning. Their target is arrival delay at touchdown over 15 minutes,
with 11.7% positives. They combine operations, destination-airport metadata,
geography, and observed hourly weather at takeoff and landing. SMOTE-ENN and
stratified k-fold validation are used. LDA is selected for recall; gradient
boosting has the highest reported ROC-AUC. SHAP and Sobol analyze model behavior.

Transferable ideas:

- Versioned airport type, coordinates, elevation, distance, calendar, and peak
  period features.
- Explicit model selection objective rather than assuming boosting always wins.
- SHAP and grouped permutation importance for explanation and stability checks.
- Sobol only for defensible numeric domains; not for arbitrary one-hot categories.

Non-transferable without a new experiment:

- Arrival target, observed weather, SCL-specific feature rankings, SMOTE-ENN,
  random/k-fold scores, and all reported metric values.

## Paper B ADS-B air traffic flow

Gui et al. aggregate ADS-B messages into hourly counts for 240 routes and predict a
continuous flow series using SVR and an LSTM with 24 time steps. They report RMSE
38.31 for SVR and 29.73 for LSTM on their route-flow task. The target is not flight
delay classification.

Transferable ideas:

- Planned origin/destination/route load by hour and calendar context.
- A separate flow-forecast task whose out-of-fold forecast may become a classifier
  feature when it is available by T-24h.
- Flow seasonality and abnormal-change diagnostics.

Constraints:

- Realized ADS-B positions after prediction time are forbidden.
- The paper's fixed threshold above 1000 for abnormal flow is dataset-specific and
  must not be copied.
- The LSTM result supports a flow-regression hypothesis, not a delay-classifier
  accuracy claim.

## Paper C Real-time delay and meteorological intelligence

Ajala et al. describe five million U.S. domestic flights, 51 origin/destination
weather variables, an arrival-delay target, stratified 80/20 split, XGBoost with
SMOTE, magnitude regression, and a prototype live architecture. The paper reports
85.6% accuracy, ROC-AUC 0.801, and F1 0.64 in Table I.

The published Figure 4 confusion matrix contains TN=6050, FP=950, FN=1420,
TP=1580. Those counts imply accuracy 0.763, precision 0.6245, recall 0.5267, and
F1 0.5714. The discrepancy may reflect different experiments or an illustrative
figure; its cause is not established.

Transferable ideas:

- Matched-sample origin and destination issued-forecast weather ablation.
- Separate delay-probability and optional departure-delay-magnitude tasks.
- Calibration, stakeholder-specific thresholds, provenance, fallback, monitoring,
  and shadow-mode evaluation.

Constraints:

- Arrival target and stratified split differ from V2.
- Observed or current weather cannot replace forecasts available at T-24h.
- SMOTE is optional and train-fold only; PR-AUC, Brier, calibration, and alert
  burden must be evaluated together.

## Experiment map

| Paper claim or method | V2 hypothesis | Required evidence | Current status |
|---|---|---|---|
| Multi-source airport/geography | Static airport features improve temporal PR-AUC | F0 vs F3 on identical folds | NOT EVALUATED |
| Peak period and time of day | U.S. calendar features improve worst-fold PR-AUC | F0 vs F1 | NOT EVALUATED |
| ADS-B hourly flow | Planned congestion proxy improves flight classifier | P1 vs P2 with as-of provenance | NOT EVALUATED |
| LSTM beats SVR for flow | A flow submodel can add useful T-24h information | OOF flow RMSE then classifier ablation | NOT EVALUATED |
| Dual-airport weather | Issued forecasts improve matched-sample PR-AUC | Same flights, same dates, coverage and issue-time audit | NOT EVALUATED |
| SMOTE/class weighting | Reweighting improves operational tradeoff | Train-fold-only comparison with Brier and PR-AUC | NOT EVALUATED |
| SHAP/Sobol | Explanations are stable across folds | Grouped SHAP/permutation; Sobol if valid | NOT EVALUATED |

