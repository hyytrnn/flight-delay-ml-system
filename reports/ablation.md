# Ablation and Experiment Status

| Experiment | Status | Evidence |
|---|---|---|
| E0 Dummy prior | Executed | Validation PR-AUC 0.2617 |
| E1 Schedule and route features with logistic regression | Executed | Validation PR-AUC 0.3931 |
| E2 Nonlinear schedule and route model | Executed | HistGradientBoosting PR-AUC 0.3969; XGBoost 0.3974 |
| E3 Weather forecast as-of | Not evaluated | Uniform issued-run coverage is unavailable across 2023-2025 |
| E4 Calibration and threshold | Executed | Dedicated October 2024 calibration; locked-test threshold 0.135 and F1 0.3336 |
| E5 Weather provider fallback | Not applicable to v1 | Flight-only model does not require the provider |

The locked-test PR-AUC is 0.2213 versus a 0.1825 positive-rate reference. The
drop from validation is retained as evidence of temporal shift; no test-driven
retuning was performed.

