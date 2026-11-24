# Flight Only Model Card

## Model overview

`flight-only-v1.0.0` is an XGBoost binary classifier that estimates the
probability of BTS `DepDel15` at T-24h. It includes its preprocessing pipeline,
Platt calibrator, decision threshold, route reference, feature contract and
training metadata in one trusted joblib artifact.

## Intended use

The model demonstrates leakage-safe tabular ML engineering for portfolio review.
It may support exploratory early-warning workflows. It must not be used for
safety-critical, financial or passenger-facing decisions without broader data,
prospective validation and operational governance.

## Training data

- Official BTS Reporting Carrier monthly archives.
- Train: four sampled months in 2023, 240,000 rows.
- Validation: January and July 2024, 120,000 rows.
- Calibration: October 2024, 60,000 rows.
- Locked test: January 2025, 60,000 rows.
- Cancelled, diverted, missing-label and invalid records are excluded under the
  documented policy.

## Features

Carrier, origin, destination, scheduled route distance, scheduled elapsed time,
cyclical scheduled departure hour, weekday, month and weekend flag. All transforms
are implemented in the shared feature builder used by training and serving.

## Model selection

| Candidate | Validation PR-AUC | Validation ROC-AUC |
|---|---:|---:|
| Dummy prior | 0.2617 | 0.5000 |
| Logistic regression | 0.3931 | 0.6557 |
| HistGradientBoosting | 0.3969 | 0.6569 |
| XGBoost | 0.3974 | 0.6586 |

XGBoost was selected by validation PR-AUC. Calibration and threshold selection
used only the dedicated October 2024 partition.

## Locked test performance

| Metric | Value |
|---|---:|
| PR-AUC | 0.2213 |
| ROC-AUC | 0.5841 |
| Precision | 0.2108 |
| Recall | 0.7989 |
| F1 | 0.3336 |
| Brier score | 0.1483 |
| Log loss | 0.4708 |
| Threshold | 0.135 |

The test positive rate is 0.1825. PR-AUC is therefore 0.0388 above a constant
prior reference, but substantially below validation PR-AUC. This temporal drop is
a key limitation and evidence that the benchmark should not be presented as a
production-ready predictor.

## Error analysis

The lowest recall among slices with at least 1,000 rows occurs at scheduled hour
06:00 (0.168). Short-distance flights contain the largest false-negative count,
while medium-distance flights contain the largest false-positive count. These are
associations, not causal explanations.

## Weather and fallback

Weather is not used. Open-Meteo issued-run coverage is not uniform across the
complete benchmark window, so ERA5 or stitched historical forecasts are not used
as substitutes. Unknown API routes use global median distance and scheduled
duration and return `route_status=fallback_global_median`.

## Risks and limitations

- BTS schedule publication timestamps are unavailable; T-24h schedule availability
  is a retrospective proxy.
- The benchmark samples eight months rather than every month in 2023-2025.
- No live flight schedule feed or prospective production labels are integrated.
- Airport, carrier, route and seasonal shifts can reduce performance.
- The joblib artifact must only be loaded from this trusted repository because
  pickle-based formats can execute code.

