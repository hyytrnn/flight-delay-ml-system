# Chronological Split Strategy

The benchmark uses calendar boundaries fixed before training:

| Partition | Included benchmark months | Purpose |
|---|---|---|
| Train | January, April, July and October 2023 | Fit preprocessing and estimators |
| Validation | January and July 2024 | Select the model family |
| Calibration | October 2024 | Fit Platt calibration and select the F1 threshold |
| Locked test | January 2025 | Final evaluation after every decision is frozen |

The source months are a bounded portfolio benchmark, not the complete BTS
population. Each month is deterministically capped during preparation to control
memory while retaining chronological separation. No random train-test split is
used and the test period is never used for model selection or threshold tuning.

