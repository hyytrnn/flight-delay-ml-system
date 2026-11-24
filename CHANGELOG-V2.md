# Flight Delay Prediction V2 Changelog

## Phase 0 Baseline Audit

- Created the research branch `feature/v2-research-eda-optuna` from V1 commit
  `079e27e4a729e7828825b36040f7200b82935a59`.
- Froze V1 metrics as historical evidence. January 2025 has already been reviewed
  and is no longer eligible as an untouched V2 holdout.
- Corrected the experiment-log date mismatch without changing historical metrics.
- Added a three-paper transfer review with explicit target, timing, and leakage
  constraints.
- Added a paper-claim audit for the internally inconsistent confusion-matrix and
  summary metrics in the real-time meteorological paper.
- Recorded the execution environment and baseline test result.

No V2 performance claim has been made. Model, feature, tuning, calibration, and
future-holdout experiments remain pending until their gates pass.

