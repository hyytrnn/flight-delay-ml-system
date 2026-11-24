# Locked Test Error Analysis

The analysis covers 60,000 flights from the locked January 2025 test period.
It reports correlations and operational slices, not causal explanations.

- Lowest recall among slices with at least 1,000 rows: `departure_hour=6` at 0.168 across 3,979 flights.
- Largest false-negative count: `distance_bin=short` with 1,005 false negatives.
- Largest false-positive count: `distance_bin=medium` with 11,912 false positives.
- Unknown routes use global median distance and duration; the API exposes this as `route_status=fallback_global_median`.
- Weather-missing slices are not applicable because v1 is explicitly flight-only.

Detailed carrier, origin, scheduled-hour and distance-bin metrics are in `reports/metrics/error_slices.csv`.
