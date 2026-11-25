# Operational Threshold Policy (`T-24h` Flight Delay Risk)

This document summarizes the canonical alerting and calibration policy for `flight-delay-ml-system` V2.1 (detailed evidence in [threshold_policy_v2_1.md](file:///D:/CV/flight-delay-ml-system/docs/threshold_policy_v2_1.md) and [error_analysis_v2_1.md](file:///D:/CV/flight-delay-ml-system/docs/error_analysis_v2_1.md)).

## Decision Regimes

- **Naive Max-F1 warning**: Do not deploy a pure spring max-F1 cutoff (`0.185` on OOF/spring validation) without an alert-rate or precision floor — under summer prevalence (`~26–29%`), it flags `62.4%` of flights (`122,622` false alerts per `300,000` flights).
- **Default cost-balanced / precision-targeted threshold**: Target a ~`35%` spring precision floor (`0.285` in the forward `2025-04..05` -> `2025-06..08` study), achieving `0.4304` precision, `0.5568` recall, `0.4855` F1, `33.7%` alert rate, `57,577` false alerts (>50% reduction), and near-optimal `2:1` expected cost (`0.42280` vs minimum `0.42256` at `0.290`).
- **Fixed-capacity ranking modes**:
  - Top 10% budget: `0.5254` precision in peak summer `2025-06..08`.
  - Top 5% budget: `0.5598` precision in peak summer `2025-06..08`.
- **High-cost contingency regimes**:
  - `4:1` missed-delay penalty: optimal threshold `0.200` (min cost `0.5858/flight`).
  - `8:1` missed-delay penalty: optimal threshold `0.125` (min cost `0.6984/flight`).

## Calibration Stability Policy

- Retain the frozen base bundle calibrator across shoulder months instead of aggressive 1-month online Platt refits (`2025-03` ECE rose `0.0168 -> 0.0321` under 1-month lag; `2025-05` rose `0.0127 -> 0.0298`).
- Monitor monthly ECE, PSI, and precision at the deployed operating point continuously via `scripts/monitor.py`.
