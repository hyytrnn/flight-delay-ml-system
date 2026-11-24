# EDA Findings



The benchmark contains **480,000 eligible flights** from 2023-01-01 to 2025-01-31.

The overall DepDel15 positive rate is **21.48%**.

The highest sampled monthly delay rate is **29.38%** in 2024-07.

The lowest sampled monthly delay rate is **13.54%** in 2024-10.

Scheduled hour 20:00 has the highest observed hourly rate at **32.65%**.

The largest carrier group is **WN** with 98,956 sampled flights.

The dataset contains **354 origins**, **354 destinations**, and **6,597 routes**.

The train split contains 240,000 rows; test contains 60,000 rows.

All figures describe the bounded monthly benchmark sample, not the full BTS population.

Cancelled, diverted, missing-label, invalid-time and unknown-airport records are excluded under the documented policy and remain visible in the data-quality report.

Post-departure fields are retained only for audit and are not included in the model feature allowlist.

The temporal target-rate chart is used as a drift warning; it is not causal evidence.

Weather ablation remains not evaluated because the complete 2023-2025 window does not have uniform issued-run coverage at T-24h.
