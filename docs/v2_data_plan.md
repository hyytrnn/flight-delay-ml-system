# V2 Continuous Data Plan

## Objective

Replace the eight isolated V1 months with continuous monthly partitions while
preserving raw-file immutability, natural label prevalence, per-month provenance,
and a future holdout that remains closed until V2 decisions are frozen.

## Planned coverage

- Development data: January 2023 through August 2025.
- Historical V1 reference: January 2025, excluded from V2 model selection claims.
- Untouched final holdout: September 2025.
- Raw data source: official BTS Reporting Carrier monthly ZIP archives.
- Initial laptop budget: at most 100,000 eligible rows per development month.
- Full-row mode remains available for a later compute-controlled run.

## Storage contract

Processed flights are written by `year=YYYY/month=MM` and never require all raw
months to coexist in memory. The monthly manifest records source URL, SHA256,
bytes, schema, raw/eligible/sample/quarantine counts, seed, and the difference
between sampled and eligible positive rates.

## Evaluation contract

Model and feature selection use the expanding-window folds in
`configs/splits_v2.yaml`. Calibration, threshold selection, pre-holdout validation,
and final holdout use non-overlapping calendar windows. January 2025 remains a
historical V1 regression reference. September 2025 is evaluated once only after
the feature set, model, calibration method, and threshold policy are frozen.

## Commands

```powershell
python scripts/download_bts.py --start-month 2023-01 --end-month 2025-09 `
  --manifest data/manifests/bts_v2_download.json --dry-run

python scripts/prepare_dataset_v2.py --start-month 2023-01 --end-month 2025-08 `
  --max-rows-per-month 100000
```

The final-holdout month is downloaded and checksum-verified separately but is not
included in EDA or development artifacts before the final-once gate.

