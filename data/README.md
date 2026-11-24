# Data Directory

- `raw/`: immutable BTS files organized by year and month; ignored by Git.
- `interim/`: quarantined or partially processed data; ignored by Git.
- `processed/`: validated Parquet outputs; ignored by Git.
- `external/`: versioned airport metadata with source and license notes.
- `samples/`: small synthetic fixtures only, never benchmark data.

The real benchmark downloads these official Reporting Carrier archives:

- 2023: January, April, July and October
- 2024: January, July and October
- 2025: January

The months provide chronological and seasonal coverage while keeping a portfolio
run feasible on a laptop. Raw ZIP files remain ignored by Git. The JSON manifest
records official URLs, byte sizes, selected fields and SHA-256 checksums.

