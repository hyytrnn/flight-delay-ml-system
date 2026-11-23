# Milestone 1 Ingestion and Data Contract Report

## Outcome

Milestone 1 is implemented and validated on a committed synthetic fixture. The
pipeline accepts BTS CSV or ZIP inputs, normalizes known source aliases, validates
the schema, localizes scheduled departure using an airport IANA timezone, derives
the T-24h prediction timestamp, and separates valid rows from quarantined rows.

Real BTS ingestion is **NOT EXECUTED** because no BTS CSV or ZIP was supplied.
No EDA statistic, model metric, latency, model artifact or deployment result is
claimed.

## Files created

- `configs/project.yaml`: frozen target, horizon and cleaning policies.
- `src/flight_delay/data/load_bts.py`: CSV/ZIP loading and source-column aliases.
- `src/flight_delay/data/validate.py`: fail-closed schema contracts.
- `src/flight_delay/data/clean.py`: HHMM, timezone, DST, label and quarantine logic.
- `src/flight_delay/features/contracts.py`: explicit leakage blacklist and baseline allowlist.
- `scripts/prepare_data.py`: reproducible CLI with SHA-256 manifest output.
- `scripts/smoke_test.py`: synthetic end-to-end command.
- `tests/`: contract, cleaning, ZIP, DST and leakage tests.
- `docs/`: problem, availability, dictionary, architecture and leakage policies.
- `reports/smoke/`: synthetic-only quality report and generated local Parquet artifacts.

## Commands executed

```powershell
$env:PYTHONPATH='D:\CV\flight-delay-ml-system\src'
& 'D:\CV\.venv\Scripts\python.exe' -m pytest
& 'D:\CV\.venv\Scripts\python.exe' scripts\smoke_test.py
& 'D:\CV\.venv\Scripts\python.exe' -m compileall -q src scripts tests
& 'D:\CV\.venv\Scripts\python.exe' -m pip install --no-build-isolation -e .
```

The available workspace interpreter was Python 3.14.3. The project target remains
Python 3.11 and must be verified in CI or a Python 3.11 environment before release.

## Observed validation evidence

- Pytest: **19 passed in 0.77s**.
- Editable package installation from `pyproject.toml`: completed successfully.
- Final pytest after editable installation: **19 passed in 0.63s**.
- Compileall: completed with exit code 0.
- Synthetic smoke input: 4 rows.
- Synthetic clean output: 2 rows.
- Synthetic quarantine output: 2 rows.
- The smoke report identifies one cancelled/missing-label record and one unknown
  origin record. These values describe only the committed fixture.
- Editor diagnostics after the implementation: no errors found.

## Expected outputs for a real run

- A clean Parquet file containing only eligible binary-label flights.
- A quarantine Parquet file with pipe-delimited reasons for every excluded row.
- A data-quality JSON report with counts, coverage, duplicate and DST statistics.
- A CSV manifest with source metadata, selected columns, byte size and SHA-256.

Exact row counts and rates are unknown until real BTS files are supplied and the
real command is executed.

## Unresolved risks and next actions

1. Supply one to three months of BTS files and a complete, licensed airport
   timezone reference, then inspect the actual column names and distributions.
2. Confirm whether `CRSDepTime=2400` occurs in the selected BTS table and retain
   the documented next-day policy only if supported by the source dictionary.
3. Review quarantined duplicates rather than automatically dropping them.
4. Benchmark ingestion memory on a real month; move from eager concatenation to a
   lazy/month-partitioned path if required.
5. Re-run all tests on Python 3.11 before Milestone 1 is accepted for release.
6. Do not begin EDA or training until the real ingestion report is reviewed and
   the contract tests still pass.

Suggested commit:

```text
feat(data): add leakage-safe BTS ingestion milestone
```
