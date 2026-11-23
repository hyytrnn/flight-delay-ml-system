# Milestone 1 Data Architecture

```text
BTS CSV or ZIP (immutable)         versioned airport reference
            |                                  |
            +---------- checksum manifest -----+
                            |
                       schema aliases
                            |
                  contract and range checks
                            |
             +--------------+----------------+
             |                               |
      clean Parquet                   quarantine Parquet
             |                               |
             +---------- data quality JSON --+
```

The pipeline fails on missing required columns and quarantines invalid row-level
values. Training code is intentionally absent from this milestone so no downstream
consumer can bypass ingestion tests.

