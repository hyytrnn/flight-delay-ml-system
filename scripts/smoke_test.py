"""Run Milestone 1 on the committed synthetic fixture only."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_data.py"),
        "--input",
        str(ROOT / "data" / "samples" / "bts_synthetic.csv"),
        "--airports",
        str(ROOT / "data" / "samples" / "airports_synthetic.csv"),
        "--output",
        str(ROOT / "reports" / "smoke" / "clean_flights.parquet"),
        "--quarantine",
        str(ROOT / "reports" / "smoke" / "quarantine.parquet"),
        "--report",
        str(ROOT / "reports" / "smoke" / "data_quality.json"),
        "--manifest",
        str(ROOT / "reports" / "smoke" / "download_manifest.csv"),
        "--run-kind",
        "synthetic-smoke",
        "--source-url",
        "synthetic://committed-fixture",
        "--license-notes",
        "Synthetic fixture; tests only",
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

