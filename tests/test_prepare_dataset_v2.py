from __future__ import annotations

from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl
import pytest

from flight_delay.models.tuning_v2 import load_development_frame
from scripts.prepare_dataset_v2 import discover_monthly_files, prepare_month


def _write_bts_zip(path: Path, flight_date: str, rows: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "FlightDate,Reporting_Airline,Flight_Number_Reporting_Airline,Origin,Dest,"
        "CRSDepTime,CRSArrTime,CRSElapsedTime,Distance,DepDel15,Cancelled,Diverted\n"
    )
    lines = [
        f"{flight_date},AA,{idx + 1},JFK,LAX,{800 + idx},1100,360,2475,{idx % 2},0,0\n"
        for idx in range(rows)
    ]
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("flights.csv", header + "".join(lines))


def test_discover_monthly_files_rejects_missing_months(tmp_path: Path) -> None:
    _write_bts_zip(tmp_path / "2024" / "01" / "jan.zip", "2024-01-05")
    _write_bts_zip(tmp_path / "2024" / "03" / "mar.zip", "2024-03-05")

    with pytest.raises(FileNotFoundError, match="2024-02"):
        discover_monthly_files(tmp_path, (2024, 1), (2024, 3))

    discovered = discover_monthly_files(tmp_path, (2024, 1), (2024, 3), allow_missing=True)
    assert [(year, month) for year, month, _ in discovered] == [(2024, 1), (2024, 3)]


def test_discover_monthly_files_rejects_duplicate_partitions(tmp_path: Path) -> None:
    _write_bts_zip(tmp_path / "2024" / "01" / "part_a.zip", "2024-01-05")
    _write_bts_zip(tmp_path / "2024" / "01" / "part_b.zip", "2024-01-06")

    with pytest.raises(ValueError, match="duplicate monthly partitions"):
        discover_monthly_files(tmp_path, (2024, 1), (2024, 1))


def test_discover_monthly_files_rejects_excluded_holdout(tmp_path: Path) -> None:
    _write_bts_zip(tmp_path / "2025" / "08" / "aug.zip", "2025-08-05")
    _write_bts_zip(tmp_path / "2025" / "09" / "sep.zip", "2025-09-05")

    with pytest.raises(ValueError, match="2025-09"):
        discover_monthly_files(
            tmp_path, (2025, 8), (2025, 9), excluded_months={(2025, 9)}
        )


def test_prepare_month_reports_sampling_rate_and_excludes_holdout_from_dev_load(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    aug_zip = raw_root / "2025" / "08" / "aug.zip"
    sep_zip = raw_root / "2025" / "09" / "sep.zip"
    _write_bts_zip(aug_zip, "2025-08-10", rows=10)
    _write_bts_zip(sep_zip, "2025-09-10", rows=6)

    airports = pl.DataFrame(
        {
            "iata_code": ["JFK", "LAX"],
            "timezone": ["America/New_York", "America/Los_Angeles"],
            "latitude": [40.64, 33.94],
            "longitude": [-73.78, -118.41],
            "elevation": [13.0, 125.0],
            "country": ["US", "US"],
            "subdivision": ["New York", "California"],
        }
    )
    out_root = tmp_path / "processed"
    quarantine_root = tmp_path / "interim"

    aug_record = prepare_month(
        year=2025,
        month=8,
        path=aug_zip,
        airports=airports,
        output_root=out_root,
        quarantine_root=quarantine_root,
        max_rows=6,
        seed=42,
    )
    prepare_month(
        year=2025,
        month=9,
        path=sep_zip,
        airports=airports,
        output_root=out_root,
        quarantine_root=quarantine_root,
        max_rows=6,
        seed=42,
    )

    assert aug_record["raw_rows"] == 10
    assert aug_record["eligible_rows"] == 10
    assert aug_record["sample_rows"] == 6
    assert aug_record["eligible_positive_rate"] == pytest.approx(0.5)
    assert aug_record["positive_rate_delta"] == pytest.approx(
        float(aug_record["sample_positive_rate"]) - 0.5
    )

    dev = load_development_frame(out_root, date(2025, 9, 1))
    assert dev.height == 6
    assert dev.get_column("flight_date").max() < date(2025, 9, 1)


def test_discover_monthly_files_rejects_prospective_v2_1_holdouts(tmp_path: Path) -> None:
    from scripts.prepare_dataset_v2 import HOLDOUT_EXCLUDED_MONTHS

    _write_bts_zip(tmp_path / "2025" / "08" / "aug.zip", "2025-08-05")
    _write_bts_zip(tmp_path / "2025" / "10" / "oct.zip", "2025-10-05")
    _write_bts_zip(tmp_path / "2025" / "11" / "nov.zip", "2025-11-05")

    with pytest.raises(ValueError, match="2025-10"):
        discover_monthly_files(
            tmp_path, (2025, 8), (2025, 11), excluded_months=HOLDOUT_EXCLUDED_MONTHS
        )
