"""Download immutable monthly BTS Reporting Carrier on-time ZIP archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import BadZipFile, ZipFile

BASE_URL = "https://www.transtats.bts.gov/PREZIP"


@dataclass(frozen=True)
class Month:
    year: int
    month: int

    @property
    def filename(self) -> str:
        return (
            "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
            f"{self.year}_{self.month}.zip"
        )


def parse_month(value: str) -> Month:
    year_text, month_text = value.split("-", maxsplit=1)
    result = Month(int(year_text), int(month_text))
    if not 1 <= result.month <= 12:
        raise argparse.ArgumentTypeError(f"Invalid month: {value}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_zip(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None and any(
                name.lower().endswith(".csv") for name in archive.namelist()
            )
    except (BadZipFile, OSError):
        return False


def download(month: Month, root: Path) -> dict[str, object]:
    destination = root / str(month.year) / f"{month.month:02d}" / month.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{month.filename}"

    if not valid_zip(destination):
        partial = destination.with_suffix(".zip.part")
        request = urllib.request.Request(url, headers={"User-Agent": "flight-delay-ml-system/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if not valid_zip(partial):
            raise RuntimeError(f"Downloaded archive failed ZIP validation: {url}")
        partial.replace(destination)

    return {
        "filename": destination.name,
        "source_url": url,
        "flight_year": month.year,
        "flight_month": month.month,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "license_notes": "U.S. DOT BTS public data; verify current BTS terms",
        "path": destination.as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", type=parse_month, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/bts"))
    parser.add_argument("--manifest", type=Path, default=Path("data/download_manifest.json"))
    args = parser.parse_args()
    records = [download(month, args.output_root) for month in args.months]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

