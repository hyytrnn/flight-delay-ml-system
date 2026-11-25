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


def month_range(start: Month, end: Month) -> list[Month]:
    start_index = start.year * 12 + start.month - 1
    end_index = end.year * 12 + end.month - 1
    if end_index < start_index:
        raise ValueError("end month must not be before start month")
    return [Month(index // 12, index % 12 + 1) for index in range(start_index, end_index + 1)]


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


def remote_size(url: str) -> int:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "flight-delay-ml-system/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def download_with_resume(url: str, partial: Path, expected_bytes: int) -> None:
    """Resume after BTS closes a response before Content-Length is reached."""

    for _ in range(12):
        current_bytes = partial.stat().st_size if partial.exists() else 0
        if current_bytes == expected_bytes:
            return
        if current_bytes > expected_bytes:
            partial.unlink()
            current_bytes = 0
        headers = {"User-Agent": "flight-delay-ml-system/1.0"}
        if current_bytes:
            headers["Range"] = f"bytes={current_bytes}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            if current_bytes and response.status != 206:
                partial.unlink(missing_ok=True)
                continue
            mode = "ab" if current_bytes else "wb"
            with partial.open(mode) as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        if partial.stat().st_size == expected_bytes:
            return
    observed = partial.stat().st_size if partial.exists() else 0
    raise RuntimeError(
        f"Download remained incomplete after resume attempts: {url} "
        f"({observed}/{expected_bytes} bytes)"
    )


def download(month: Month, root: Path) -> dict[str, object]:
    destination = root / str(month.year) / f"{month.month:02d}" / month.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{month.filename}"

    if not valid_zip(destination):
        partial = destination.with_suffix(".zip.part")
        download_with_resume(url, partial, remote_size(url))
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


def plan(month: Month, root: Path) -> dict[str, object]:
    destination = root / str(month.year) / f"{month.month:02d}" / month.filename
    return {
        "flight_year": month.year,
        "flight_month": month.month,
        "source_url": f"{BASE_URL}/{month.filename}",
        "path": destination.as_posix(),
        "exists": destination.exists(),
        "valid_zip": valid_zip(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", type=parse_month)
    parser.add_argument("--start-month", type=parse_month)
    parser.add_argument("--end-month", type=parse_month)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/bts"))
    parser.add_argument("--manifest", type=Path, default=Path("data/download_manifest.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.months and (args.start_month or args.end_month):
        parser.error("use either --months or --start-month/--end-month")
    if args.months:
        months = args.months
    elif args.start_month and args.end_month:
        months = month_range(args.start_month, args.end_month)
    else:
        parser.error("provide --months or both --start-month and --end-month")

    records = (
        [plan(month, args.output_root) for month in months]
        if args.dry_run
        else [download(month, args.output_root) for month in months]
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

