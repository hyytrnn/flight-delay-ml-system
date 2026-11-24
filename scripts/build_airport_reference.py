"""Create a versioned IATA/timezone reference from the pinned airportsdata package."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import airportsdata
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/airports.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    airports = airportsdata.load("IATA")
    rows = []
    package_version = version("airportsdata")
    for iata_code, item in sorted(airports.items()):
        timezone = item.get("tz")
        if not timezone:
            continue
        rows.append(
            {
                "iata_code": iata_code,
                "timezone": timezone,
                "latitude": item.get("lat"),
                "longitude": item.get("lon"),
                "elevation": item.get("elevation"),
                "icao_code": item.get("icao"),
                "name": item.get("name"),
                "city": item.get("city"),
                "subdivision": item.get("subd"),
                "country": item.get("country"),
                "source": f"airportsdata=={package_version}",
                "license": "MIT package; upstream datasets documented by airportsdata",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(args.output)
    print(f"wrote {len(rows)} airports to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

