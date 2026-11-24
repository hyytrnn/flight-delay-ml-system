"""Audit historical V2 Optuna studies and maintain clean V2.1 study storage."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from flight_delay.models.tuning_v2 import cleanup_stale_running_trials


def inspect_sqlite_studies(db_path: Path) -> list[dict[str, object]]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        studies = conn.execute("SELECT study_id, study_name FROM studies ORDER BY study_id").fetchall()
        report: list[dict[str, object]] = []
        for study_id, study_name in studies:
            rows = conn.execute(
                "SELECT state, COUNT(*) FROM trials WHERE study_id = ? GROUP BY state",
                (study_id,),
            ).fetchall()
            counts = {str(state): int(cnt) for state, cnt in rows}
            running_nums = [
                int(r[0])
                for r in conn.execute(
                    "SELECT number FROM trials WHERE study_id = ? AND state = 'RUNNING'",
                    (study_id,),
                ).fetchall()
            ]
            report.append(
                {
                    "study_name": str(study_name),
                    "state_counts": counts,
                    "running_trial_numbers": running_nums,
                }
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-db", type=Path, default=Path("reports/optuna/v2_study.db"))
    parser.add_argument(
        "--v2-1-db", type=Path, default=Path("reports/optuna_v2_1/v2_1_study.db")
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("reports/optuna_v2_1/stale_trials_audit.json"),
    )
    args = parser.parse_args()

    v2_1_storage = f"sqlite:///{args.v2_1_db.as_posix()}"
    args.v2_1_db.parent.mkdir(parents=True, exist_ok=True)
    cleaned_v2_1 = cleanup_stale_running_trials(v2_1_storage)

    audit = {
        "historical_v2_db": str(args.v2_db),
        "historical_v2_studies": inspect_sqlite_studies(args.v2_db),
        "historical_v2_db_modified": False,
        "v2_1_db": str(args.v2_1_db),
        "v2_1_stale_running_cleaned": cleaned_v2_1,
        "v2_1_studies": inspect_sqlite_studies(args.v2_1_db),
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
