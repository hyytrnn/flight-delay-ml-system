"""Recompute auditable paper claims without treating them as project metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def confusion_metrics(*, tn: int, fp: int, fn: int, tp: int) -> dict[str, float | int]:
    total = tn + fp + fn + tp
    return {
        "rows": total,
        "accuracy": (tn + tp) / total,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def build_audit() -> dict[str, object]:
    figure = confusion_metrics(tn=6050, fp=950, fn=1420, tp=1580)
    table = {"accuracy": 0.856, "roc_auc": 0.801, "f1": 0.64}
    return {
        "source": "Ajala et al. Real-Time Flight Delay Prediction",
        "figure_4_confusion_matrix": figure,
        "table_1_reported": table,
        "absolute_discrepancy": {
            "accuracy": abs(float(figure["accuracy"]) - table["accuracy"]),
            "f1": abs(float(figure["f1"]) - table["f1"]),
        },
        "interpretation": (
            "Source-internal inconsistency or different evaluation subsets; "
            "cause not established and values are not V2 benchmarks."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = build_audit()
    payload = json.dumps(audit, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

