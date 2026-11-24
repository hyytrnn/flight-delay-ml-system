"""Explainability V2: SHAP TreeExplainer, grouped permutation importance, and Sobol audit."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
from sklearn.metrics import average_precision_score

from flight_delay.features.v2 import FeatureGroup, build_v2_feature_frame, v2_feature_columns
from flight_delay.models.predictor import CatBoostBundleEstimator, load_bundle


def _base_feature_name(encoded_name: str, categorical_cols: list[str]) -> str:
    if "__" in encoded_name:
        prefix, rest = encoded_name.split("__", 1)
        if prefix == "num":
            if rest.startswith("missingindicator_"):
                return rest.replace("missingindicator_", "", 1)
            return rest
        for col in sorted(categorical_cols, key=len, reverse=True):
            if rest == col or rest.startswith(f"{col}_"):
                return col
    return encoded_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("models/flight_delay_v2_bundle.joblib"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--eval-start", type=date.fromisoformat, default=date(2025, 6, 1))
    parser.add_argument("--eval-end-exclusive", type=date.fromisoformat, default=date(2025, 9, 1))
    parser.add_argument("--sample-size", type=int, default=2_000)
    parser.add_argument("--output-root", type=Path, default=Path("reports/explainability"))
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    feature_group: FeatureGroup = bundle.feature_group  # type: ignore[assignment]
    numeric_cols, categorical_cols = v2_feature_columns(feature_group)

    pattern = str(args.data_root / "year=*" / "month=*" / "flights.parquet")
    eval_df = (
        pl.scan_parquet(pattern)
        .filter(
            (pl.col("flight_date") >= args.eval_start)
            & (pl.col("flight_date") < args.eval_end_exclusive)
        )
        .collect(engine="streaming")
    )
    sample_pl = eval_df.sample(
        n=min(args.sample_size, eval_df.height), seed=42, shuffle=True
    )
    x_raw = build_v2_feature_frame(sample_pl, feature_group)
    y_eval = sample_pl.get_column("dep_del15").to_numpy().astype(int)
    probs = bundle.predict_proba(x_raw)

    args.output_root.mkdir(parents=True, exist_ok=True)

    # 1. SHAP TreeExplainer
    if isinstance(bundle.model, CatBoostBundleEstimator):
        x_cb = x_raw.copy()
        for col in categorical_cols:
            x_cb[col] = x_cb[col].astype("string").fillna("UNKNOWN")
        explainer = shap.TreeExplainer(bundle.model.estimator)
        shap_values = np.asarray(explainer.shap_values(x_cb))
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.asarray(base_val).ravel()[-1])
        else:
            base_val = float(base_val)
        grouped_shap = pd.DataFrame(shap_values, columns=x_cb.columns)
        plot_frame = x_cb
    else:
        pre = bundle.model.named_steps["preprocessor"]
        clf = bundle.model.named_steps["classifier"]
        x_enc = pre.transform(x_raw)
        if hasattr(pre, "get_feature_names_out"):
            x_enc_input = x_enc.toarray() if hasattr(x_enc, "toarray") else np.asarray(x_enc)
            enc_names = list(pre.get_feature_names_out())
        else:
            x_enc_input = x_enc
            enc_names = list(x_enc.columns)
        if hasattr(clf, "get_booster"):
            import xgboost as xgb

            dmat = xgb.DMatrix(x_enc_input, enable_categorical=True)
            contribs = np.asarray(clf.get_booster().predict(dmat, pred_contribs=True))
            shap_values = contribs[:, :-1]
            base_val = float(contribs[0, -1])
        else:
            explainer = shap.TreeExplainer(clf)
            shap_values = np.asarray(explainer.shap_values(x_enc_input))
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(np.asarray(base_val).ravel()[-1])
            else:
                base_val = float(base_val)

        # Aggregate encoded columns back to original raw feature groups
        grouped_dict: dict[str, np.ndarray] = {}
        for col_idx, enc_name in enumerate(enc_names):
            base_name = _base_feature_name(enc_name, categorical_cols)
            if base_name not in grouped_dict:
                grouped_dict[base_name] = np.zeros(len(x_raw), dtype=float)
            grouped_dict[base_name] += shap_values[:, col_idx]
        grouped_shap = pd.DataFrame(grouped_dict)
        plot_frame = x_raw

    # Summary bar of grouped raw features
    mean_abs = grouped_shap.abs().mean().sort_values(ascending=False)
    mean_abs.reset_index().rename(
        columns={"index": "feature", 0: "mean_abs_shap"}
    ).to_csv(args.output_root / "shap_grouped_importance.csv", index=False)

    plt.figure(figsize=(8.0, 5.2))
    top_feats = mean_abs.head(15)[::-1]
    plt.barh(top_feats.index, top_feats.values, color="#1f6f8b")
    plt.xlabel("Mean |SHAP value| (log-odds)")
    plt.title(f"Grouped SHAP feature attribution ({bundle.metadata.get('selected_model')}, n={len(x_raw)})")
    plt.tight_layout()
    plt.savefig(args.output_root / "shap_bar.png", dpi=160)
    plt.close()

    # Beeswarm on numeric + top grouped SHAP
    num_in_grouped = [c for c in numeric_cols if c in grouped_shap.columns]
    shap.summary_plot(
        grouped_shap[num_in_grouped].to_numpy(),
        plot_frame[num_in_grouped].astype(float),
        show=False,
        max_display=15,
    )
    plt.title(f"SHAP beeswarm (numeric features, n={len(x_raw)})")
    plt.tight_layout()
    plt.savefig(args.output_root / "shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Dependence plots for dep_hour_sin and distance
    for dep_col in ["dep_hour_sin", "distance"]:
        if dep_col in grouped_shap.columns and dep_col in plot_frame.columns:
            plt.figure(figsize=(6.8, 4.4))
            plt.scatter(
                pd.to_numeric(plot_frame[dep_col], errors="coerce"),
                grouped_shap[dep_col],
                alpha=0.25,
                s=10,
                color="#2e6f95",
            )
            plt.axhline(0.0, linestyle="--", color="gray", alpha=0.6)
            plt.xlabel(dep_col)
            plt.ylabel(f"SHAP contribution ({dep_col})")
            plt.title(f"SHAP dependence: {dep_col}")
            plt.tight_layout()
            plt.savefig(args.output_root / f"shap_dependence_{dep_col}.png", dpi=160)
            plt.close()

    # TP, FP, FN, TN waterfall examples
    preds = (probs >= bundle.threshold).astype(int)
    outcomes = {
        "TP": np.where((y_eval == 1) & (preds == 1))[0],
        "FP": np.where((y_eval == 0) & (preds == 1))[0],
        "FN": np.where((y_eval == 1) & (preds == 0))[0],
        "TN": np.where((y_eval == 0) & (preds == 0))[0],
    }
    for label_type, indices in outcomes.items():
        if len(indices) == 0:
            continue
        idx = int(indices[0])
        exp = shap.Explanation(
            values=grouped_shap.iloc[idx].to_numpy(),
            base_values=base_val,
            data=plot_frame.iloc[idx][grouped_shap.columns].astype(str).to_numpy(),
            feature_names=grouped_shap.columns.tolist(),
        )
        plt.figure(figsize=(8.0, 5.2))
        shap.plots.waterfall(exp, max_display=12, show=False)
        plt.title(f"Waterfall example: {label_type} (p={probs[idx]:.3f}, y={y_eval[idx]})")
        plt.tight_layout()
        plt.savefig(args.output_root / f"shap_waterfall_{label_type}.png", dpi=160, bbox_inches="tight")
        plt.close()

    # 2. Grouped permutation importance (PR-AUC drop on validation sample)
    baseline_pr = float(average_precision_score(y_eval, probs))
    rng = np.random.default_rng(42)
    perm_rows: list[dict[str, object]] = []
    for col in x_raw.columns:
        drops = []
        for _ in range(3):
            x_perm = x_raw.copy()
            x_perm[col] = rng.permutation(x_perm[col].to_numpy())
            perm_pr = float(average_precision_score(y_eval, bundle.predict_proba(x_perm)))
            drops.append(baseline_pr - perm_pr)
        perm_rows.append(
            {
                "feature": col,
                "baseline_pr_auc": baseline_pr,
                "mean_pr_auc_drop": float(np.mean(drops)),
                "std_pr_auc_drop": float(np.std(drops)),
            }
        )
    perm_df = pd.DataFrame(perm_rows).sort_values("mean_pr_auc_drop", ascending=False)
    perm_df.to_csv(args.output_root / "permutation_importance_v2.csv", index=False)

    # 3. Sobol applicability audit (do not fabricate Sobol on correlated mixed categoricals)
    sobol_audit = {
        "status": "NOT_APPLICABLE_FOR_FULL_MODEL",
        "reason": (
            "Standard Sobol variance decomposition assumes independent continuous inputs "
            "over factorizable marginal bounds. V2 F3 combines high-cardinality correlated "
            "categoricals (reporting_airline, origin, dest, carrier_route, origin_hour) "
            "and paired cyclical trig terms (sin/cos) plus airport coordinates/distance, "
            "where independent uniform sampling creates physically impossible flights. "
            "Grouped permutation importance and TreeSHAP are used instead."
        ),
        "sample_size": len(x_raw),
        "eval_window": f"{args.eval_start}..{args.eval_end_exclusive}",
    }
    (args.output_root / "sobol_if_valid.json").write_text(
        json.dumps(sobol_audit, indent=2), encoding="utf-8"
    )
    print(perm_df.head(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
