#!/usr/bin/env python3
"""
EDA/Model Step 8: Disease-Level Performance Breakdown.

Joins per-class test metrics (produced by train_nextvisit_clean.py, saved as
per_class_metrics.csv in a clean_run_* directory) with the support-band
categories used in EDA Step 3 (03_label_support_analysis.py), then summarizes
how well the model performs across support tiers (ultra-rare -> very-common).

Answers: does performance degrade for rare diseases, and are there
under/over-performing outliers worth calling out in the thesis?

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3.12 eda/08_disease_level_performance_breakdown.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Same support-tier boundaries as eda/03_label_support_analysis.py, based on
# train-set support so the tiers are consistent across analyses.
SUPPORT_TIERS = [
    ("ultra_rare_lt10", 0, 10),
    ("very_rare_10_99", 10, 100),
    ("rare_100_999", 100, 1000),
    ("common_1k_9k", 1000, 10000),
    ("very_common_gte10k", 10000, float("inf")),
]


def find_latest_run_dir(project_root):
    runs_root = project_root / "data" / "models" / "clean_runs"
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / "per_class_metrics.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No clean_run_* directories with per_class_metrics.csv found under {runs_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def assign_support_tier(train_support):
    for tier_name, lower, upper in SUPPORT_TIERS:
        if lower <= train_support < upper:
            return tier_name
    return "unknown"


def summarize_tier(df_tier):
    valid_auc = df_tier["roc_auc"].dropna()
    return {
        "num_labels": int(len(df_tier)),
        "mean_train_support": float(df_tier["train_support"].mean()),
        "mean_test_support": float(df_tier["test_support"].mean()),
        "mean_precision": float(df_tier["precision"].mean()),
        "mean_recall": float(df_tier["recall"].mean()),
        "mean_f1": float(df_tier["f1"].mean()),
        "mean_average_precision": float(df_tier["average_precision"].mean()),
        "mean_roc_auc": float(valid_auc.mean()) if len(valid_auc) else float("nan"),
        "median_roc_auc": float(valid_auc.median()) if len(valid_auc) else float("nan"),
        "labels_with_zero_recall": int((df_tier["recall"] == 0).sum()),
    }


def main():
    project_root = PROJECT_ROOT
    run_dir_env = os.getenv("RUN_DIR", "")
    run_dir = Path(run_dir_env) if run_dir_env else find_latest_run_dir(project_root)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    csv_path = run_dir / "per_class_metrics.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded per-class metrics from: {csv_path}")
    print(f"Total labels: {len(df)}")

    df["support_tier"] = df["train_support"].apply(assign_support_tier)

    tier_order = [name for name, _, _ in SUPPORT_TIERS]
    tier_summaries = {}
    print("\n" + "=" * 100)
    print(f"{'Tier':<22} {'#Labels':>8} {'MeanTrainSup':>13} {'MeanF1':>8} {'MeanAUC':>8} {'MedAUC':>8} {'Zero-Recall':>12}")
    print("-" * 100)
    for tier_name in tier_order:
        df_tier = df[df["support_tier"] == tier_name]
        if len(df_tier) == 0:
            continue
        summary = summarize_tier(df_tier)
        tier_summaries[tier_name] = summary
        print(
            f"{tier_name:<22} {summary['num_labels']:>8} {summary['mean_train_support']:>13.1f} "
            f"{summary['mean_f1']:>8.4f} {summary['mean_roc_auc']:>8.4f} {summary['median_roc_auc']:>8.4f} "
            f"{summary['labels_with_zero_recall']:>12}"
        )
    print("=" * 100)

    # Outliers: labels that beat their tier's median AUC by a wide margin (best performers
    # despite low support) and labels that badly underperform their tier (worst performers).
    outliers = {}
    for tier_name in tier_order:
        df_tier = df[df["support_tier"] == tier_name].dropna(subset=["roc_auc"])
        if len(df_tier) < 3:
            continue
        top = df_tier.sort_values("roc_auc", ascending=False).head(3)
        bottom = df_tier.sort_values("roc_auc", ascending=True).head(3)
        outliers[tier_name] = {
            "best": top[["label", "train_support", "test_support", "roc_auc", "f1"]].to_dict("records"),
            "worst": bottom[["label", "train_support", "test_support", "roc_auc", "f1"]].to_dict("records"),
        }

    print("\nBest/worst performers per tier (by ROC-AUC):")
    for tier_name, entries in outliers.items():
        print(f"\n  {tier_name}:")
        print("    Best:")
        for row in entries["best"]:
            print(f"      {row['label']:<16} train_n={row['train_support']:<7} AUC={row['roc_auc']:.4f} F1={row['f1']:.4f}")
        print("    Worst:")
        for row in entries["worst"]:
            print(f"      {row['label']:<16} train_n={row['train_support']:<7} AUC={row['roc_auc']:.4f} F1={row['f1']:.4f}")

    result = {
        "run_dir": str(run_dir.relative_to(project_root)),
        "total_labels": int(len(df)),
        "support_tier_definitions": {name: {"min": lower, "max": upper if upper != float("inf") else None} for name, lower, upper in SUPPORT_TIERS},
        "tier_summaries": tier_summaries,
        "outliers_per_tier": outliers,
        "per_label_records": df.to_dict("records"),
    }

    output_dir = project_root / "eda" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "08_disease_level_performance_breakdown.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)

    print(f"\nSaved disease-level performance breakdown to: {output_path}")

    # Headline assessment: does AUC/F1 monotonically improve with support?
    mean_aucs = [tier_summaries[t]["mean_roc_auc"] for t in tier_order if t in tier_summaries]
    monotonic = all(a <= b or np.isnan(a) or np.isnan(b) for a, b in zip(mean_aucs, mean_aucs[1:]))
    print("\nAssessment:")
    print(f"  Mean AUC by tier ({' -> '.join(tier_order)}):")
    print(f"  {[round(v, 4) if not np.isnan(v) else None for v in mean_aucs]}")
    print(f"  Monotonically increasing with support: {monotonic}")


if __name__ == "__main__":
    main()
