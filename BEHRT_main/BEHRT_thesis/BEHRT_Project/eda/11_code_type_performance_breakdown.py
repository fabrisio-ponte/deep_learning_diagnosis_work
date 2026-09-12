#!/usr/bin/env python3
"""
EDA Step 11: Code-Type Performance Breakdown.

The project's own docs/publication_readiness_summary.md ("Critical Insight #2:
Code type heterogeneity") states 14.4% of CCSR next-visit targets are NOT
disease diagnoses -- they are symptoms (7.0%), administrative/encounter codes
(4.0%), injuries/trauma (2.7%), or pregnancy/congenital conditions (0.7%) --
and explicitly calls for a code-type performance breakdown to support the
"honest framing" limitations section. This has not been built until now
(the closest existing analysis, eda/08, breaks down by LABEL SUPPORT tier,
not by clinical code TYPE).

This script joins per-class test metrics (per_class_metrics.csv, produced by
scripts/train_nextvisit_clean.py) with a CCSR-category -> code-type mapping
(disease / symptom / administrative / injury / pregnancy-and-perinatal /
congenital), then reports:
  - the label-count and train-support-weighted composition of the target
    vocabulary by code type (to verify/refresh the 85.6/7.0/4.0/2.7/0.7% split
    cited in the publication summary against the CURRENT clean run's label
    vocab, which may differ slightly after top_k_labels/min_label_freq
    filtering)
  - mean precision/recall/F1/AUC/AP per code type
  - the same breakdown restricted to TEST-SET-WEIGHTED occurrences (i.e.
    weighting each label's metrics by its test_support), which better reflects
    "how does the model perform on an actual next-visit code, regardless of
    which code type it happens to be" than an unweighted per-label mean.

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3.12 eda/11_code_type_performance_breakdown.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CCSR category (3-letter segment of CCSR_<CAT><NUM>) -> code type.
# Matches the category grouping already used in
# utils/comprehensive_disease_analysis/comprehensive_disease_analysis.py and
# the code-type split cited in docs/publication_readiness_summary.md:
# disease=85.6%, symptom=7.0%, administrative=4.0%, injury=2.7%,
# pregnancy/congenital=0.7%.
CATEGORY_TO_CODE_TYPE = {
    "SYM": "symptom",
    "FAC": "administrative",
    "UTL": "administrative",
    "INJ": "injury",
    "PRG": "pregnancy_and_perinatal",
    "PNL": "pregnancy_and_perinatal",
    "CON": "pregnancy_and_perinatal",
    "MAL": "administrative",  # malingering/factors -- kept out of "disease" bucket
}
DEFAULT_CODE_TYPE = "disease"


def find_latest_run_dir(project_root):
    runs_root = project_root / "data" / "models" / "clean_runs"
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / "per_class_metrics.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No clean_run_* directories with per_class_metrics.csv found under {runs_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def label_category(label):
    """CCSR_END010 -> END. Falls back to the whole label if pattern doesn't match."""
    match = re.match(r"^CCSR_([A-Z]+)\d+$", label)
    return match.group(1) if match else label


def label_code_type(label):
    cat = label_category(label)
    return CATEGORY_TO_CODE_TYPE.get(cat, DEFAULT_CODE_TYPE)


def summarize_code_type(df_type, total_train_support, total_test_support):
    valid_auc = df_type["roc_auc"].dropna()
    return {
        "num_labels": int(len(df_type)),
        "label_count_share": float(len(df_type) / total_train_support[1]),
        "train_support_sum": int(df_type["train_support"].sum()),
        "train_support_share": float(df_type["train_support"].sum() / total_train_support[0]),
        "test_support_sum": int(df_type["test_support"].sum()),
        "test_support_share": float(df_type["test_support"].sum() / total_test_support),
        # Unweighted (per-label) means -- each label counts equally regardless of frequency.
        "mean_precision_per_label": float(df_type["precision"].mean()),
        "mean_recall_per_label": float(df_type["recall"].mean()),
        "mean_f1_per_label": float(df_type["f1"].mean()),
        "mean_average_precision_per_label": float(df_type["average_precision"].mean()),
        "mean_roc_auc_per_label": float(valid_auc.mean()) if len(valid_auc) else float("nan"),
        # Test-support-weighted means -- reflects performance on an actual
        # next-visit occurrence of this code type, not per distinct label.
        "test_support_weighted_precision": float(np.average(df_type["precision"], weights=df_type["test_support"])),
        "test_support_weighted_recall": float(np.average(df_type["recall"], weights=df_type["test_support"])),
        "test_support_weighted_f1": float(np.average(df_type["f1"], weights=df_type["test_support"])),
        "test_support_weighted_average_precision": float(
            np.average(df_type["average_precision"], weights=df_type["test_support"])
        ),
        "labels_with_zero_recall": int((df_type["recall"] == 0).sum()),
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

    df["category"] = df["label"].apply(label_category)
    df["code_type"] = df["label"].apply(label_code_type)

    total_train_support = (int(df["train_support"].sum()), len(df))
    total_test_support = int(df["test_support"].sum())

    code_type_order = ["disease", "symptom", "administrative", "injury", "pregnancy_and_perinatal"]
    code_type_summaries = {}

    print("\n" + "=" * 130)
    print(
        f"{'CodeType':<24} {'#Labels':>8} {'LabelShare':>11} {'TrainSupShare':>14} "
        f"{'TestSupShare':>13} {'wF1':>7} {'wAUC(unwt)':>11} {'wAP':>7} {'ZeroRecall':>11}"
    )
    print("-" * 130)
    for code_type in code_type_order:
        df_type = df[df["code_type"] == code_type]
        if len(df_type) == 0:
            continue
        summary = summarize_code_type(df_type, total_train_support, total_test_support)
        code_type_summaries[code_type] = summary
        print(
            f"{code_type:<24} {summary['num_labels']:>8} {summary['label_count_share']:>11.1%} "
            f"{summary['train_support_share']:>14.1%} {summary['test_support_share']:>13.1%} "
            f"{summary['test_support_weighted_f1']:>7.4f} {summary['mean_roc_auc_per_label']:>11.4f} "
            f"{summary['test_support_weighted_average_precision']:>7.4f} {summary['labels_with_zero_recall']:>11}"
        )
    print("=" * 130)

    # Sanity check against docs/publication_readiness_summary.md's cited split
    # (85.6% disease / 7.0% symptom / 4.0% admin / 2.7% injury / 0.7% preg),
    # which was computed on the FULL target distribution (train_df["label"]
    # occurrences), not on the possibly-filtered label VOCAB used by this run.
    print("\nNote: shares above are computed on THIS run's label vocab/support")
    print("(may differ slightly from the docs/publication_readiness_summary.md")
    print("figures, which were computed on the full unfiltered target distribution).")

    disease_vs_nondisease = {
        "disease_test_support_share": code_type_summaries.get("disease", {}).get("test_support_share"),
        "non_disease_test_support_share": 1.0
        - code_type_summaries.get("disease", {}).get("test_support_share", 0.0),
        "disease_weighted_f1": code_type_summaries.get("disease", {}).get("test_support_weighted_f1"),
        "non_disease_weighted_f1_avg": float(
            np.average(
                df[df["code_type"] != "disease"]["f1"],
                weights=df[df["code_type"] != "disease"]["test_support"],
            )
        )
        if (df["code_type"] != "disease").any()
        else None,
    }
    print(
        f"\nDisease vs non-disease (test-support-weighted F1): "
        f"disease={disease_vs_nondisease['disease_weighted_f1']:.4f} vs "
        f"non-disease={disease_vs_nondisease['non_disease_weighted_f1_avg']:.4f} "
        f"(non-disease share of test occurrences: {disease_vs_nondisease['non_disease_test_support_share']:.1%})"
    )

    result = {
        "run_dir": str(run_dir.relative_to(project_root)),
        "total_labels": int(len(df)),
        "total_train_support": total_train_support[0],
        "total_test_support": total_test_support,
        "category_to_code_type_mapping": CATEGORY_TO_CODE_TYPE,
        "default_code_type": DEFAULT_CODE_TYPE,
        "code_type_summaries": code_type_summaries,
        "disease_vs_non_disease": disease_vs_nondisease,
        "per_label_records": df.to_dict("records"),
    }

    output_dir = project_root / "eda" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "11_code_type_performance_breakdown.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)

    print(f"\nSaved code-type performance breakdown to: {output_path}")


if __name__ == "__main__":
    main()
