#!/usr/bin/env python3
"""
EDA Step 7: Cleaning Impact Analysis
Compares `label` (post-cleaning) vs `label_original` (pre-cleaning) to quantify
what the cleaning step removed, and checks whether it distorted class imbalance.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "eda" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "07_cleaning_impact_analysis.json"

DATASETS = {
    "train": DATA_DIR / "train_nextvisit_ccsr_clean.parquet",
    "val": DATA_DIR / "val_nextvisit_ccsr_clean.parquet",
    "test": DATA_DIR / "test_nextvisit_ccsr_clean.parquet",
}

# Cross-reference against cleaning/clean_data.py's documented removal strategy:
# codes are removed if they are (1) generic 'XXX' CCSR codes, (2) high-frequency
# (>5%) generic codes, or (3) match suspicious patterns '000', '999', 'UNK'.
# See BEHRT_Project/cleaning/clean_data.py docstring and _identify_problematic_codes().
KNOWN_REMOVAL_PATTERNS = ["XXX", "000", "999", "UNK"]


def matches_known_removal_strategy(code: str) -> bool:
    return any(pattern in code for pattern in KNOWN_REMOVAL_PATTERNS)


def imbalance_summary(label_counter: Counter, total_samples: int) -> dict:
    if not label_counter:
        return {
            "unique_labels": 0,
            "max_count": 0,
            "min_count": 0,
            "imbalance_ratio": 0.0,
        }
    counts = list(label_counter.values())
    max_count = max(counts)
    min_count = min(counts)
    return {
        "unique_labels": int(len(label_counter)),
        "max_count": int(max_count),
        "min_count": int(min_count),
        "imbalance_ratio": float(max_count / min_count) if min_count > 0 else float("inf"),
        "top_5": [
            {"label": label, "count": int(count), "fraction_of_samples": float(count / total_samples)}
            for label, count in label_counter.most_common(5)
        ],
    }


def analyze_split(split_name: str, df: pd.DataFrame) -> dict:
    total_samples = len(df)

    has_original = "label_original" in df.columns
    if not has_original:
        return {
            "samples": int(total_samples),
            "has_label_original": False,
            "note": "label_original column not present; cannot compute cleaning diff for this split.",
        }

    removed_counter = Counter()
    added_counter = Counter()
    per_row_removed_counts = np.zeros(total_samples, dtype=int)
    rows_with_removal = 0
    rows_with_addition = 0

    label_after_counter = Counter()
    label_before_counter = Counter()

    for i, (before, after) in enumerate(zip(df["label_original"], df["label"])):
        before_set = set(before)
        after_set = set(after)

        removed = before_set - after_set
        added = after_set - before_set

        if removed:
            rows_with_removal += 1
            for code in removed:
                removed_counter[code] += 1
        if added:
            rows_with_addition += 1
            for code in added:
                added_counter[code] += 1

        per_row_removed_counts[i] = len(removed)

        for code in before:
            label_before_counter[code] += 1
        for code in after:
            label_after_counter[code] += 1

    total_labels_before = sum(label_before_counter.values())
    total_labels_after = sum(label_after_counter.values())

    result = {
        "samples": int(total_samples),
        "has_label_original": True,
        "rows_with_any_code_removed": int(rows_with_removal),
        "pct_rows_with_any_code_removed": float(rows_with_removal / total_samples * 100),
        "rows_with_any_code_added": int(rows_with_addition),
        "pct_rows_with_any_code_added": float(rows_with_addition / total_samples * 100),
        "avg_codes_removed_per_row": float(np.mean(per_row_removed_counts)),
        "total_label_instances_before": int(total_labels_before),
        "total_label_instances_after": int(total_labels_after),
        "pct_label_instances_removed": float(
            (total_labels_before - total_labels_after) / total_labels_before * 100
        ) if total_labels_before > 0 else 0.0,
        "top_10_codes_removed": [
            {"label": code, "times_removed": int(count)}
            for code, count in removed_counter.most_common(10)
        ],
        "removed_codes_matching_known_strategy": [
            {"label": code, "times_removed": int(count)}
            for code, count in removed_counter.most_common()
            if matches_known_removal_strategy(code)
        ],
        "removed_codes_not_matching_known_strategy": [
            {"label": code, "times_removed": int(count)}
            for code, count in removed_counter.most_common()
            if not matches_known_removal_strategy(code)
        ],
        "unexpected_added_codes": [
            {"label": code, "times_added": int(count)}
            for code, count in added_counter.most_common(10)
        ],
        "imbalance_before_cleaning": imbalance_summary(label_before_counter, total_samples),
        "imbalance_after_cleaning": imbalance_summary(label_after_counter, total_samples),
    }
    return result


def main() -> None:
    print("=" * 80)
    print("CLEANING IMPACT ANALYSIS (label vs label_original)")
    print("=" * 80)

    split_results = {}
    for split_name, path in DATASETS.items():
        print(f"\n--- {split_name.upper()} ---")
        df = pd.read_parquet(path)
        result = analyze_split(split_name, df)
        split_results[split_name] = result

        if not result.get("has_label_original", False):
            print(f"  {result.get('note')}")
            continue

        print(f"Samples: {result['samples']:,}")
        print(f"Rows with >=1 code removed: {result['rows_with_any_code_removed']:,} "
              f"({result['pct_rows_with_any_code_removed']:.2f}%)")
        print(f"Rows with unexpected code added: {result['rows_with_any_code_added']:,} "
              f"({result['pct_rows_with_any_code_added']:.2f}%)")
        print(f"Avg codes removed per row: {result['avg_codes_removed_per_row']:.3f}")
        print(f"Label instances removed: "
              f"{result['total_label_instances_before'] - result['total_label_instances_after']:,} "
              f"({result['pct_label_instances_removed']:.2f}%)")
        print(f"Imbalance ratio before cleaning: {result['imbalance_before_cleaning']['imbalance_ratio']:.1f}:1")
        print(f"Imbalance ratio after cleaning:  {result['imbalance_after_cleaning']['imbalance_ratio']:.1f}:1")
        print("Top 10 codes removed by cleaning:")
        for item in result["top_10_codes_removed"]:
            print(f"  {item['label']}: removed {item['times_removed']:,} times")

        not_matching = result["removed_codes_not_matching_known_strategy"]
        if not_matching:
            print(f"  NOTE: {len(not_matching)} removed code(s) do NOT match the documented "
                  f"XXX/000/999/UNK removal strategy in clean_data.py - review these:")
            for item in not_matching[:10]:
                print(f"    {item['label']}: removed {item['times_removed']:,} times")
        else:
            print("  All removed codes match the documented removal strategy "
                  "(generic XXX / high-freq generic / 000-999-UNK patterns).")

        if result["unexpected_added_codes"]:
            print("  WARNING: cleaning added codes that were not in label_original:")
            for item in result["unexpected_added_codes"]:
                print(f"    {item['label']}: added {item['times_added']:,} times")

    results = {
        "step": 7,
        "title": "Cleaning Impact Analysis",
        "splits": split_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Cleaning impact analysis saved to: {OUTPUT_PATH}")
    print(f"{'=' * 80}")

    # Human-readable assessment across splits (train is most informative, largest N)
    train_result = split_results.get("train", {})
    if train_result.get("has_label_original"):
        any_added = any(
            split_results[s].get("rows_with_any_code_added", 0) > 0 for s in split_results
        )
        print("\nAssessment:")
        if any_added:
            print("  WARNING: cleaning is not purely subtractive - unexpected codes were added.")
        else:
            print("  Cleaning is purely subtractive across all splits (no unexpected additions).")
        before_ratio = train_result["imbalance_before_cleaning"]["imbalance_ratio"]
        after_ratio = train_result["imbalance_after_cleaning"]["imbalance_ratio"]
        print(f"  Train imbalance ratio: {before_ratio:.1f}:1 (before) -> {after_ratio:.1f}:1 (after)")
        if abs(before_ratio - after_ratio) / before_ratio < 0.1:
            print("  Interpretation: cleaning did not materially change class imbalance.")
        else:
            print("  Interpretation: cleaning meaningfully changed class imbalance; review top removed codes.")


if __name__ == "__main__":
    main()
