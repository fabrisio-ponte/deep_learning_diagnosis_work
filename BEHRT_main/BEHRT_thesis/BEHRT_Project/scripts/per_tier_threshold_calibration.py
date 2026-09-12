#!/usr/bin/env python3
"""
Per-support-tier threshold calibration.

Motivation (from eda/08_disease_level_performance_breakdown.py): a single
global decision threshold (tuned to maximize overall micro-F1, which is
dominated by common labels) drives recall to zero for ~30% of labels
(139/469), even though rarer tiers actually have comparable or *better*
mean ROC-AUC than common tiers. This means the model has ranking signal for
rare diseases that a single global threshold silently discards.

This script calibrates one threshold PER SUPPORT TIER (same tiers as
eda/03 and eda/08: ultra_rare_lt10, very_rare_10_99, rare_100_999,
common_1k_9k, very_common_gte10k) instead of one threshold for all 469
labels, using only the VALIDATION set to select thresholds (avoiding test
leakage), then evaluates both the single-global-threshold baseline and the
per-tier thresholds on the held-out TEST set for a fair comparison.

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3.12 scripts/per_tier_threshold_calibration.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit
from model.NextXVisit import BertForMultiLabelPrediction

from scripts.train_nextvisit_clean import (
    BertConfig,
    build_label_subset,
    build_label_support_counts,
    collect_eval_arrays,
    format_label_vocab,
)

# Same support-tier boundaries as eda/03_label_support_analysis.py and
# eda/08_disease_level_performance_breakdown.py.
SUPPORT_TIERS = [
    ("ultra_rare_lt10", 0, 10),
    ("very_rare_10_99", 10, 100),
    ("rare_100_999", 100, 1000),
    ("common_1k_9k", 1000, 10000),
    ("very_common_gte10k", 10000, float("inf")),
]
TIER_ORDER = [name for name, _, _ in SUPPORT_TIERS]
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.05)


def find_latest_run_dir(project_root):
    runs_root = project_root / "data" / "models" / "clean_runs"
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No clean_run_* directories with metrics.json found under {runs_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def assign_tier(train_support):
    for tier_name, lower, upper in SUPPORT_TIERS:
        if lower <= train_support < upper:
            return tier_name
    return "unknown"


def tier_label_indices(label_vocab, train_supports):
    """Map each tier name to the list of class indices (columns of y_true/y_prob)
    belonging to that tier, based on train-set support."""
    tiers = {name: [] for name in TIER_ORDER}
    for label, idx in label_vocab.items():
        tier = assign_tier(train_supports.get(label, 0))
        if tier in tiers:
            tiers[tier].append(idx)
    return tiers


def micro_prf1_for_indices(y_true, y_prob, indices, threshold):
    if not indices:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    sel_true = y_true[:, indices].astype(bool)
    sel_pred = (y_prob[:, indices] >= threshold).astype(bool)
    tp = int(np.sum(sel_true & sel_pred))
    fp = int(np.sum(~sel_true & sel_pred))
    fn = int(np.sum(sel_true & ~sel_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def zero_recall_count(y_true, y_prob, indices, threshold):
    count = 0
    for idx in indices:
        true_col = y_true[:, idx].astype(bool)
        support = int(true_col.sum())
        if support == 0:
            continue
        pred_col = (y_prob[:, idx] >= threshold).astype(bool)
        tp = int(np.sum(true_col & pred_col))
        if tp == 0:
            count += 1
    return count


def select_tier_threshold(y_true_val, y_prob_val, indices):
    """Sweep THRESHOLD_GRID on the validation subset for this tier's label
    columns, aggregate TP/FP/FN across all labels in the tier at each
    threshold, and return the threshold maximizing tier-level micro-F1."""
    best = {"threshold": 0.5, "f1": -1.0}
    history = []
    for threshold in THRESHOLD_GRID:
        stats = micro_prf1_for_indices(y_true_val, y_prob_val, indices, float(threshold))
        history.append({"threshold": float(threshold), **stats})
        if stats["f1"] > best["f1"]:
            best = {"threshold": float(threshold), "f1": stats["f1"]}
    return best["threshold"], history


def main():
    project_root = PROJECT_ROOT
    data_dir = project_root / "data" / "processed"

    run_dir_env = os.getenv("RUN_DIR", "")
    run_dir = Path(run_dir_env) if run_dir_env else find_latest_run_dir(project_root)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
        run_metrics = json.load(f)

    print(f"Loaded run: {run_dir}")

    model_config = run_metrics["model_config"]
    top_k_labels = int(run_metrics["run_controls"].get("top_k_labels", 0) or 0)
    min_label_freq = float(run_metrics["run_controls"].get("min_label_freq", 0.0) or 0.0)
    metric_exclude_labels = tuple(run_metrics["run_controls"].get("metric_exclude_labels", ["UNK"]))
    global_threshold = float(run_metrics["threshold_tuning"]["selected_threshold"])

    train_path = data_dir / run_metrics["data"]["train"]
    val_path = data_dir / run_metrics["data"]["val"]
    test_path = data_dir / run_metrics["data"]["test"]
    vocab_path = data_dir / run_metrics["data"]["vocab"].replace(".pkl", "")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    bert_vocab = load_obj(str(vocab_path))
    age_vocab_dict, _ = age_vocab(max_age=110, symbol=None)
    base_label_vocab = format_label_vocab(bert_vocab["token2idx"])

    label_vocab, train_df, _ = build_label_subset(
        train_df, base_label_vocab, top_k_labels=top_k_labels, min_label_freq=min_label_freq
    )
    train_supports = build_label_support_counts(train_df, label_vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len_seq = model_config["max_position_embedding"]

    mlb = MultiLabelBinarizer(classes=list(label_vocab.values()))
    mlb.fit([[x] for x in list(label_vocab.values())])

    def make_loader(df):
        dataset = NextVisit(
            token2idx=bert_vocab["token2idx"],
            label2idx=label_vocab,
            age2idx=age_vocab_dict,
            dataframe=df,
            max_len=max_len_seq,
        )
        return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    val_loader = make_loader(val_df)
    test_loader = make_loader(test_df)

    conf = BertConfig(model_config)
    model = BertForMultiLabelPrediction(
        conf, num_labels=len(label_vocab), feature_dict={"word": True, "seg": True, "age": True, "position": True}
    )
    model.load_state_dict(torch.load(run_dir / "behrt_nextvisit_ccsr_clean_best.pt", map_location=device))
    model = model.to(device)

    print("Recomputing validation-set predictions (for threshold selection)...")
    y_true_val, y_prob_val = collect_eval_arrays(model, val_loader, mlb, device)
    print("Recomputing test-set predictions (for final evaluation)...")
    y_true_test, y_prob_test = collect_eval_arrays(model, test_loader, mlb, device)

    if metric_exclude_labels:
        excluded_indices = {int(label_vocab[label]) for label in metric_exclude_labels if label in label_vocab}
        keep_indices = [idx for idx in range(y_true_val.shape[1]) if idx not in excluded_indices]
        y_true_val = y_true_val[:, keep_indices]
        y_prob_val = y_prob_val[:, keep_indices]
        y_true_test = y_true_test[:, keep_indices]
        y_prob_test = y_prob_test[:, keep_indices]
        idx_remap = {old: new for new, old in enumerate(keep_indices)}
        label_vocab = {label: idx_remap[idx] for label, idx in label_vocab.items() if idx in idx_remap}

    tiers = tier_label_indices(label_vocab, train_supports)

    print("Selecting per-tier thresholds on the validation set...")
    tier_thresholds = {}
    tier_val_sweeps = {}
    for tier_name in TIER_ORDER:
        indices = tiers[tier_name]
        if not indices:
            continue
        threshold, history = select_tier_threshold(y_true_val, y_prob_val, indices)
        tier_thresholds[tier_name] = threshold
        tier_val_sweeps[tier_name] = history

    print("=" * 100)
    print(f"{'Tier':<22} {'#Labels':>8} {'SelectedThreshold':>18}")
    print("-" * 100)
    for tier_name in TIER_ORDER:
        if tier_name not in tier_thresholds:
            continue
        print(f"{tier_name:<22} {len(tiers[tier_name]):>8} {tier_thresholds[tier_name]:>18.2f}")
    print("=" * 100)

    # Evaluate on TEST set: (a) single global threshold baseline, (b) per-tier thresholds.
    print("\nTest-set comparison: global threshold vs per-tier thresholds")
    print("-" * 100)
    header = f"{'Tier':<22} {'#Labels':>8} {'GlobalF1':>10} {'GlobalRec':>10} {'GlobalZeroRec':>14} {'TierF1':>10} {'TierRec':>10} {'TierZeroRec':>12}"
    print(header)
    print("-" * 100)

    tier_comparison = {}
    for tier_name in TIER_ORDER:
        indices = tiers.get(tier_name, [])
        if not indices:
            continue
        global_stats = micro_prf1_for_indices(y_true_test, y_prob_test, indices, global_threshold)
        global_zero = zero_recall_count(y_true_test, y_prob_test, indices, global_threshold)

        tier_threshold = tier_thresholds[tier_name]
        tier_stats = micro_prf1_for_indices(y_true_test, y_prob_test, indices, tier_threshold)
        tier_zero = zero_recall_count(y_true_test, y_prob_test, indices, tier_threshold)

        print(
            f"{tier_name:<22} {len(indices):>8} {global_stats['f1']:>10.4f} {global_stats['recall']:>10.4f} "
            f"{global_zero:>14} {tier_stats['f1']:>10.4f} {tier_stats['recall']:>10.4f} {tier_zero:>12}"
        )

        tier_comparison[tier_name] = {
            "num_labels": len(indices),
            "global_threshold": global_threshold,
            "global_metrics": global_stats,
            "global_zero_recall_labels": global_zero,
            "tier_threshold": tier_threshold,
            "tier_metrics": tier_stats,
            "tier_zero_recall_labels": tier_zero,
        }

    # Overall micro-F1 across ALL labels: global threshold everywhere vs per-tier thresholds.
    all_indices = list(range(y_true_test.shape[1]))
    overall_global = micro_prf1_for_indices(y_true_test, y_prob_test, all_indices, global_threshold)

    # Build a per-label threshold vector for the "per-tier" scheme, then compute overall stats.
    label_threshold = np.full(y_true_test.shape[1], global_threshold, dtype=float)
    for tier_name, indices in tiers.items():
        if tier_name in tier_thresholds:
            for idx in indices:
                label_threshold[idx] = tier_thresholds[tier_name]

    y_pred_tiered = (y_prob_test >= label_threshold[np.newaxis, :]).astype(bool)
    y_true_bool = y_true_test.astype(bool)
    tp = int(np.sum(y_true_bool & y_pred_tiered))
    fp = int(np.sum(~y_true_bool & y_pred_tiered))
    fn = int(np.sum(y_true_bool & ~y_pred_tiered))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    overall_tiered = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    total_global_zero = zero_recall_count(y_true_test, y_prob_test, all_indices, global_threshold)
    total_tiered_zero = sum(
        zero_recall_count(y_true_test, y_prob_test, [idx], label_threshold[idx]) for idx in all_indices
    )

    print("-" * 100)
    print(
        f"{'OVERALL (all labels)':<22} {len(all_indices):>8} {overall_global['f1']:>10.4f} "
        f"{overall_global['recall']:>10.4f} {total_global_zero:>14} {overall_tiered['f1']:>10.4f} "
        f"{overall_tiered['recall']:>10.4f} {total_tiered_zero:>12}"
    )
    print("=" * 100)

    result = {
        "run_dir": str(run_dir.relative_to(project_root)),
        "global_threshold": global_threshold,
        "tier_thresholds": tier_thresholds,
        "tier_validation_sweeps": tier_val_sweeps,
        "tier_test_comparison": tier_comparison,
        "overall_test_comparison": {
            "global_threshold_everywhere": {**overall_global, "zero_recall_labels": total_global_zero},
            "per_tier_thresholds": {**overall_tiered, "zero_recall_labels": total_tiered_zero},
        },
    }

    output_path = run_dir / "per_tier_threshold_calibration.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved per-tier threshold calibration results to: {output_path}")


if __name__ == "__main__":
    main()
