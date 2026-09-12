#!/usr/bin/env python3
"""
Error / confusion analysis for BEHRT NextVisit predictions.

Goes beyond aggregate precision/recall (per_class_metrics.csv,
eda/08_disease_level_performance_breakdown.py) to answer: WHICH true
diagnoses drive false-positive predictions for a given code, and WHICH
diagnoses does the model over-predict instead when it misses (false
negative) a given true code?

For each label L, among test rows where L is falsely predicted positive
(false positive), we measure how often each OTHER true label T co-occurs in
that same row, versus T's baseline occurrence rate across all test rows.
The ratio (lift) highlights labels that are disproportionately associated
with L's false positives, i.e. likely confusions rather than coincidence.

We also aggregate confusions at the CCSR CATEGORY level (e.g. CIR, END,
DIG - the 3-letter segment of "CCSR_<CAT><NUM>") to see whether errors are
mostly within clinically-related categories (informative, expected) or
scattered across unrelated categories (noise).

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3.12 eda/09_error_confusion_analysis.py
"""

import json
import os
import re
import sys
from collections import defaultdict
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

TOP_N_CONFUSIONS = 5
MIN_FP_COUNT = 10  # only analyze labels with at least this many false positives
MIN_COOCCUR_COUNT = 3  # only report a confusion pair if it co-occurs at least this often


def find_latest_run_dir(project_root):
    runs_root = project_root / "data" / "models" / "clean_runs"
    candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No clean_run_* directories with metrics.json found under {runs_root}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def label_category(label):
    """CCSR_END010 -> END. Falls back to the whole label if pattern doesn't match."""
    match = re.match(r"^CCSR_([A-Z]+)\d+$", label)
    return match.group(1) if match else label


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
    threshold = float(run_metrics["threshold_tuning"]["selected_threshold"])

    train_path = data_dir / run_metrics["data"]["train"]
    test_path = data_dir / run_metrics["data"]["test"]
    vocab_path = data_dir / run_metrics["data"]["vocab"].replace(".pkl", "")

    train_df = pd.read_parquet(train_path)
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

    test_set = NextVisit(
        token2idx=bert_vocab["token2idx"],
        label2idx=label_vocab,
        age2idx=age_vocab_dict,
        dataframe=test_df,
        max_len=max_len_seq,
    )
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    mlb = MultiLabelBinarizer(classes=list(label_vocab.values()))
    mlb.fit([[x] for x in list(label_vocab.values())])

    conf = BertConfig(model_config)
    model = BertForMultiLabelPrediction(
        conf, num_labels=len(label_vocab), feature_dict={"word": True, "seg": True, "age": True, "position": True}
    )
    model.load_state_dict(torch.load(run_dir / "behrt_nextvisit_ccsr_clean_best.pt", map_location=device))
    model = model.to(device)

    print("Recomputing test-set predictions...")
    y_true, y_prob = collect_eval_arrays(model, test_loader, mlb, device)

    idx_to_label = {idx: label for label, idx in label_vocab.items()}
    if metric_exclude_labels:
        excluded_indices = {int(label_vocab[label]) for label in metric_exclude_labels if label in label_vocab}
        keep_indices = [idx for idx in range(y_true.shape[1]) if idx not in excluded_indices]
        y_true = y_true[:, keep_indices]
        y_prob = y_prob[:, keep_indices]
        idx_to_label = {new_idx: idx_to_label[old_idx] for new_idx, old_idx in enumerate(keep_indices)}

    y_pred = (y_prob >= threshold).astype(bool)
    y_true_bool = y_true.astype(bool)
    num_labels = y_true.shape[1]
    all_labels = [idx_to_label[i] for i in range(num_labels)]
    label_baseline_rate = y_true_bool.mean(axis=0)  # P(label present) across all test rows

    print(f"Threshold used for predictions: {threshold:.2f}")
    print(f"Analyzing {num_labels} labels for false-positive confusions...")

    label_confusions = {}
    for l_idx in range(num_labels):
        fp_mask = y_pred[:, l_idx] & ~y_true_bool[:, l_idx]
        fp_count = int(fp_mask.sum())
        if fp_count < MIN_FP_COUNT:
            continue

        # Among FP rows for this label, how often is each OTHER true label present?
        cooccur_counts = y_true_bool[fp_mask].sum(axis=0)  # shape (num_labels,)
        cooccur_rate = cooccur_counts / fp_count

        candidates = []
        for t_idx in range(num_labels):
            if t_idx == l_idx:
                continue
            count = int(cooccur_counts[t_idx])
            if count < MIN_COOCCUR_COUNT:
                continue
            baseline = label_baseline_rate[t_idx]
            lift = (cooccur_rate[t_idx] / baseline) if baseline > 0 else float("inf")
            candidates.append(
                {
                    "confused_with_label": idx_to_label[t_idx],
                    "cooccur_count": count,
                    "cooccur_rate_given_fp": float(cooccur_rate[t_idx]),
                    "baseline_rate": float(baseline),
                    "lift": float(lift),
                }
            )

        candidates.sort(key=lambda row: row["lift"], reverse=True)
        top_candidates = candidates[:TOP_N_CONFUSIONS]

        own_label = idx_to_label[l_idx]
        own_category = label_category(own_label)
        same_category_count = sum(
            1 for c in candidates if label_category(c["confused_with_label"]) == own_category
        )

        label_confusions[own_label] = {
            "label": own_label,
            "category": own_category,
            "fp_count": fp_count,
            "train_support": int(train_supports.get(own_label, 0)),
            "num_candidate_confusions": len(candidates),
            "same_category_confusions": same_category_count,
            "top_confusions": top_candidates,
        }

    # Category-level aggregation: of all (label -> top confusion) pairs, what fraction
    # are within the same clinical category vs cross-category?
    total_top_pairs = 0
    same_category_top_pairs = 0
    category_confusion_counts = defaultdict(int)
    for entry in label_confusions.values():
        for c in entry["top_confusions"]:
            total_top_pairs += 1
            src_cat = entry["category"]
            dst_cat = label_category(c["confused_with_label"])
            if src_cat == dst_cat:
                same_category_top_pairs += 1
            category_confusion_counts[f"{src_cat}->{dst_cat}"] += 1

    same_category_fraction = (same_category_top_pairs / total_top_pairs) if total_top_pairs else float("nan")

    # Sort labels by FP count (biggest offenders first) for the printed summary.
    sorted_labels = sorted(label_confusions.values(), key=lambda row: row["fp_count"], reverse=True)

    print("=" * 110)
    print(f"Top false-positive offenders and their strongest confusions (lift = cooccur_rate / baseline_rate):")
    print("-" * 110)
    for entry in sorted_labels[:20]:
        print(f"\n{entry['label']:<16} (train_n={entry['train_support']:<7} FP_count={entry['fp_count']:<6} same_cat={entry['same_category_confusions']}/{entry['num_candidate_confusions']})")
        for c in entry["top_confusions"]:
            print(
                f"    -> {c['confused_with_label']:<16} lift={c['lift']:>7.2f}  "
                f"P(present|FP)={c['cooccur_rate_given_fp']:.3f}  baseline={c['baseline_rate']:.4f}  n={c['cooccur_count']}"
            )
    print("=" * 110)

    print(f"\nOverall: {same_category_top_pairs}/{total_top_pairs} top confusion pairs ({same_category_fraction:.1%}) are WITHIN the same CCSR category")
    print("Top cross-category confusion category-pairs:")
    top_category_pairs = sorted(category_confusion_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    for pair, count in top_category_pairs:
        print(f"  {pair:<12} n={count}")

    result = {
        "run_dir": str(run_dir.relative_to(project_root)),
        "threshold": threshold,
        "min_fp_count": MIN_FP_COUNT,
        "min_cooccur_count": MIN_COOCCUR_COUNT,
        "top_n_confusions": TOP_N_CONFUSIONS,
        "num_labels_analyzed": len(label_confusions),
        "same_category_fraction_of_top_pairs": same_category_fraction,
        "category_confusion_pair_counts": dict(category_confusion_counts),
        "label_confusions": label_confusions,
    }

    output_dir = project_root / "eda" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "09_error_confusion_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved error/confusion analysis to: {output_path}")


if __name__ == "__main__":
    main()
